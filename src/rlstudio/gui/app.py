"""Interface graphique : gestion de projets RL, edition de l'environnement, entrainement."""
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import numpy as np
import pyqtgraph as pg
import yaml
from PySide6.QtGui import QColor, QIcon
from PySide6.QtCore import QSize, QTimer, Qt, QThread, Signal, Slot, QObject
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from ..envs.spec import validate_spec
from ..paths import ICON_FILE
from ..project import Project, ProjectStore
from ..settings import Settings
from ..training import ALGOS, fmt_duration
from . import theme
from .theme import C
from .. import runtime
from .distill_tab import DistillTab
from .identify_tab import IdentifyTab
from .eval_tab import EvalTab
from .explorer import SimulinkExplorer
from .live_plots import SeriesPlot
from .runner import RuntimeMissing, TrainRunner, WorkerJob
from .settings_dialog import SettingsDialog
from .widgets import Card, Kpi, button, column, fmt_run_date, label, scroll_page

# colonnes des tables Simulink : (cle, titre, type)
ACTION_COLS = [("name", "Nom", str), ("block", "Bloc", str), ("param", "Parametre", str),
               ("variable", "ou Variable", str), ("low", "Min", float), ("high", "Max", float)]
OBS_COLS = [("name", "Nom", str), ("block", "Bloc (sortie)", str), ("port", "Port", int),
            ("size", "Taille", int), ("low", "Min", float), ("high", "Max", float)]
INIT_COLS = [("variable", "Variable", str), ("low", "Min", float), ("high", "Max", float)]


LOG_MAX_LINES = 5000


class SpecTable(QTableWidget):
    """Table editable <-> liste de dicts."""

    changed = Signal()

    def __init__(self, cols, defaults=None):
        super().__init__(0, len(cols))
        self.cols, self.defaults = cols, defaults or {}
        self.setHorizontalHeaderLabels([c[1] for c in cols])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(32)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setShowGrid(False)
        self.setMinimumHeight(130)
        self.setMaximumHeight(210)
        widths = {"name": 120, "block": 210, "param": 110, "variable": 110, "port": 60, "size": 60,
                  "low": 80, "high": 80}
        for j, (key, _, _) in enumerate(cols[:-1]):
            self.setColumnWidth(j, widths.get(key, 100))
        self._loading = False
        self.cellChanged.connect(lambda *_: None if self._loading else self.changed.emit())

    def set_rows(self, rows):
        self._loading = True
        self.setRowCount(0)
        for r in rows or []:
            self.add_row(r, emit=False)
        self._loading = False

    def add_row(self, values=None, emit=True):
        values = {**self.defaults, **(values or {})}
        was = self._loading
        self._loading = True
        row = self.rowCount()
        self.insertRow(row)
        for j, (key, _, _) in enumerate(self.cols):
            self.setItem(row, j, QTableWidgetItem("" if values.get(key) is None else str(values[key])))
        self._loading = was
        if emit:
            self.changed.emit()

    def remove_selected(self):
        for idx in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(idx)
        self.changed.emit()

    def rows(self):
        out = []
        for i in range(self.rowCount()):
            d = {}
            for j, (key, _, typ) in enumerate(self.cols):
                txt = (self.item(i, j).text() if self.item(i, j) else "").strip()
                if txt == "":
                    continue
                try:
                    d[key] = typ(float(txt)) if typ is int else typ(txt)
                except ValueError:
                    d[key] = txt  # la validation signalera la valeur invalide
            out.append(d)
        return out


# -- fenetre principale ------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, store=None, settings=None):
        super().__init__()
        self.settings = settings or Settings()
        self._plots = []
        self.theme_name = theme.DEFAULT_THEME
        self._apply_theme(self.settings.get("theme", theme.DEFAULT_THEME))
        self.store = store or ProjectStore(root=self.settings.projects_dir)
        self.project = None
        self.dirty = False
        self._loading = False
        self.worker = None
        self._rew_timer = QTimer(self)
        self._rew_timer.setSingleShot(True)
        self._rew_timer.setInterval(40)
        self._rew_timer.timeout.connect(self._flush_rew)
        self.setWindowTitle("RL Studio")
        self.resize(1440, 900)
        if ICON_FILE.exists():  # icone de la fenetre, de la barre des taches et des boites de dialogue
            self._icon = QIcon(str(ICON_FILE))
            QApplication.instance().setWindowIcon(self._icon)
            self.setWindowIcon(self._icon)
        self._build_ui()
        self.setAcceptDrops(True)
        self._refresh_list()

    # -- themes -------------------------------------------------------------------------
    def _all_themes(self):
        return {**theme.BUILTIN, **(self.settings.get("custom_themes") or {})}

    def _apply_theme(self, name, save=False, base=None):
        """Applique un theme par nom (et le memorise si save) ou directement un dict `base` (apercu)."""
        if base is None:
            themes = self._all_themes()
            if name not in themes:
                name = theme.DEFAULT_THEME
            base = themes[name]
            self.theme_name = name
            if save:
                self.settings.set("theme", name)
        theme.apply(QApplication.instance(), base)
        if self._plots:
            self._style_plots()
            if self.runs_list.currentItem():
                self._show_run(self.runs_list.currentItem(), None)

    # -- parametres ---------------------------------------------------------------------
    def _open_settings(self, page=0):
        if self.worker:
            QMessageBox.information(self, "Parametres", "Arrete l'entrainement en cours avant de modifier les parametres.")
            return
        page = page if isinstance(page, int) and not isinstance(page, bool) else 0
        dlg = SettingsDialog(self, self.settings, self._apply_theme, self._all_themes, self.theme_name)
        dlg.nav.setCurrentRow(page)
        if dlg.exec() == QDialog.Accepted and dlg.projects_dir_changed:
            if self.dirty:
                self._save()
            self.store = ProjectStore(root=self.settings.projects_dir)
            self.project = None
            self._set_dirty(False)
            self._refresh_list()
        self._update_runtime_warning()

    def _update_runtime_warning(self):
        self.btn_runtime_warn.setVisible(not runtime.status(self.settings)["ok"])

    # -- squelette ----------------------------------------------------------------------
    def _build_ui(self):
        # barre laterale
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(280)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 20, 0, 16)
        sl.setSpacing(10)
        head = QHBoxLayout()
        head.setContentsMargins(20, 0, 10, 0)
        head.setSpacing(10)
        if ICON_FILE.exists():
            logo = QLabel()
            logo.setPixmap(QIcon(str(ICON_FILE)).pixmap(QSize(36, 36)))
            logo.setFixedSize(36, 36)
            head.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("RL Studio", "appTitle"))
        titles.addWidget(label("Apprentissage par renforcement", "muted"))
        head.addLayout(titles, 1)
        sl.addLayout(head)
        newrow = QHBoxLayout()
        newrow.setContentsMargins(16, 6, 16, 0)
        newrow.addWidget(button("+  Nouveau projet", "primary", self._new))
        sl.addLayout(newrow)
        improw = QHBoxLayout()
        improw.setContentsMargins(16, 0, 16, 0)
        improw.addWidget(button("Importer un projet (.rls)...", "ghost", self._import_dialog))
        sl.addLayout(improw)
        convrow = QHBoxLayout()
        convrow.setContentsMargins(16, 0, 16, 0)
        convrow.addWidget(button("Convertir un modele Simulink...", "ghost", self._convert_dialog))
        sl.addLayout(convrow)
        lab = label("PROJETS", "sectionLabel")
        lab.setContentsMargins(20, 8, 0, 0)
        sl.addWidget(lab)
        self.plist = QListWidget()
        self.plist.currentItemChanged.connect(self._on_select)
        sl.addWidget(self.plist, 1)
        self.btn_runtime_warn = button("⚠  Composants de calcul manquants", "danger", lambda: self._open_settings(4))
        wrow = QHBoxLayout()
        wrow.setContentsMargins(12, 0, 12, 0)
        wrow.addWidget(self.btn_runtime_warn)
        sl.addLayout(wrow)
        self.btn_actions = button("Actions du projet  ▾", "ghost")
        menu = QMenu(self.btn_actions)
        self.act_dup = menu.addAction("Dupliquer", self._duplicate)
        self.act_ren = menu.addAction("Renommer...", self._rename)
        self.act_dir = menu.addAction("Ouvrir le dossier", self._open_folder)
        self.act_exp = menu.addAction("Exporter (.rls)...", self._export_dialog)
        menu.addSeparator()
        self.act_del = menu.addAction("Supprimer...", self._delete)
        self.btn_actions.setMenu(menu)
        arow = QHBoxLayout()
        arow.setContentsMargins(12, 0, 12, 0)
        arow.addWidget(self.btn_actions)
        sl.addLayout(arow)
        self.btn_settings = button("⚙  Parametres", "ghost", self._open_settings)
        srow = QHBoxLayout()
        srow.setContentsMargins(12, 0, 12, 0)
        srow.addWidget(self.btn_settings)
        sl.addLayout(srow)
        self._update_runtime_warning()

        # zone principale : barre du haut + onglets
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        top = QWidget()
        top.setObjectName("topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(28, 18, 28, 10)
        tl.setSpacing(12)
        self.top_title = label("", "h1")
        self.top_badge = label("", "badge")
        self.lbl_dirty = label("●  Modifications non enregistrees", "dirty")
        self.lbl_dirty.hide()
        self.btn_save = button("Enregistrer", "primary", self._save)
        tl.addWidget(self.top_title)
        tl.addWidget(self.top_badge)
        tl.addStretch(1)
        tl.addWidget(self.lbl_dirty)
        tl.addWidget(self.btn_save)
        ml.addWidget(top)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._tab_general(), "Projet")
        self.tabs.addTab(self._tab_env(), "Environnement")
        self.tabs.addTab(self._tab_algo(), "Algorithme")
        self.tabs.addTab(self._tab_train(), "Entrainement")
        self.tabs.addTab(self._tab_runs(), "Resultats")
        self.distill = DistillTab(self.settings)
        self.distill.busy.connect(lambda b: self.plist.setEnabled(not b))
        self.identify = IdentifyTab(self.settings)
        self.identify.busy.connect(lambda b: self.plist.setEnabled(not b))
        self.evaltab = EvalTab(self.settings)
        self.evaltab.busy.connect(lambda b: self.plist.setEnabled(not b))
        self.tabs.insertTab(4, self.evaltab, "Evaluation")
        self.tabs.addTab(self.distill, "Modele mathematique")
        self.tabs.addTab(self.identify, "Identification")
        ml.addWidget(self.tabs, 1)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(side)
        root.addWidget(main, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._set_dirty(False)

    # -- onglet Projet ------------------------------------------------------------------
    def _tab_general(self):
        self.f_name = QLineEdit()
        self.f_name.setPlaceholderText("Nom du projet")
        self.f_desc = QPlainTextEdit()
        self.f_desc.setPlaceholderText("Decris l'objectif : systeme controle, critere a optimiser...")
        self.f_desc.setFixedHeight(88)

        self.k_kind, self.k_algo, self.k_steps = Kpi("Type"), Kpi("Algorithme"), Kpi("Duree")
        self.k_runs, self.k_last = Kpi("Entrainements"), Kpi("Dernier entrainement")
        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        for k in (self.k_kind, self.k_algo, self.k_steps, self.k_runs, self.k_last):
            kpis.addWidget(k, 1)

        env = Card("Environnement")
        self.v_env = label("", wrap=True)
        self.v_env.setTextFormat(Qt.RichText)
        self.v_env.setTextInteractionFlags(Qt.TextSelectableByMouse)
        env.body.addWidget(self.v_env)

        self.btn_go_train = button("▶  Aller a l'entrainement", "primary", lambda: self.tabs.setCurrentIndex(3))
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.btn_go_train)
        row.addWidget(button("Modifier l'environnement", None, lambda: self.tabs.setCurrentIndex(1)))
        row.addWidget(button("Ouvrir le dossier", "ghost", self._open_folder))
        row.addStretch()
        self.v_path = label("", "muted")
        self.v_path.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.f_name.textChanged.connect(self._mark_dirty)
        self.f_desc.textChanged.connect(self._mark_dirty)
        fields = QVBoxLayout()
        fields.setSpacing(6)
        fields.addWidget(label("NOM DU PROJET", "sectionLabel"))
        fields.addWidget(self.f_name)
        fields.addSpacing(6)
        fields.addWidget(label("DESCRIPTION", "sectionLabel"))
        fields.addWidget(self.f_desc)
        return scroll_page(column(fields, kpis, env, row, self.v_path), 1000)

    def _update_dashboard(self):
        p = self.project
        if p is None:
            return
        c, env, a = p.cfg, p.cfg["env"], p.cfg["algo"]
        kind = "Python" if p.kind == "python" else "Simulink"
        self.top_title.setText(p.name)
        self.top_badge.setText(kind)
        self.k_kind.set(kind)
        self.k_algo.set(a["name"])
        self.k_steps.set(f"{int(a['total_timesteps']):,} pas".replace(",", " "))
        runs = p.list_runs()
        self.k_runs.set(str(len(runs)))
        self.k_last.set(fmt_run_date(runs[0].name) if runs else "aucun")
        created = str(c.get("created", "-"))  # AAAA-MM-JJ HH:MM -> JJ/MM/AAAA HH:MM
        if len(created) == 16 and created[4] == "-":
            created = f"{created[8:10]}/{created[5:7]}/{created[0:4]} {created[11:]}"
        if p.kind == "python":
            keys = ", ".join(map(str, (env.get("params") or {}).keys())) or "aucun"
            self.v_env.setText(f"Point d'entree : <b>{env.get('entry', 'env.py:make_env')}</b><br>"
                               f"Groupes de parametres : {keys}")
        else:
            acts, obs = env.get("actions") or [], env.get("observations") or []
            self.v_env.setText(
                f"Modele : <b>{env.get('model', '-')}</b><br>"
                f"Pas de decision : {env.get('step_s', '-')} s &nbsp;·&nbsp; episode : {env.get('episode_s', '-')} s<br>"
                f"{len(acts)} action(s) : {', '.join(x.get('name', '?') for x in acts) or '-'}<br>"
                f"{len(obs)} observation(s) : {', '.join(x.get('name', '?') for x in obs) or '-'}<br>"
                f"Recompense : <code>{env.get('reward', '-')}</code>")
        self.v_path.setText(f"Cree le {created}   ·   {p.path}")

    # -- onglet Environnement -----------------------------------------------------------
    def _tab_env(self):
        # Python
        self.py_entry = QLineEdit()
        self.py_params = QPlainTextEdit()
        self.py_params.setObjectName("mono")
        self.py_params.setMinimumHeight(340)
        entry = Card("Point d'entree", "Fichier et fonction qui construisent l'environnement Gymnasium.")
        erow = QHBoxLayout()
        erow.addWidget(self.py_entry, 1)
        erow.addWidget(button("Ouvrir dans l'editeur", None, self._open_env_file))
        entry.body.addLayout(erow)
        params = Card("Parametres", "YAML passe a make_env(**params).")
        params.body.addWidget(self.py_params)
        self.py_page = scroll_page(column(entry, params), 1000)
        self.py_entry.textChanged.connect(self._mark_dirty)
        self.py_params.textChanged.connect(self._mark_dirty)

        # Simulink
        self.sl_model = QLineEdit()
        self.sl_step = QDoubleSpinBox()
        self.sl_step.setRange(1e-6, 1e9)
        self.sl_step.setDecimals(4)
        self.sl_step.setSuffix(" s")
        self.sl_episode = QDoubleSpinBox()
        self.sl_episode.setRange(1e-6, 1e12)
        self.sl_episode.setDecimals(2)
        self.sl_episode.setSuffix(" s")
        self.sl_reward = QLineEdit()
        self.sl_reward.setPlaceholderText("ex. -(temp - 20)**2 - 0.01*u**2")
        self.sl_terminate = QLineEdit()
        self.sl_terminate.setPlaceholderText("optionnel, ex. temp > 100")
        self.sl_vars = label("", "hint", wrap=True)

        model = Card("Modele Simulink")
        mrow = QHBoxLayout()
        mrow.addWidget(self.sl_model, 1)
        mrow.addWidget(button("Parcourir...", None, self._browse_model))
        mrow.addWidget(button("Explorer le modele", "primary", self._explore_model))
        model.body.addLayout(mrow)
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setLabelAlignment(Qt.AlignLeft)
        form.addRow("Pas de decision", self.sl_step)
        form.addRow("Duree d'un episode", self.sl_episode)
        model.body.addLayout(form)

        self.t_actions = SpecTable(ACTION_COLS, {"low": 0.0, "high": 1.0})
        self.t_obs = SpecTable(OBS_COLS, {"port": 1, "size": 1})
        self.t_init = SpecTable(INIT_COLS, {"low": 0.0, "high": 1.0})
        cards = [model]
        for title, hint, table in [
            ("Actions", "Parametres de blocs ou variables que l'agent commande, avec leurs bornes.", self.t_actions),
            ("Observations", "Signaux (sorties de blocs) lus par l'agent a chaque pas.", self.t_obs),
            ("Etat initial aleatoire", "Variables du workspace tirees dans [min, max] a chaque episode (optionnel).", self.t_init),
        ]:
            card = Card(title, hint)
            card.body.addWidget(table)
            row = QHBoxLayout()
            row.addWidget(button("+  Ajouter", "small", table.add_row))
            row.addWidget(button("Supprimer la selection", "small", table.remove_selected))
            row.addStretch()
            card.body.addLayout(row)
            table.changed.connect(self._mark_dirty)
            table.changed.connect(self._update_vars_hint)
            cards.append(card)

        reward = Card("Recompense", "Expression evaluee a chaque pas ; les noms des actions et observations sont disponibles.")
        reward.body.addWidget(label("Recompense", "sectionLabel"))
        reward.body.addWidget(self.sl_reward)
        reward.body.addWidget(label("Arret anticipe", "sectionLabel"))
        reward.body.addWidget(self.sl_terminate)
        reward.body.addWidget(self.sl_vars)
        cards.insert(1, reward)

        self.sl_status = label("", "statusbox", wrap=True)
        self.sl_status.hide()
        check = QHBoxLayout()
        check.addWidget(button("Verifier la configuration", None, self._check_spec))
        check.addStretch()
        cards += [check, self.sl_status]
        self.sl_page = scroll_page(column(*cards), 1000)
        for wdg in (self.sl_model, self.sl_reward, self.sl_terminate):
            wdg.textChanged.connect(self._mark_dirty)
        self.sl_reward.textChanged.connect(self._update_vars_hint)
        for wdg in (self.sl_step, self.sl_episode):
            wdg.valueChanged.connect(self._mark_dirty)

        self.env_stack = QTabWidget()  # sert uniquement de pile ; onglets caches
        self.env_stack.tabBar().hide()
        self.env_stack.addTab(self.py_page, "py")
        self.env_stack.addTab(self.sl_page, "sl")
        return self.env_stack

    def _update_vars_hint(self, *_):
        names = [r.get("name") for r in self.t_actions.rows() + self.t_obs.rows() if r.get("name")]
        self.sl_vars.setText("Variables disponibles : " + ", ".join(names + ["t", "dt", "step"]))

    # -- onglet Algorithme --------------------------------------------------------------
    def _tab_algo(self):
        self.a_name = QComboBox()
        self.a_name.addItems(ALGOS)
        self.a_steps = QSpinBox()
        self.a_steps.setRange(100, 100_000_000)
        self.a_steps.setSingleStep(10_000)
        self.a_steps.setGroupSeparatorShown(True)
        self.a_steps.setSuffix(" pas")
        self.a_seed = QSpinBox()
        self.a_seed.setRange(0, 99999)
        self.a_device = QComboBox()
        self.a_device.addItems(["cpu", "cuda", "auto"])
        self.a_base = QSpinBox()
        self.a_base.setRange(0, 100)
        self.a_base.setSuffix(" episode(s)")
        self.a_kwargs = QPlainTextEdit()
        self.a_kwargs.setObjectName("mono")
        self.a_kwargs.setPlaceholderText("learning_rate: 0.0003\nbatch_size: 256")
        self.a_kwargs.setMinimumHeight(180)

        base = Card("Entrainement", "Les petits reseaux MLP sont generalement plus rapides sur CPU que sur GPU.")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)
        form.addRow("Algorithme", self.a_name)
        form.addRow("Duree", self.a_steps)
        form.addRow("Graine aleatoire", self.a_seed)
        form.addRow("Peripherique", self.a_device)
        form.addRow("Evaluation de reference", self.a_base)
        base.body.addLayout(form)
        hyper = Card("Hyperparametres", "YAML transmis a Stable-Baselines3 (optionnel). Laisse vide pour les valeurs par defaut.")
        hyper.body.addWidget(self.a_kwargs)

        self.a_name.currentIndexChanged.connect(self._mark_dirty)
        self.a_device.currentIndexChanged.connect(self._mark_dirty)
        for s in (self.a_steps, self.a_seed, self.a_base):
            s.valueChanged.connect(self._mark_dirty)
        self.a_kwargs.textChanged.connect(self._mark_dirty)
        return scroll_page(column(base, hyper), 720)

    # -- onglet Entrainement ------------------------------------------------------------
    def _plot(self, title):
        pw = pg.PlotWidget()
        pw.setObjectName("plot")
        pw._title = title
        pw.showGrid(x=True, y=True, alpha=0.12)
        pw.setMenuEnabled(False)
        pw.setMinimumHeight(220)
        self._plots.append(pw)
        return pw

    def _series_colors(self):
        return [C["accent"], C["warning"], C["success"], C["magenta"], C["danger"]]

    def _style_plots(self):
        """(Re)colore graphiques et courbes selon le theme actif."""
        for pw in self._plots:
            pw.setBackground(C["surface"])
            pw.setTitle(pw._title, color=C["text"], size="11pt")
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setPen(C["border2"])
                pw.getAxis(ax).setTextPen(C["muted"])
        self.c_reward.setPen(pg.mkPen(C["accent"], width=1))
        self.c_reward.setSymbolBrush(C["accent"])
        self.c_smooth.setPen(pg.mkPen(C["warning"], width=3))
        self.base_line.setPen(pg.mkPen(C["danger"], width=1.5, style=Qt.DashLine))
        self.c_rew_step.setPen(pg.mkPen(C["success"], width=2))
        if hasattr(self, "distill"):
            self.distill.restyle()
            self.identify.restyle()
            self.evaltab.restyle()
        self.p_obs.restyle()
        self.p_act.restyle()

    def _tab_train(self):
        self.btn_start = button("▶  Demarrer", "primary", self._start)
        self.btn_stop = button("■  Arreter", "danger", self._stop)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.status = label("Pret.", "muted")
        bar = QHBoxLayout()
        bar.setSpacing(10)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addSpacing(8)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.status)

        self.m_eps, self.m_steps = Kpi("Episodes"), Kpi("Pas")
        self.m_last, self.m_best, self.m_base = Kpi("Derniere recompense"), Kpi("Meilleure"), Kpi("Reference")
        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        for k in (self.m_eps, self.m_steps, self.m_last, self.m_best, self.m_base):
            kpis.addWidget(k, 1)

        self.p_reward = self._plot("Recompense par episode")
        self.c_reward = self.p_reward.plot(pen=pg.mkPen(C["accent"], width=1), symbol="o", symbolSize=4,
                                           symbolBrush=C["accent"], symbolPen=None)
        self.c_smooth = self.p_reward.plot(pen=pg.mkPen(C["warning"], width=3))
        self.base_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(C["danger"], width=1.5, style=Qt.DashLine))
        self.p_reward.addItem(self.base_line)
        self.base_line.hide()
        self.p_obs = SeriesPlot("Observations (episode en cours)")
        self.p_act = SeriesPlot("Actions (episode en cours)")
        self.p_obs.setMinimumHeight(260)
        self.p_act.setMinimumHeight(260)
        self.chk_norm = QCheckBox("Normaliser les courbes (comparer des signaux d'echelles differentes)")
        self.chk_norm.toggled.connect(lambda f: (self.p_obs.set_normalized(f), self.p_act.set_normalized(f)))
        self.p_rew_step = self._plot("Recompense par pas (episode en cours)")
        self.c_rew_step = self.p_rew_step.plot(pen=pg.mkPen(C["success"], width=2))
        self._style_plots()
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self.p_reward, 0, 0)
        grid.addWidget(self.p_obs, 0, 1)
        grid.addWidget(self.p_act, 1, 0)
        grid.addWidget(self.p_rew_step, 1, 1)

        self.logbox = QPlainTextEdit()
        self.logbox.setObjectName("mono")
        self.logbox.setReadOnly(True)
        self.logbox.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.logbox.setMaximumBlockCount(LOG_MAX_LINES)  # un journal qui grossit sans limite ralentit tout l'affichage
        self.logbox.setPlaceholderText("Le journal de l'entrainement s'affiche ici : demarrage de MATLAB, "
                                       "evaluation de reference, episodes, vitesse et temps restant.")
        side = Card("Journal")
        side.setFixedWidth(380)
        side.body.addWidget(self.logbox, 1)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 20, 28, 20)
        lay.setSpacing(14)
        lay.addLayout(bar)
        lay.addLayout(kpis)
        lay.addWidget(self.chk_norm)
        main = QHBoxLayout()
        main.setSpacing(12)
        main.addLayout(grid, 1)
        main.addWidget(side)
        lay.addLayout(main, 1)
        self._reset_train_state()
        return page

    # -- onglet Resultats ---------------------------------------------------------------
    def _tab_runs(self):
        self.runs_list = QListWidget()
        self.runs_list.setObjectName("runs")
        self.runs_list.currentItemChanged.connect(self._show_run)
        self.runs_empty = label("Aucun entrainement pour ce projet.\nLance-en un depuis l'onglet Entrainement.",
                                "muted", wrap=True)
        self.runs_empty.setAlignment(Qt.AlignCenter)
        left = Card("Entrainements")
        left.body.addWidget(self.runs_list, 1)
        left.body.addWidget(self.runs_empty)
        left.setFixedWidth(330)

        self.run_title = label("", "h1")
        self.run_info = label("", wrap=True)
        self.run_info.setTextFormat(Qt.RichText)
        self.run_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.run_cfg = QPlainTextEdit()
        self.run_cfg.setObjectName("mono")
        self.run_cfg.setReadOnly(True)
        self.btn_run_dir = button("Ouvrir le dossier du run", None, self._open_run_folder)
        detail = Card()
        detail.body.addWidget(self.run_title)
        detail.body.addWidget(self.run_info)
        detail.body.addWidget(label("Configuration utilisee", "sectionLabel"))
        detail.body.addWidget(self.run_cfg, 1)
        drow = QHBoxLayout()
        drow.addWidget(self.btn_run_dir)
        drow.addStretch()
        detail.body.addLayout(drow)

        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(28, 20, 28, 20)
        lay.setSpacing(14)
        lay.addWidget(left)
        lay.addWidget(detail, 1)
        return page

    # -- liste de projets ---------------------------------------------------------------
    def _item_widget(self, p: Project):
        w = QWidget()
        w.setAttribute(Qt.WA_TransparentForMouseEvents)
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(1)
        t = label(p.name)
        t.setStyleSheet("font-weight: 600; font-size: 14px;")
        kind = "Python" if p.kind == "python" else "Simulink"
        lay.addWidget(t)
        lay.addWidget(label(f"{kind}  ·  {p.cfg['algo']['name']}", "muted"))
        return w

    def _refresh_list(self, select=None):
        self.plist.blockSignals(True)
        self.plist.clear()
        for p in self.store.list():
            it = QListWidgetItem()
            it.setData(Qt.UserRole, p.slug)
            it.setSizeHint(QSize(0, 56))
            self.plist.addItem(it)
            self.plist.setItemWidget(it, self._item_widget(p))
            if p.slug == select:
                self.plist.setCurrentItem(it)
        self.plist.blockSignals(False)
        if select is None and self.plist.count():
            self.plist.setCurrentRow(0)
        elif self.plist.count() == 0:
            self._load_project(None)

    def _on_select(self, cur, prev):
        if self.worker:  # pas de changement pendant un entrainement
            return
        if self.dirty and not self._ask_save():
            self.plist.blockSignals(True)
            for i in range(self.plist.count()):
                if self.project and self.plist.item(i).data(Qt.UserRole) == self.project.slug:
                    self.plist.setCurrentRow(i)
            self.plist.blockSignals(False)
            return
        self._load_project(self.store.get(cur.data(Qt.UserRole)) if cur else None)

    def _ask_save(self):
        r = QMessageBox.question(self, "Modifications non enregistrees", "Enregistrer les modifications ?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self._save()
        self._set_dirty(False)
        return True

    def _new(self):
        if self.dirty and not self._ask_save():
            return
        templates = self.store.list_templates()
        labels = [f"{n} - {d}" for _, n, d in templates]
        choice, ok = QInputDialog.getItem(self, "Nouveau projet", "Modele de depart :", labels, 0, False)
        if not ok:
            return
        tid = templates[labels.index(choice)][0]
        name, ok = QInputDialog.getText(self, "Nouveau projet", "Nom du projet :")
        if not ok or not name.strip():
            return
        try:
            p = self.store.create(name, tid)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
            return
        self._set_dirty(False)
        self._refresh_list(select=p.slug)
        self._load_project(p)

    def _duplicate(self):
        if self.project and (not self.dirty or self._ask_save()):
            p = self.store.duplicate(self.project)
            self._refresh_list(select=p.slug)
            self._load_project(p)

    def _rename(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "Renommer", "Nouveau nom :", text=self.project.name)
        if ok and name.strip():
            self.store.rename(self.project, name)
            self._refresh_list(select=self.project.slug)
            self._load_project(self.store.get(self.project.slug))

    def _delete(self):
        if not self.project:
            return
        r = QMessageBox.question(
            self, "Supprimer", f"Supprimer le projet « {self.project.name} » ?\n"
            "Il sera deplace dans projects/.trash/ et pourra etre recupere.")
        if r == QMessageBox.Yes:
            self.store.delete(self.project)
            self.project = None
            self._set_dirty(False)
            self._refresh_list()

    # -- conversion d'un modele Simulink existant ---------------------------------------
    def _convert_dialog(self):
        from .convert_dialog import ConvertDialog

        if self.dirty and not self._ask_save():
            return
        if self.worker or any(getattr(t, "runner", None) for t in (self.distill, self.evaltab, self.identify)):
            QMessageBox.information(self, "Convertir un modele", "Attends la fin du calcul en cours avant de convertir un modele.")
            return
        dlg = ConvertDialog(self, self.settings, self.store)
        dlg.exec()
        if dlg.created_slug:
            self._set_dirty(False)
            self._refresh_list(select=dlg.created_slug)
            self._load_project(self.store.get(dlg.created_slug))

    # -- partage de projets (.rls) -----------------------------------------------------
    def _export_dialog(self):
        from ..archive import export_project

        if not self.project or (self.dirty and not self._save()):
            return
        labels = ["Sans resultats d'entrainement (le plus leger)", "Avec le dernier entrainement et son modele",
                  "Avec tous les entrainements"]
        choice, ok = QInputDialog.getItem(self, "Exporter le projet", "Contenu de l'archive :", labels, 0, False)
        if not ok:
            return
        mode = ("none", "last", "all")[labels.index(choice)]
        default = str(Path.home() / "Documents" / f"{self.project.slug}.rls")
        path, _ = QFileDialog.getSaveFileName(self, "Exporter le projet", default, "Projet RL Studio (*.rls)")
        if not path:
            return
        try:
            out = export_project(self.project, path, runs=mode)
        except Exception as e:
            QMessageBox.critical(self, "Export impossible", str(e))
            return
        QMessageBox.information(self, "Projet exporte", f"{out}\n\nTaille : {out.stat().st_size / 1024:.0f} Ko")

    def _import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importer un projet", str(Path.home()), "Projet RL Studio (*.rls)")
        if path:
            self.import_file(path)

    def _confirm_import(self, report) -> bool:
        m = report["manifest"]
        lines = [f"<b>{m['name']}</b> ({m['kind']}) - {report['files']} fichiers, {report['size'] / 1024:.0f} Ko"]
        if report["runs"]:
            lines.append(f"Resultats d'entrainement inclus : {len(report['runs'])}")
        if report["code_files"]:
            shown = ", ".join(report["code_files"][:8]) + (" ..." if len(report["code_files"]) > 8 else "")
            lines.append("<br><b>Attention :</b> ce projet contient du <b>code qui s'executera sur votre ordinateur</b> "
                         f"(environnement Python, modele Simulink, scripts) :<br><i>{shown}</i><br><br>"
                         "N'importez que des projets provenant de personnes de confiance.")
        box = QMessageBox(self)
        box.setWindowTitle("Importer un projet")
        box.setTextFormat(Qt.RichText)
        box.setText("<br>".join(lines))
        box.setIcon(QMessageBox.Warning if report["code_files"] else QMessageBox.Question)
        yes = box.addButton("Importer", QMessageBox.AcceptRole)
        box.addButton("Annuler", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is yes

    def import_file(self, path, confirm=None):
        """Importe un .rls (menu, glisser-deposer ou double-clic). `confirm(report) -> bool` remplace la boite de dialogue."""
        from ..archive import ArchiveError, import_project, inspect_archive

        try:
            report = inspect_archive(path)
            if report["problems"]:
                raise ArchiveError("Import refuse :\n- " + "\n- ".join(report["problems"]))
            if not (confirm or self._confirm_import)(report):
                return None
            if self.dirty and not self._ask_save():
                return None
            project = import_project(path, self.store)
        except ArchiveError as e:
            QMessageBox.warning(self, "Import impossible", str(e))
            return None
        self._set_dirty(False)
        self._refresh_list(select=project.slug)
        self._load_project(project)
        return project

    def dragEnterEvent(self, event):
        if any(u.toLocalFile().lower().endswith(".rls") for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for u in event.mimeData().urls():
            if u.toLocalFile().lower().endswith(".rls"):
                self.import_file(u.toLocalFile())

    def _open_folder(self):
        target = self.project.path if self.project else self.store.root
        if sys.platform == "win32":
            os.startfile(target)

    def _open_env_file(self):
        if not self.project:
            return
        filename = self.py_entry.text().partition(":")[0] or "env.py"
        path = self.project.path / filename
        if path.exists() and sys.platform == "win32":
            os.startfile(path)

    # -- chargement / enregistrement ----------------------------------------------------
    def _set_enabled(self, on):
        self.tabs.setEnabled(on)
        self.btn_actions.setEnabled(on)
        for a in (self.act_dup, self.act_ren, self.act_dir, self.act_exp, self.act_del):
            a.setEnabled(on)

    def _set_dirty(self, flag):
        self.dirty = flag
        self.lbl_dirty.setVisible(flag)
        self.btn_save.setEnabled(flag)

    def _load_project(self, project: Project | None):
        self.project = project
        self._loading = True
        self._set_enabled(project is not None)
        if project is None:
            self.top_title.setText("Aucun projet")
            self.top_badge.setText("")
            self._set_dirty(False)
            self._loading = False
            return
        c, env = project.cfg, project.cfg["env"]
        self.f_name.setText(project.name)
        self.f_desc.setPlainText(c.get("description", ""))
        is_py = project.kind == "python"
        self.env_stack.setCurrentIndex(0 if is_py else 1)
        if is_py:
            self.py_entry.setText(env.get("entry", "env.py:make_env"))
            self.py_params.setPlainText(yaml.safe_dump(env.get("params") or {}, sort_keys=False, allow_unicode=True))
        else:
            self.sl_model.setText(env.get("model", ""))
            self.sl_step.setValue(float(env.get("step_s", 60)))
            self.sl_episode.setValue(float(env.get("episode_s", 3600)))
            self.sl_reward.setText(env.get("reward", ""))
            self.sl_terminate.setText(env.get("terminate", "") or "")
            self.t_actions.set_rows(env.get("actions"))
            self.t_obs.set_rows(env.get("observations"))
            self.t_init.set_rows(env.get("initial"))
            self.sl_status.hide()
            self._update_vars_hint()
        a = c["algo"]
        self.a_name.setCurrentText(a["name"])
        self.a_steps.setValue(int(a["total_timesteps"]))
        self.a_seed.setValue(int(a["seed"]))
        self.a_device.setCurrentText(a["device"])
        self.a_base.setValue(int(a.get("baseline_episodes", 0)))
        self.a_kwargs.setPlainText(yaml.safe_dump(a.get("kwargs") or {}, sort_keys=False) if a.get("kwargs") else "")
        self._refresh_runs()
        self.distill.set_project(project)
        self.identify.set_project(project)
        self.evaltab.set_project(project)
        self._reset_train_state()
        self._update_dashboard()
        self._set_dirty(False)
        self._loading = False
        self.setWindowTitle(f"RL Studio - {project.name}")

    def _mark_dirty(self, *_):
        if not self._loading and self.project:
            self._set_dirty(True)

    def _collect(self):
        """Ecrit les widgets dans project.cfg ; leve ValueError si une saisie est invalide."""
        c = self.project.cfg
        c["name"] = self.f_name.text().strip() or self.project.slug
        c["description"] = self.f_desc.toPlainText()
        try:
            if self.project.kind == "python":
                c["env"]["entry"] = self.py_entry.text().strip()
                c["env"]["params"] = yaml.safe_load(self.py_params.toPlainText()) or {}
                if not isinstance(c["env"]["params"], dict):
                    raise ValueError("Les parametres doivent etre un dictionnaire YAML.")
            else:
                c["env"].update({
                    "model": self.sl_model.text().strip(), "step_s": self.sl_step.value(),
                    "episode_s": self.sl_episode.value(), "reward": self.sl_reward.text().strip(),
                    "terminate": self.sl_terminate.text().strip(), "actions": self.t_actions.rows(),
                    "observations": self.t_obs.rows(), "initial": self.t_init.rows(),
                })
            kw = yaml.safe_load(self.a_kwargs.toPlainText()) or {}
            if not isinstance(kw, dict):
                raise ValueError("Les hyperparametres doivent etre un dictionnaire YAML.")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML invalide : {e}")
        c["algo"].update(name=self.a_name.currentText(), total_timesteps=self.a_steps.value(),
                         seed=self.a_seed.value(), device=self.a_device.currentText(),
                         baseline_episodes=self.a_base.value(), kwargs=kw)

    def _save(self):
        if not self.project:
            return True
        try:
            self._collect()
        except ValueError as e:
            QMessageBox.warning(self, "Saisie invalide", str(e))
            return False
        self.project.save()
        self._update_dashboard()
        self._set_dirty(False)
        self.setWindowTitle(f"RL Studio - {self.project.name}")
        self._refresh_list(select=self.project.slug)
        return True

    # -- Simulink -----------------------------------------------------------------------
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Modele Simulink", str(self.project.path), "Simulink (*.slx *.mdl)")
        if not path:
            return
        src = os.path.abspath(path)
        if not src.startswith(str(self.project.path)):
            dst = self.project.path / os.path.basename(src)
            shutil.copy2(src, dst)
            self._log(f"Modele copie dans le projet : {dst}")
            src = str(dst)
        self.sl_model.setText(os.path.relpath(src, self.project.path))

    def _explore_model(self):
        if not self.sl_model.text().strip():
            QMessageBox.information(self, "Modele", "Choisis d'abord un modele Simulink.")
            return
        path = self.project.path / self.sl_model.text().strip()
        if not path.exists():
            QMessageBox.warning(self, "Modele", f"Fichier introuvable : {path}")
            return
        if getattr(self, "_explore_job", None):
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._explore_job = WorkerJob(self.settings, ["inspect", "--model", str(path)], self)
        self._explore_job.done.connect(self._explore_done)
        self._explore_job.start()

    def _explore_done(self, res, err):
        QApplication.restoreOverrideCursor()
        self._explore_job = None
        if res is None:
            QMessageBox.critical(self, "MATLAB", f"Impossible d'inspecter le modele :\n{err}")
            return
        dlg = SimulinkExplorer(self, res["blocks"])
        dlg.on_observation = self.t_obs.add_row
        dlg.on_action = self.t_actions.add_row
        dlg.exec()

    def _set_status(self, text, tone):
        self.sl_status.setText(text)
        self.sl_status.setProperty("tone", tone)
        self.sl_status.style().unpolish(self.sl_status)
        self.sl_status.style().polish(self.sl_status)
        self.sl_status.show()

    def _check_spec(self):
        try:
            self._collect()
        except ValueError as e:
            self._set_status(str(e), "bad")
            return
        errs = validate_spec(self.project.cfg["env"])
        if errs:
            self._set_status("\n".join("•  " + e for e in errs), "bad")
        else:
            self._set_status("✓  Configuration valide.", "ok")

    # -- entrainement -------------------------------------------------------------------
    def _reset_train_state(self):
        self.xs, self.rewards = [], []
        self.c_reward.setData([], [])
        self.c_smooth.setData([], [])
        self.base_line.hide()
        self.progress.setValue(0)
        self.status.setText("Pret.")
        for k in (self.m_eps, self.m_steps, self.m_last, self.m_best, self.m_base):
            k.set("-")
        self._reset_trace()

    def _reset_trace(self):
        self.tr_rew, self._ep_start = [], None
        self._rew_timer.stop()
        self.p_obs.reset()
        self.p_act.reset()
        self.c_rew_step.setData([], [])

    def _log(self, text):
        stamp = time.strftime("%H:%M:%S")
        for line in str(text).rstrip().splitlines() or [""]:
            self.logbox.appendPlainText(f"[{stamp}] {line}")

    def _start(self):
        if not self.project or not self._save():
            return
        if self.project.kind == "simulink":
            errs = validate_spec(self.project.cfg["env"])
            if errs:
                QMessageBox.warning(self, "Configuration Simulink", "- " + "\n- ".join(errs))
                return
        self._reset_train_state()
        self.logbox.clear()
        self._total_steps = int(self.project.cfg["algo"]["total_timesteps"])
        self.progress.setRange(0, self._total_steps)
        self._t_run, self._t_learn, self._steps_learn0, self._phase_base = time.monotonic(), None, 0, False
        self._log("Lancement de l'entrainement...")
        self.worker = TrainRunner(self.project, self.settings, self)
        self.worker.episode.connect(self._on_episode)
        self.worker.progress.connect(self._on_progress)
        self.worker.state.connect(self._on_state)
        self.worker.baseline.connect(self._on_baseline)
        self.worker.meta.connect(self._on_meta)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        try:
            self.worker.start()
        except RuntimeMissing as e:
            self.worker = None
            QMessageBox.warning(self, "Composants de calcul", str(e))
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.plist.setEnabled(False)
        self.status.setText("Demarrage de l'environnement...")

    def _on_progress(self, stage, fraction):
        if stage == "baseline":
            self._phase_base = True
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
            self.status.setText(f"Evaluation de reference : {fraction * 100:.0f} %")

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("Arret...")

    def _on_meta(self, obs_names, act_names):
        self.p_obs.set_names(obs_names)
        self.p_act.set_names(act_names)

    def _on_baseline(self, mean, lbl):
        self.base_line.setValue(mean)
        self.base_line.show()
        self.m_base.set(f"{mean:.2f}")
        self._log(f"Reference tracee en rouge : {lbl}")

    def _on_episode(self, n, reward, length, steps):
        self.xs.append(n)
        self.rewards.append(reward)
        self.c_reward.setData(self.xs, self.rewards)
        if len(self.rewards) >= 5:
            w = min(10, len(self.rewards))
            self.c_smooth.setData(self.xs[w - 1:], np.convolve(self.rewards, np.ones(w) / w, mode="valid"))
        self.m_eps.set(str(n))
        self.m_last.set(f"{reward:.2f}")
        self.m_best.set(f"{max(self.rewards):.2f}")
        extra = ""
        if self._t_learn is not None and steps > self._steps_learn0:
            done = steps - self._steps_learn0
            rate = done / max(time.monotonic() - self._t_learn, 1e-6)
            left = max(self._total_steps - steps, 0) / rate
            extra = f" | {rate:.1f} pas/s | reste ~{fmt_duration(left)}"
            self.status.setText(f"Episode {n} - reste ~{fmt_duration(left)}")
        else:
            self.status.setText(f"Episode {n}")
        self._log(f"Episode {n} | {length} pas | recompense {reward:.2f} | meilleure {max(self.rewards):.2f}{extra}")
        self._reset_trace()

    def _on_state(self, s):
        if self._t_learn is None:  # premier pas d'apprentissage : fin de la phase de preparation
            self._t_learn, self._steps_learn0 = time.monotonic(), s["timesteps"] - 1
            self._log(f"Apprentissage demarre (preparation : {fmt_duration(self._t_learn - self._t_run)}).")
            self.progress.setRange(0, self._total_steps)
            self.status.setText("Apprentissage en cours...")
        self.m_steps.set(f"{s['timesteps']:,}".replace(",", " "))
        self.progress.setValue(min(s["timesteps"], self._total_steps))
        if self._ep_start is None:
            self._ep_start = s["timesteps"]
        t = s["timesteps"] - self._ep_start
        self.p_obs.append(t, s["obs"])
        self.p_act.append(t, np.atleast_1d(s["action"]))
        self.tr_rew.append(s["reward"])
        if not self._rew_timer.isActive():  # dessin regroupe (~25 / s) : le nombre de points recus ne charge pas l'interface
            self._rew_timer.start()

    def _flush_rew(self):
        self.c_rew_step.setData(self.tr_rew)

    def _on_finished(self, message):
        self._log(message + f"  (duree totale : {fmt_duration(time.monotonic() - self._t_run)})")
        self.status.setText("Termine." if message.startswith("Termine") else
                            "Arrete." if message.startswith("Arrete") else "Erreur - voir le journal")
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.plist.setEnabled(True)
        self._refresh_runs()
        self.distill.set_project(self.project)
        self.identify.set_project(self.project)
        self.evaltab.set_project(self.project)
        self._update_dashboard()

    # -- resultats ----------------------------------------------------------------------
    def _refresh_runs(self):
        self.runs_list.clear()
        self.run_title.setText("")
        self.run_info.setText("")
        self.run_cfg.clear()
        runs = self.project.list_runs() if self.project else []
        for d in runs:
            sub = []
            base = d / "baseline.json"
            if base.exists():
                try:
                    sub.append(f"reference {json.loads(base.read_text(encoding='utf-8'))['mean']:.2f}")
                except Exception:
                    pass
            sub.append("modele ✓" if (d / "model.zip").exists() else "sans modele")
            it = QListWidgetItem(f"{fmt_run_date(d.name)}\n{'  ·  '.join(sub)}")
            it.setData(Qt.UserRole, str(d))
            self.runs_list.addItem(it)
        self.runs_empty.setVisible(not runs)
        self.runs_list.setVisible(bool(runs))
        self.btn_run_dir.setEnabled(bool(runs))
        if runs:
            self.runs_list.setCurrentRow(0)

    def _show_run(self, cur, prev):
        if not cur:
            return
        d = cur.data(Qt.UserRole)
        self.run_title.setText(fmt_run_date(os.path.basename(d)))
        rows = []
        has_model = os.path.exists(os.path.join(d, "model.zip"))
        rows.append(f"Modele : <b>{'enregistre' if has_model else 'absent (entrainement interrompu ou en erreur)'}</b>")
        base = os.path.join(d, "baseline.json")
        if os.path.exists(base):
            b = json.loads(open(base, encoding="utf-8").read())
            rows.append(f"Reference ({b.get('label', '?')}) : <b>{b['mean']:.3f}</b> ± {b['std']:.3f}")
        rows.append(f"<span style='color:{C['muted']}'>{d}</span>")
        self.run_info.setText("<br>".join(rows))
        cfg = os.path.join(d, "project.yaml")
        self.run_cfg.setPlainText(open(cfg, encoding="utf-8").read() if os.path.exists(cfg) else "")

    def _open_run_folder(self):
        it = self.runs_list.currentItem()
        if it and sys.platform == "win32":
            os.startfile(it.data(Qt.UserRole))

    def closeEvent(self, event):
        if self.dirty and not self._ask_save():
            event.ignore()
            return
        if self.worker:
            self.worker.stop()
            if not self.worker.proc.waitForFinished(8000):
                self.worker.proc.kill()
        super().closeEvent(event)


def main(open_file=None):
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    open_file = open_file or next((a for a in sys.argv[1:] if a.lower().endswith(".rls")), None)
    if open_file:
        QTimer.singleShot(200, lambda: win.import_file(open_file))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
