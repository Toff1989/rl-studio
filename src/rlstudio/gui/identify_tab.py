"""Onglet « Identification » : modele reduit sans politique, en regime permanent (entrees -> sorties) ou dynamique."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..identify import METHOD_LABELS, METHODS
from ..identify.pipeline import ensure_files, identify_dir
from .runner import IdentifyRunner, RuntimeMissing, WorkerJob
from .theme import C
from .widgets import Card, button, label, scroll_page

MODES = [("steady", "Regime permanent : entrees -> sorties  (y = f(u))"), ("dynamic", "Dynamique : x(k+1) = x(k) + f(x, u)")]
IN_COLS = ["Balayer", "Entree", "Min", "Max", "Valeur fixee"]
STEADY_COLS = ["Modele", "Complexite", "R2", "NRMSE", "Erreur max", ""]
DYN_COLS = ["Modele", "Complexite", "R2 un pas", "Fidelite sim. libre", "NRMSE", ""]


def _floats(text, what):
    try:
        out = [float(t) for t in text.replace(";", ",").split(",") if t.strip()]
    except ValueError:
        raise ValueError(f"{what} : liste de nombres separes par des virgules attendue.") from None
    if not out:
        raise ValueError(f"{what} : au moins une valeur.")
    return out


def _spin(lo, hi, val, dec=None, suffix=""):
    w = QSpinBox() if dec is None else QDoubleSpinBox()
    w.setRange(lo, hi)
    if dec is not None:
        w.setDecimals(dec)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class IdentifyTab(QWidget):
    busy = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project = None
        self.report = None
        self.report_dir = None
        self.runner = None
        self.job = None
        self.signals = None  # {"actions": [{name, low, high}], "observations": [...], "initial": [...]}
        self._build()

    # -- construction -----------------------------------------------------------------------
    def _build(self):
        self.mode = QComboBox()
        for key, text in MODES:
            self.mode.addItem(text, key)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.method_boxes = {}
        meth = QHBoxLayout()
        for m in METHODS:
            cb = QCheckBox(METHOD_LABELS[m])
            cb.setChecked(True)
            self.method_boxes[m] = cb
            meth.addWidget(cb)
        meth.addStretch()

        # entrees / sorties
        self.in_table = QTableWidget(0, len(IN_COLS))
        self.in_table.setHorizontalHeaderLabels(IN_COLS)
        self.in_table.verticalHeader().setVisible(False)
        self.in_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.in_table.setMaximumHeight(190)
        self.btn_read = button("Lire les signaux du projet", None, self._read_signals)
        self.out_list = QListWidget()
        self.out_list.setMaximumHeight(120)
        self.out_box = Card()
        self.out_box.setObjectName("card")

        # parametres du balayage
        self.s_samples = _spin(5, 100000, 200)
        self.s_val = _spin(2, 100000, 50)
        self.s_extreme = _spin(0, 100, 15, 0, " %")
        self.s_rtol = _spin(0.00001, 100, 0.1, 5, " %")
        self.s_atol = _spin(0, 1000, 0.000001, 9)
        self.s_window = _spin(1, 100, 3)
        self.s_settle = _spin(0, 1000000, 0)
        self.s_settle.setSpecialValueText("duree d'un episode")
        self.s_scenario = QCheckBox("Variables `initial` (perturbations) = entrees du modele")
        self.d_episodes = _spin(2, 100000, 40)
        self.d_val = _spin(2, 100000, 10)
        self.d_rounds = _spin(0, 50, 0)
        self.d_round_eps = _spin(1, 100000, 20)
        self.d_hold = _spin(1, 1000, 10)
        self.d_extreme = _spin(0, 100, 20, 0, " %")
        self.d_maxsteps = _spin(0, 1000000, 0)
        self.d_maxsteps.setSpecialValueText("duree d'un episode")
        # modeles
        self.f_poly = QLineEdit("2, 3")
        self.f_sparse = QLineEdit("0.3, 0.1, 0.03")
        self.f_hidden = QLineEdit("32, 32")
        self.f_epochs = _spin(1, 100000, 400)
        self.f_margin = _spin(0, 50, 2, 1, " %")
        self.f_seed = _spin(0, 10 ** 6, 0)
        self.f_poly.setToolTip("Degres des polynomes testes (un modele par degre)")
        self.f_sparse.setToolTip("Seuils de la regression parcimonieuse (SINDy) : plus grand = moins de termes")
        self.f_hidden.setToolTip("Neurones par couche cachee du reseau (tanh)")

        self.steady_box = self._grid_card("Balayage en regime permanent", [
            ("Points d'apprentissage", self.s_samples), ("Points de validation", self.s_val), ("Part aux bornes", self.s_extreme),
            ("Stabilisation : tol. relative", self.s_rtol), ("Stabilisation : tol. absolue", self.s_atol), ("Pas consecutifs stables", self.s_window),
            ("Duree max d'etablissement (pas)", self.s_settle)], extra=self.s_scenario,
            hint="Chaque point est simule, entrees constantes, jusqu'a stabilisation des sorties ; les points qui ne convergent pas sont ecartes.")
        self.dyn_box = self._grid_card("Balayage dynamique", [
            ("Episodes d'apprentissage", self.d_episodes), ("Episodes de validation", self.d_val), ("Tours de collecte en plus", self.d_rounds),
            ("Episodes / tour", self.d_round_eps), ("Palier max (pas)", self.d_hold), ("Part aux bornes", self.d_extreme),
            ("Pas max / episode", self.d_maxsteps)],
            hint="Paliers d'action aleatoires de duree variable ; validation en simulation libre sur des episodes jamais vus.")
        models = self._grid_card("Modeles", [
            ("Degres du polynome", self.f_poly), ("Seuils parcimonieux (SINDy)", self.f_sparse), ("Couches cachees (reseau)", self.f_hidden),
            ("Epoques (reseau)", self.f_epochs), ("Tolerance de recommandation", self.f_margin), ("Graine", self.f_seed)])

        self.btn_start = button("Identifier le modele", "primary", self._start)
        self.btn_stop = button("Arreter", None, self._stop)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p %")
        self.status = label("", "muted", wrap=True)
        self.stage = label("", wrap=True)
        self.timing = label("", "muted")
        self.logbox = QPlainTextEdit()
        self.logbox.setObjectName("mono")
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(5000)
        self.logbox.setFixedHeight(120)
        self.logbox.hide()
        self._t0, self._fraction = None, 0.0
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)
        row = QHBoxLayout()
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addWidget(self.progress, 1)

        cfg = Card("Identification", "Sans politique ni entrainement : le modele (Simulink ou Python) est balaye puis remplace par des equations ou un "
                   "reseau de neurones, valides sur des points ou des episodes jamais vus.")
        top = QHBoxLayout()
        top.addWidget(label("Mode", "sectionLabel"))
        top.addWidget(self.mode, 1)
        cfg.body.addLayout(top)
        cfg.body.addWidget(label("Familles de modeles", "sectionLabel"))
        cfg.body.addLayout(meth)

        ins = Card("Entrees (actions)", "Decocher une entree pour la maintenir a une valeur fixe ; modifier Min / Max pour restreindre la plage balayee.")
        ins.body.addWidget(self.in_table)
        self.outs_label = label("Sorties a modeliser (observations)", "sectionLabel")
        ins.body.addWidget(self.outs_label)
        ins.body.addWidget(self.out_list)
        ins.body.addWidget(self.btn_read, 0, Qt.AlignLeft)
        self.ins = ins

        run_card = Card()
        run_card.body.addLayout(row)
        run_card.body.addWidget(self.stage)
        run_card.body.addWidget(self.timing)
        run_card.body.addWidget(self.logbox)
        run_card.body.addWidget(self.status)

        # resultats
        self.result_combo = QComboBox()
        self.result_combo.currentIndexChanged.connect(self._load_report)
        self.table = QTableWidget(0, len(STEADY_COLS))
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_candidate)
        self.table.setMinimumHeight(180)
        self.pareto = self._plot()
        self.detail = self._plot()
        self.plot_out = QComboBox()
        self.plot_out.currentIndexChanged.connect(self._draw_plots)
        plots = QHBoxLayout()
        plots.addWidget(self.pareto)
        right = QVBoxLayout()
        right.addWidget(self.plot_out)
        right.addWidget(self.detail, 1)
        plots.addLayout(right)
        self.equations = QPlainTextEdit()
        self.equations.setObjectName("mono")
        self.equations.setReadOnly(True)
        self.equations.setMinimumHeight(150)
        self.notes = label("", "hint", wrap=True)
        self.btn_copy_py = button("Copier Python", None, lambda: self._copy(".py"))
        self.btn_copy_m = button("Copier MATLAB", None, lambda: self._copy(".m"))
        self.btn_save = button("Enregistrer...", None, self._save_as)
        self.btn_slx = button("Exporter vers Simulink...", None, self._export_simulink)
        self.btn_open = button("Ouvrir le dossier", None, self._open_dir)
        actions = QHBoxLayout()
        for b in (self.btn_copy_py, self.btn_copy_m, self.btn_save, self.btn_slx, self.btn_open):
            actions.addWidget(b)
        actions.addStretch()
        res = Card("Resultats")
        top2 = QHBoxLayout()
        top2.addWidget(label("Identification", "sectionLabel"))
        top2.addWidget(self.result_combo, 1)
        res.body.addLayout(top2)
        self.summary = label("Aucune identification pour ce projet.", "muted", wrap=True)
        res.body.addWidget(self.summary)
        res.body.addWidget(self.table)
        res.body.addLayout(plots)
        res.body.addWidget(label("Equations / description du modele selectionne", "sectionLabel"))
        res.body.addWidget(self.equations)
        res.body.addLayout(actions)
        res.body.addWidget(self.notes)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(28, 20, 28, 20)
        lay.setSpacing(14)
        for w in (cfg, ins, self.steady_box, self.dyn_box, models, run_card):
            lay.addWidget(w)
        lay.addWidget(res, 1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll_page(inner))
        self._set_result_buttons(False)
        self._mode_changed()
        self.restyle()

    def _grid_card(self, title, fields, extra=None, hint=None):
        card = Card(title, hint)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        for i, (txt, w) in enumerate(fields):
            grid.addWidget(label(txt, "sectionLabel"), (i // 4) * 2, i % 4)
            grid.addWidget(w, (i // 4) * 2 + 1, i % 4)
        card.body.addLayout(grid)
        if extra is not None:
            card.body.addWidget(extra)
        return card

    def _plot(self):
        pw = pg.PlotWidget()
        pw.setObjectName("plot")
        pw.showGrid(x=True, y=True, alpha=0.12)
        pw.setMenuEnabled(False)
        pw.setMinimumHeight(240)
        pw._title = pw._xl = pw._yl = ""
        return pw

    def restyle(self):
        if "surface" not in C:
            return
        for pw in (self.pareto, self.detail):
            pw.setBackground(C["surface"])
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setPen(C["border2"])
                pw.getAxis(ax).setTextPen(C["muted"])
        self._draw_plots()

    def current_mode(self):
        return self.mode.currentData()

    def _mode_changed(self, *_):
        steady = self.current_mode() == "steady"
        self.steady_box.setVisible(steady)
        self.dyn_box.setVisible(not steady)
        self.out_list.setVisible(steady)
        self.outs_label.setVisible(steady)
        self.f_margin.setValue(2.0 if steady else 5.0)
        self.f_epochs.setValue(400 if steady else 300)

    # -- projet et signaux -------------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.signals = None
        self._fill_signals(None)
        if project is not None and project.kind == "simulink":  # lecture directe de la configuration : pas de processus
            cfg = project.cfg.get("env", {})
            obs = [o["name"] if int(o.get("size", 1)) == 1 else f"{o['name']}[{i}]" for o in cfg.get("observations", []) for i in range(int(o.get("size", 1)))]
            self._fill_signals({"actions": [{"name": a["name"], "low": float(a["low"]), "high": float(a["high"])} for a in cfg.get("actions", [])],
                                "observations": obs, "initial": [i["variable"] for i in cfg.get("initial", [])]})
        self.btn_start.setEnabled(project is not None and self.runner is None)
        self.btn_read.setEnabled(project is not None)
        self._load_options()
        self._refresh_results()

    def _read_signals(self):
        if not self.project:
            return
        self.job = WorkerJob(self.settings, ["env-signals", "--project", str(self.project.path)], self)
        self.job.done.connect(lambda res, err: self._fill_signals(res) if res else self.status.setText(f"ERREUR : {err}"))
        self.status.setText("Lecture des signaux du projet ...")
        self.job.start()

    def _fill_signals(self, sig):
        self.signals = sig
        self.in_table.setRowCount(0)
        self.out_list.clear()
        if not sig:
            return
        self.in_table.setRowCount(len(sig["actions"]))
        for r, a in enumerate(sig["actions"]):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked)
            name = QTableWidgetItem(a["name"])
            name.setFlags(Qt.ItemIsEnabled)
            name.setData(Qt.UserRole, (a["low"], a["high"]))
            self.in_table.setItem(r, 0, chk)
            self.in_table.setItem(r, 1, name)
            self.in_table.setItem(r, 2, QTableWidgetItem(f"{a['low']:g}"))
            self.in_table.setItem(r, 3, QTableWidgetItem(f"{a['high']:g}"))
            self.in_table.setItem(r, 4, QTableWidgetItem(f"{(a['low'] + a['high']) / 2:g}"))
        for n in sig["observations"]:
            it = QListWidgetItem(n)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked)
            self.out_list.addItem(it)
        self.status.setText("")

    # -- options -----------------------------------------------------------------------------
    def options(self):
        """Dictionnaire d'options du worker ; leve ValueError si un champ est invalide."""
        steady = self.current_mode() == "steady"
        methods = [m for m, cb in self.method_boxes.items() if cb.isChecked()]
        if not methods:
            raise ValueError("Coche au moins une famille de modeles.")
        o = {"mode": self.current_mode(), "methods": methods, "poly_degrees": [int(v) for v in _floats(self.f_poly.text(), "Degres du polynome")],
             "sparse_thresholds": _floats(self.f_sparse.text(), "Seuils parcimonieux"),
             "hidden": [int(v) for v in _floats(self.f_hidden.text(), "Couches cachees")], "epochs": self.f_epochs.value(),
             "margin": self.f_margin.value() / 100.0, "seed": self.f_seed.value()}
        if min(o["poly_degrees"]) < 1 or min(o["hidden"]) < 1 or min(o["sparse_thresholds"]) <= 0:
            raise ValueError("Degres, neurones et seuils doivent etre positifs.")
        if steady:
            o.update(samples=self.s_samples.value(), val_samples=self.s_val.value(), extreme=self.s_extreme.value() / 100.0,
                     rtol=self.s_rtol.value() / 100.0, atol=self.s_atol.value(), window=self.s_window.value(),
                     max_settle_steps=self.s_settle.value() or None, scenario=self.s_scenario.isChecked())
        else:
            o.update(episodes=self.d_episodes.value(), val_episodes=self.d_val.value(), rounds=self.d_rounds.value(),
                     round_episodes=self.d_round_eps.value(), hold_max=self.d_hold.value(), extreme=self.d_extreme.value() / 100.0,
                     max_steps=self.d_maxsteps.value() or None)
        ranges, fixed = {}, {}
        for r in range(self.in_table.rowCount()):
            name_item = self.in_table.item(r, 1)
            name, (lo0, hi0) = name_item.text(), name_item.data(Qt.UserRole)
            try:
                lo, hi, val = (float(self.in_table.item(r, c).text().replace(",", ".")) for c in (2, 3, 4))
            except ValueError:
                raise ValueError(f"Entree {name} : Min, Max et Valeur fixee doivent etre des nombres.") from None
            if self.in_table.item(r, 0).checkState() == Qt.Checked:
                if not lo < hi:
                    raise ValueError(f"Entree {name} : Min doit etre inferieur a Max.")
                if (lo, hi) != (lo0, hi0):
                    ranges[name] = [lo, hi]
            else:
                fixed[name] = val
        if self.signals and len(fixed) == self.in_table.rowCount() and fixed:
            raise ValueError("Toutes les entrees sont fixees : il n'y a rien a balayer.")
        o["ranges"], o["fixed"] = ranges, fixed
        if steady and self.out_list.count():
            checked = [self.out_list.item(i).text() for i in range(self.out_list.count()) if self.out_list.item(i).checkState() == Qt.Checked]
            if not checked:
                raise ValueError("Coche au moins une sortie a modeliser.")
            if len(checked) < self.out_list.count():
                o["outputs"] = checked
        return o

    # -- reglages du projet (identify/options.json : voyagent avec le projet, .rls compris) ----------
    def _options_file(self):
        return identify_dir(self.project) / "options.json" if self.project else None

    def _save_options(self, opts):
        f = self._options_file()
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(opts, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_options(self):
        f = self._options_file()
        try:
            o = json.loads(f.read_text(encoding="utf-8-sig")) if f and f.is_file() else None
        except (OSError, ValueError):
            o = None
        if isinstance(o, dict):
            self.apply_options(o)

    def apply_options(self, o):
        """Remet dans l'interface des options sauvegardees (champs absents ou invalides ignores)."""
        def num(widget, key, scale=1.0):
            if key in o and o[key] is not None:
                try:
                    widget.setValue(type(widget.value())(o[key] * scale) if isinstance(widget, QSpinBox) else float(o[key]) * scale)
                except (TypeError, ValueError):
                    pass

        i = self.mode.findData(o.get("mode"))
        if i >= 0:
            self.mode.setCurrentIndex(i)
        for m, cb in self.method_boxes.items():
            if "methods" in o:
                cb.setChecked(m in o["methods"])
        for widget, key, scale in ((self.s_samples, "samples", 1), (self.s_val, "val_samples", 1), (self.s_extreme, "extreme", 100),
                                   (self.s_rtol, "rtol", 100), (self.s_atol, "atol", 1), (self.s_window, "window", 1),
                                   (self.f_epochs, "epochs", 1), (self.f_margin, "margin", 100), (self.f_seed, "seed", 1)):
            if self.current_mode() == "steady" or key not in ("extreme",):
                num(widget, key, scale)
        if self.current_mode() == "dynamic":
            for widget, key, scale in ((self.d_episodes, "episodes", 1), (self.d_val, "val_episodes", 1), (self.d_rounds, "rounds", 1),
                                       (self.d_round_eps, "round_episodes", 1), (self.d_hold, "hold_max", 1), (self.d_extreme, "extreme", 100),
                                       (self.d_maxsteps, "max_steps", 1)):
                num(widget, key, scale)
        self.s_settle.setValue(int(o.get("max_settle_steps") or 0))
        self.s_scenario.setChecked(bool(o.get("scenario", False)))
        for edit, key in ((self.f_poly, "poly_degrees"), (self.f_sparse, "sparse_thresholds"), (self.f_hidden, "hidden")):
            if o.get(key):
                edit.setText(", ".join(f"{v:g}" for v in o[key]))
        ranges, fixed = o.get("ranges") or {}, o.get("fixed") or {}
        for r in range(self.in_table.rowCount()):
            name = self.in_table.item(r, 1).text()
            if name in ranges:
                self.in_table.item(r, 2).setText(f"{ranges[name][0]:g}")
                self.in_table.item(r, 3).setText(f"{ranges[name][1]:g}")
            if name in fixed:
                self.in_table.item(r, 0).setCheckState(Qt.Unchecked)
                self.in_table.item(r, 4).setText(f"{fixed[name]:g}")
        if o.get("outputs"):
            for k in range(self.out_list.count()):
                self.out_list.item(k).setCheckState(Qt.Checked if self.out_list.item(k).text() in o["outputs"] else Qt.Unchecked)

    # -- calcul ------------------------------------------------------------------------------
    def _start(self):
        if not self.project:
            return
        try:
            opts = self.options()
        except ValueError as e:
            QMessageBox.information(self, "Identification", str(e))
            return
        self._save_options(opts)
        try:
            self.runner = IdentifyRunner(self.project, self.settings, opts, self)
            self.runner.log.connect(self._on_log)
            self.runner.progress.connect(self._on_progress)
            self.runner.finished.connect(self._on_finished)
            self.runner.start()
        except RuntimeMissing as e:
            self.runner = None
            QMessageBox.warning(self, "Composants manquants", str(e))
            return
        for w in (self.btn_start, self.mode, self.btn_read):
            w.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("")
        self.logbox.clear()
        self.logbox.show()
        self.stage.setText("Demarrage du processus de calcul ...")
        self._t0, self._fraction = time.monotonic(), 0.0
        self._tick()
        self.clock.start()
        self.busy.emit(True)

    def _stop(self):
        if self.runner:
            self.runner.stop()
            self.stage.setText("Arret...")

    def _on_log(self, text):
        self.logbox.appendPlainText(text.rstrip())

    def _on_progress(self, stage, fraction):
        self._fraction = fraction
        self.progress.setValue(int(fraction * 1000))
        self.stage.setText(stage)

    @staticmethod
    def _fmt(seconds):
        seconds = int(max(seconds, 0))
        return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"

    def _tick(self):
        if self._t0 is None:
            return
        elapsed = time.monotonic() - self._t0
        text = f"Ecoule : {self._fmt(elapsed)}"
        if self._fraction > 0.03:
            text += f"   ·   reste environ {self._fmt(elapsed * (1 - self._fraction) / self._fraction)}"
        self.timing.setText(text)

    def _on_finished(self, message):
        self.runner = None
        self.clock.stop()
        elapsed = time.monotonic() - self._t0 if self._t0 else 0
        self._t0 = None
        self.btn_stop.setEnabled(False)
        for w in (self.mode, self.btn_read):
            w.setEnabled(True)
        self.btn_start.setEnabled(self.project is not None)
        self.busy.emit(False)
        self.stage.setText("")
        self.timing.setText(f"Duree : {self._fmt(elapsed)}")
        if message.startswith("ERREUR"):
            self.logbox.appendPlainText(message)
            self.status.setText("Echec de l'identification : voir le journal ci-dessus.")
            return
        self.status.setText("Identification terminee." if message.startswith("Termine") else "Identification arretee.")
        if message.startswith("Termine"):
            self.progress.setValue(1000)
            self._refresh_results(select_latest=True)

    # -- resultats ---------------------------------------------------------------------------
    def _refresh_results(self, select_latest=False):
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        base = identify_dir(self.project) if self.project else None
        dirs = sorted((d for d in base.iterdir() if (d / "report.json").exists()), reverse=True) if base and base.exists() else []
        for d in dirs:
            try:
                rep = json.loads((d / "report.json").read_text(encoding="utf-8-sig"))
                kind = "regime permanent" if rep.get("mode") == "steady" else "dynamique"
            except ValueError:
                continue
            self.result_combo.addItem(f"{d.name[6:8]}/{d.name[4:6]}/{d.name[:4]}  {d.name[9:11]}:{d.name[11:13]}  -  {kind}", str(d))
        self.result_combo.blockSignals(False)
        if self.result_combo.count():
            self.result_combo.setCurrentIndex(0)
        self._load_report()

    def _load_report(self, *_):
        data = self.result_combo.currentData()
        self.report_dir = Path(data) if data else None
        try:
            self.report = json.loads((self.report_dir / "report.json").read_text(encoding="utf-8-sig")) if self.report_dir else None
        except (OSError, ValueError):
            self.report = None
        if self.report is None:
            self.table.setRowCount(0)
            self.equations.clear()
            self.notes.setText("")
            self.summary.setText("Aucune identification pour ce projet.")
            self._set_result_buttons(False)
            self._draw_plots()
            return
        self._fill_results()

    def _steady(self):
        return bool(self.report) and self.report.get("mode") == "steady"

    def _score(self, c):
        return c["r2"] if self._steady() else c["free_r2"]

    def _fill_results(self):
        rep, steady = self.report, self._steady()
        cols = STEADY_COLS if steady else DYN_COLS
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        if steady:
            self.summary.setText(f"{rep['n_train']} points d'apprentissage, {rep['n_val']} de validation, {rep['dropped']} ecarte(s) ; entrees : "
                                 f"{', '.join(rep['in_names'])} ; sorties : {', '.join(rep['out_names'])}. Du {rep['created']} ({rep['duration_s']:.0f} s).")
        else:
            self.summary.setText(f"{rep['n_train']} transitions d'apprentissage, {rep['n_val']} de validation, pas de temps {rep['dt']:g} s. "
                                 f"Du {rep['created']} ({rep['duration_s']:.0f} s).")
        cands = sorted(rep["candidates"], key=lambda c: c["complexity"])
        self.table.setRowCount(len(cands))
        for r, c in enumerate(cands):
            vals = ([c["label"], str(c["complexity"]), f"{c['r2']:.4f}", f"{c['nrmse']:.3f}", f"{c['max_abs_err']:.2f}"] if steady else
                    [c["label"], str(c["complexity"]), f"{c['r2_one_step']:.4f}", f"{c['free_r2']:.4f}", f"{c['nrmse']:.3f}"])
            vals.append("recommande" if c["id"] == rep["recommended"] else "")
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if 0 < k < 5:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                it.setData(Qt.UserRole, c["id"])
                self.table.setItem(r, k, it)
        self.plot_out.blockSignals(True)
        self.plot_out.clear()
        self.plot_out.addItems(rep["out_names"] if steady else rep["obs_names"])
        self.plot_out.setVisible(steady)
        self.plot_out.blockSignals(False)
        self.notes.setText("\n".join("• " + n for n in rep["notes"]))
        self._set_result_buttons(True)
        rec = next((r for r in range(self.table.rowCount()) if self.table.item(r, 0).data(Qt.UserRole) == rep["recommended"]), 0)
        self.table.selectRow(rec)
        self._draw_plots()

    def _candidate(self):
        if not self.report:
            return None
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        cid = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        return next((c for c in self.report["candidates"] if c["id"] == cid), None)

    def _show_candidate(self):
        c = self._candidate()
        self.equations.setPlainText(c["label"] + "\n\n" + "\n".join(c["equations"]) if c else "")
        self._draw_plots()

    def _draw_plots(self, *_):
        self.pareto.clear()
        self.detail.clear()
        if not self.report or "accent" not in C:
            return
        rep, sel, steady = self.report, self._candidate(), self._steady()
        txt = C["muted"]
        self.pareto.setLabel("bottom", "Complexite (termes ou parametres, echelle log)", color=txt)
        self.pareto.setLabel("left", "R2 (points neufs)" if steady else "Fidelite en simulation libre", color=txt)
        xs = [np.log10(max(c["complexity"], 1)) for c in rep["candidates"]]
        ys = [max(self._score(c), -0.2) for c in rep["candidates"]]
        brushes = [pg.mkBrush(C["success"] if c["id"] == rep["recommended"] else C["accent"]) for c in rep["candidates"]]
        self.pareto.addItem(pg.ScatterPlotItem(xs, ys, size=[16 if sel and c["id"] == sel["id"] else 10 for c in rep["candidates"]],
                                               brush=brushes, pen=pg.mkPen(C["text"], width=1)))
        self.pareto.addItem(pg.InfiniteLine(pos=1.0 - rep["options"]["margin"], angle=0, pen=pg.mkPen(C["warning"], style=Qt.DashLine, width=1.5)))
        for c, x, y in zip(rep["candidates"], xs, ys):
            t = pg.TextItem(c["id"], color=txt, anchor=(0, 1))
            t.setPos(x, y)
            self.pareto.addItem(t)
        if not sel:
            return
        if steady:
            pts = np.asarray(sel.get("scatter", {}).get(self.plot_out.currentText(), []))
            self.detail.setLabel("bottom", "Sortie du systeme", color=txt)
            self.detail.setLabel("left", "Sortie du modele", color=txt)
            if len(pts):
                lo, hi = float(pts.min()), float(pts.max())
                self.detail.addItem(pg.PlotDataItem([lo, hi], [lo, hi], pen=pg.mkPen(C["warning"], style=Qt.DashLine, width=1.5)))
                self.detail.addItem(pg.ScatterPlotItem(pts[:, 0], pts[:, 1], size=6, brush=pg.mkBrush(C["accent"]), pen=None))
        else:
            hs = sel.get("nrmse_by_horizon", {})
            self.detail.setLabel("bottom", "Horizon de prediction (pas)", color=txt)
            self.detail.setLabel("left", "Erreur normalisee (NRMSE)", color=txt)
            if hs:
                x = [float(h) for h in hs]
                self.detail.addItem(pg.PlotDataItem(np.log10(x), list(hs.values()), pen=pg.mkPen(C["accent"], width=2), symbol="o", symbolSize=7,
                                                    symbolBrush=C["accent"]))

    def _set_result_buttons(self, on):
        for b in (self.btn_copy_py, self.btn_copy_m, self.btn_save, self.btn_slx, self.btn_open):
            b.setEnabled(on)

    # -- exports -----------------------------------------------------------------------------
    def _stem(self):
        return "steady_state" if self._steady() else "dynamics"

    def _file(self, name):
        c = self._candidate()
        if not c or not self.report_dir:
            return None
        try:
            ensure_files(self.report_dir, c["id"])  # un projet importe (.rls) n'embarque pas le code genere
        except (OSError, ValueError, KeyError):
            pass
        return self.report_dir / c["id"] / name

    def _copy(self, ext):
        f = self._file(self._stem() + ext)
        if f and f.exists():
            QApplication.clipboard().setText(f.read_text(encoding="utf-8"))
            self.status.setText(f"{f.name} copie dans le presse-papiers.")

    def _save_as(self):
        c = self._candidate()
        if not c:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Enregistrer le modele", f"{c['id'].replace('-', '_')}.py",
                                                "Python (*.py);;MATLAB (*.m);;Equations (*.txt);;JSON (*.json)")
        if not target:
            return
        src = {".py": self._stem() + ".py", ".m": self._stem() + ".m", ".txt": "equations.txt", ".json": "model.json"}.get(Path(target).suffix.lower())
        f = self._file(src) if src else None
        if f is None or not f.exists():
            QMessageBox.warning(self, "Enregistrer", "Extension non prise en charge (.py, .m, .txt ou .json).")
            return
        text = f.read_text(encoding="utf-8")
        if src.endswith(".m"):  # MATLAB exige que le nom de fonction == nom de fichier
            text = text.replace(self._stem(), Path(target).stem)
        Path(target).write_text(text, encoding="utf-8")
        self.status.setText(f"Enregistre : {target}")

    def _export_simulink(self):
        c = self._candidate()
        if not c or not self.report_dir or not self.project:
            return
        suggestion = f"{'regime_permanent' if self._steady() else 'dynamique'}_{c['id'].replace('-', '_').replace('.', '_')}.slx"
        target, _ = QFileDialog.getSaveFileName(self, "Modele Simulink", suggestion, "Modele Simulink (*.slx)")
        if not target:
            return
        args = ["export-identify-simulink", "--project", str(self.project.path), "--report", self.report_dir.name, "--candidate", c["id"],
                "--out", target]
        self.job = WorkerJob(self.settings, args, self)
        self.job.done.connect(lambda res, err: self.status.setText(f"Modele Simulink cree : {res['path']}" if res else f"ERREUR : {err}"))
        self.job.done.connect(lambda *_: self.btn_slx.setEnabled(True))
        self.btn_slx.setEnabled(False)
        self.status.setText("Generation du modele Simulink (demarrage de MATLAB) ...")
        self.job.start()

    def _open_dir(self):
        if self.report_dir and sys.platform == "win32" and self.report_dir.exists():
            os.startfile(str(self.report_dir))
