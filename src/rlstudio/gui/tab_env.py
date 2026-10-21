"""Onglet Environnement (Python et Simulink)."""
import os
import shutil
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QMessageBox, QPlainTextEdit, QTabWidget,
)

from ..envs.spec import validate_spec
from .explorer import SimulinkExplorer
from .runner import WorkerJob
from .widgets import Card, button, column, label, scroll_page
from .spec_table import ACTION_COLS, INIT_COLS, OBS_COLS, SpecTable


class EnvTab:
    """Methodes de MainWindow (onglet environnement)."""

    def _tab_env(self):
        # Python
        self.py_entry = QLineEdit()
        self.py_params = QPlainTextEdit()
        self.py_params.setObjectName("mono")
        self.py_params.setMinimumHeight(340)
        entry = Card("Point d'entrée", "Fichier et fonction qui construisent l'environnement Gymnasium.")
        erow = QHBoxLayout()
        erow.addWidget(self.py_entry, 1)
        erow.addWidget(button("Ouvrir dans l'éditeur", None, self._open_env_file))
        entry.body.addLayout(erow)
        params = Card("Paramètres", "YAML passé à make_env(**params).")
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

        model = Card("Modèle Simulink")
        mrow = QHBoxLayout()
        mrow.addWidget(self.sl_model, 1)
        mrow.addWidget(button("Parcourir...", None, self._browse_model))
        mrow.addWidget(button("Explorer le modèle", "primary", self._explore_model))
        model.body.addLayout(mrow)
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setLabelAlignment(Qt.AlignLeft)
        form.addRow("Pas de décision", self.sl_step)
        form.addRow("Durée d'un épisode", self.sl_episode)
        model.body.addLayout(form)

        self.t_actions = SpecTable(ACTION_COLS, {"low": 0.0, "high": 1.0})
        self.t_obs = SpecTable(OBS_COLS, {"port": 1, "size": 1})
        self.t_init = SpecTable(INIT_COLS, {"low": 0.0, "high": 1.0})
        cards = [model]
        for title, hint, table in [
            ("Actions", "Paramètres de blocs ou variables que l'agent commande, avec leurs bornes.", self.t_actions),
            ("Observations", "Signaux (sorties de blocs) lus par l'agent à chaque pas.", self.t_obs),
            ("État initial aléatoire", "Variables du workspace tirées dans [min, max] à chaque épisode (optionnel).", self.t_init),
        ]:
            card = Card(title, hint)
            card.body.addWidget(table)
            row = QHBoxLayout()
            row.addWidget(button("+  Ajouter", "small", table.add_row))
            row.addWidget(button("Supprimer la sélection", "small", table.remove_selected))
            row.addStretch()
            card.body.addLayout(row)
            table.changed.connect(self._mark_dirty)
            table.changed.connect(self._update_vars_hint)
            cards.append(card)

        reward = Card("Récompense", "Expression évaluée à chaque pas ; les noms des actions et observations sont disponibles.")
        reward.body.addWidget(label("Récompense", "sectionLabel"))
        reward.body.addWidget(self.sl_reward)
        reward.body.addWidget(label("Arrêt anticipé", "sectionLabel"))
        reward.body.addWidget(self.sl_terminate)
        reward.body.addWidget(self.sl_vars)
        cards.insert(1, reward)

        self.sl_status = label("", "statusbox", wrap=True)
        self.sl_status.hide()
        check = QHBoxLayout()
        check.addWidget(button("Vérifier la configuration", None, self._check_spec))
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

    def _open_env_file(self):
        if not self.project:
            return
        filename = self.py_entry.text().partition(":")[0] or "env.py"
        path = self.project.path / filename
        if path.exists() and sys.platform == "win32":
            os.startfile(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Modèle Simulink", str(self.project.path), "Simulink (*.slx *.mdl)")
        if not path:
            return
        src = os.path.abspath(path)
        if not src.startswith(str(self.project.path)):
            dst = self.project.path / os.path.basename(src)
            shutil.copy2(src, dst)
            self._log(f"Modèle copie dans le projet : {dst}")
            src = str(dst)
        self.sl_model.setText(os.path.relpath(src, self.project.path))

    def _explore_model(self):
        if not self.sl_model.text().strip():
            QMessageBox.information(self, "Modèle", "Choisis d'abord un modèle Simulink.")
            return
        path = self.project.path / self.sl_model.text().strip()
        if not path.exists():
            QMessageBox.warning(self, "Modèle", f"Fichier introuvable : {path}")
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
            QMessageBox.critical(self, "MATLAB", f"Impossible d'inspecter le modèle :\n{err}")
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
