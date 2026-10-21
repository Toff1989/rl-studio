"""Onglet « Evaluation » : rejouer un modele entraine (ou un modele extrait) et suivre observations/actions en direct."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QMessageBox, QPlainTextEdit, QSpinBox,
                               QVBoxLayout, QWidget)

from ..evaluation import policy_choices
from .live_plots import SeriesPlot
from .runner import EvalRunner, RuntimeMissing
from .theme import C
from .widgets import Card, Kpi, button, fmt_run_date, label

SPEEDS = [("Maximale", 0.0), ("Rapide (100 pas/s)", 0.01), ("Normale (30 pas/s)", 0.033), ("Lente (10 pas/s)", 0.1)]


class EvalTab(QWidget):
    busy = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings, self.project, self.runner = settings, None, None
        self.returns = []

        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._fill_policies)
        self.policy_combo = QComboBox()
        self.f_episodes = QSpinBox(); self.f_episodes.setRange(1, 1000); self.f_episodes.setValue(5)
        self.f_seed = QSpinBox(); self.f_seed.setRange(0, 1_000_000); self.f_seed.setValue(1000)
        self.speed = QComboBox()
        for txt, _ in SPEEDS:
            self.speed.addItem(txt)
        self.speed.setCurrentIndex(2)
        self.btn_start = button("▶  Lancer l'évaluation", "primary", self._start)
        self.btn_stop = button("■  Arrêter", "danger", self._stop)
        self.btn_stop.setEnabled(False)
        self.status = label("Choisis un run entraîné.", "muted", wrap=True)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        for i, (txt, w, span) in enumerate([("Run", self.run_combo, 1), ("Politique", self.policy_combo, 1),
                                            ("Épisodes", self.f_episodes, 1), ("Graine", self.f_seed, 1), ("Vitesse", self.speed, 1)]):
            form.addWidget(label(txt, "sectionLabel"), 0, i)
            form.addWidget(w, 1, i)
        form.setColumnStretch(0, 2)
        form.setColumnStretch(1, 3)
        bar = QHBoxLayout()
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.status, 1)
        cfg = Card("Évaluation d'un modèle", "Rejoue des épisodes avec la politique déterministe (sans bruit d'exploration) et "
                   "trace les observations et les actions pas à pas. Peut aussi rejouer un modèle mathématique extrait.")
        cfg.body.addLayout(form)
        cfg.body.addLayout(bar)

        self.k_ep, self.k_step, self.k_ret, self.k_mean = Kpi("Épisode"), Kpi("Pas"), Kpi("Retour en cours"), Kpi("Retour moyen")
        kpis = QHBoxLayout()
        for k in (self.k_ep, self.k_step, self.k_ret, self.k_mean):
            kpis.addWidget(k, 1)

        self.p_obs = SeriesPlot("Observations", "pas")
        self.p_act = SeriesPlot("Actions", "pas")
        self.p_cum = SeriesPlot("Récompense cumulée", "pas")
        self.p_cum.set_names(["récompense cumulée"])
        self.p_ret = pg.PlotWidget()
        self.p_ret.setObjectName("plot")
        self.p_ret.showGrid(x=True, y=True, alpha=0.12)
        self.p_ret.setMenuEnabled(False)
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush=C.get("accent", "#4f7cf0"))
        self.p_ret.addItem(self.bars)
        for p in (self.p_obs, self.p_act, self.p_cum, self.p_ret):
            p.setMinimumHeight(240)
        self.chk_norm = QCheckBox("Normaliser les courbes")
        self.chk_norm.toggled.connect(lambda f: (self.p_obs.set_normalized(f), self.p_act.set_normalized(f)))
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self.p_obs, 0, 0)
        grid.addWidget(self.p_act, 0, 1)
        grid.addWidget(self.p_cum, 1, 0)
        grid.addWidget(self.p_ret, 1, 1)

        self.logbox = QPlainTextEdit()
        self.logbox.setObjectName("mono")
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(5000)  # journal borne : il ne ralentit pas l'affichage
        self.logbox.setFixedHeight(80)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 20)
        lay.setSpacing(14)
        lay.addWidget(cfg)
        lay.addLayout(kpis)
        lay.addWidget(self.chk_norm)
        lay.addLayout(grid, 1)
        lay.addWidget(self.logbox)
        self.restyle()

    def restyle(self):
        for p in (self.p_obs, self.p_act, self.p_cum):
            p.restyle()
        if "surface" in C:
            self.p_ret.setBackground(C["surface"])
            self.p_ret.setTitle("Retour par épisode", color=C["text"], size="11pt")
            for ax in ("left", "bottom"):
                self.p_ret.getAxis(ax).setPen(C["border2"])
                self.p_ret.getAxis(ax).setTextPen(C["muted"])
            self.bars.setOpts(brush=C["accent"])

    # -- projet ----------------------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        runs = [d for d in (project.list_runs() if project else []) if (d / "model.zip").exists()]
        for d in runs:
            self.run_combo.addItem(fmt_run_date(d.name), str(d))
        self.run_combo.blockSignals(False)
        self._fill_policies()
        self.btn_start.setEnabled(bool(runs) and self.runner is None)
        self.status.setText("Prêt." if runs else "Aucun run avec modèle : entraîne le projet d'abord.")

    def _fill_policies(self, *_):
        self.policy_combo.clear()
        d = self.run_combo.currentData()
        if not d:
            return
        from pathlib import Path

        for pid, txt in policy_choices(Path(d)):
            self.policy_combo.addItem(txt, pid)

    # -- evaluation ------------------------------------------------------------------------
    def _reset_views(self):
        self.returns = []
        for p in (self.p_obs, self.p_act, self.p_cum):
            p.reset()
        self.bars.setOpts(x=[], height=[])
        for k in (self.k_ep, self.k_step, self.k_ret, self.k_mean):
            k.set("-")
        self.logbox.clear()
        self._episode = 0

    def _start(self):
        d, pid = self.run_combo.currentData(), self.policy_combo.currentData()
        if not d or not pid or not self.project:
            return
        from pathlib import Path

        self._reset_views()
        delay = SPEEDS[self.speed.currentIndex()][1]
        try:
            self.runner = EvalRunner(self.project, self.settings, Path(d).name, pid, self.f_episodes.value(),
                                     self.f_seed.value(), delay, self)
            self.runner.meta.connect(self._on_meta)
            self.runner.step.connect(self._on_step)
            self.runner.episode_done.connect(self._on_episode)
            self.runner.log.connect(self.logbox.appendPlainText)
            self.runner.finished.connect(self._on_finished)
            self.runner.start()
        except RuntimeMissing as e:
            self.runner = None
            QMessageBox.warning(self, "Composants de calcul", str(e))
            return
        for w in (self.btn_start, self.run_combo, self.policy_combo):
            w.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Évaluation en cours ...")
        self.busy.emit(True)

    def _stop(self):
        if self.runner:
            self.runner.stop()
            self.status.setText("Arrêt...")

    def _on_meta(self, obs_names, act_names):
        self.p_obs.set_names(obs_names)
        self.p_act.set_names(act_names)

    def _on_step(self, s):
        if s["episode"] != self._episode:  # nouvel episode : on repart d'un graphique vide
            self._episode = s["episode"]
            for p in (self.p_obs, self.p_act, self.p_cum):
                p.reset()
            self.k_ep.set(f"{s['episode']} / {self.f_episodes.value()}")
        self.p_obs.append(s["step"], s["obs"])
        self.p_act.append(s["step"], s["action"])
        self.p_cum.append(s["step"], [s["total"]])
        self.k_step.set(str(s["step"]))
        self.k_ret.set(f"{s['total']:.2f}")

    def _on_episode(self, ev):
        self.returns.append(ev["reward"])
        self.bars.setOpts(x=np.arange(1, len(self.returns) + 1), height=self.returns)
        self.k_mean.set(f"{np.mean(self.returns):.2f}")
        self.logbox.appendPlainText(f"Episode {ev['n']} : retour {ev['reward']:.3f}, {ev['length']} pas"
                                    + (" (arrêt anticipé)" if ev.get("terminated") else ""))

    def _on_finished(self, message):
        self.runner = None
        for w in (self.btn_start, self.run_combo, self.policy_combo):
            w.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.busy.emit(False)
        self.status.setText(message[:400] if message.startswith("ERREUR") else
                            "Évaluation terminée." if message.startswith("Terminé") else "Évaluation arrêtée.")
