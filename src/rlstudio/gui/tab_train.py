"""Onglet Entraînement : graphiques en direct, journal, démarrage et arrêt."""
import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QMessageBox, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget,
)

from ..envs.spec import validate_spec
from ..training import fmt_duration
from .theme import C
from .live_plots import SeriesPlot
from .notify import looks_like_error, summarize
from .runner import RuntimeMissing, TrainRunner
from .widgets import Card, Kpi, button, label, tips


LOG_MAX_LINES = 5000


class TrainTab:
    """Méthodes de MainWindow (onglet Entraînement)."""

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
        self.btn_start = button("▶  Démarrer", "primary", self._start)
        self.btn_stop = button("■  Arrêter", "danger", self._stop)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.status = label("Prêt.", "muted")
        bar = QHBoxLayout()
        bar.setSpacing(10)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addSpacing(8)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.status)

        self.m_eps, self.m_steps = Kpi("Épisodes"), Kpi("Pas")
        self.m_last, self.m_best, self.m_base = Kpi("Dernière récompense"), Kpi("Meilleure"), Kpi("Référence")
        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        for k in (self.m_eps, self.m_steps, self.m_last, self.m_best, self.m_base):
            kpis.addWidget(k, 1)

        self.p_reward = self._plot("Récompense par épisode")
        self.c_reward = self.p_reward.plot(pen=pg.mkPen(C["accent"], width=1), symbol="o", symbolSize=4,
                                           symbolBrush=C["accent"], symbolPen=None)
        self.c_smooth = self.p_reward.plot(pen=pg.mkPen(C["warning"], width=3))
        self.base_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(C["danger"], width=1.5, style=Qt.DashLine))
        self.p_reward.addItem(self.base_line)
        self.base_line.hide()
        self.p_obs = SeriesPlot("Observations (épisode en cours)")
        self.p_act = SeriesPlot("Actions (épisode en cours)")
        self.p_obs.setMinimumHeight(260)
        self.p_act.setMinimumHeight(260)
        self.chk_norm = QCheckBox("Normaliser les courbes (comparer des signaux d'échelles différentes)")
        self.chk_norm.toggled.connect(lambda f: (self.p_obs.set_normalized(f), self.p_act.set_normalized(f)))
        self.p_rew_step = self._plot("Récompense par pas (épisode en cours)")
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
        self.logbox.setPlaceholderText("Le journal de l'entraînement s'affiche ici : démarrage de MATLAB, "
                                       "évaluation de référence, épisodes, vitesse et temps restant.")
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
        tips(self, {
            "btn_start": "Enregistre le projet puis lance l'entraînement. Le premier démarrage prend quelques secondes (chargement de PyTorch, de MATLAB pour un projet Simulink).",
            "btn_stop": "Interrompt l'entraînement tout de suite. Le modèle partiel est conservé s'il a pu être enregistré.",
            "progress": "Avancement : d'abord l'évaluation de référence, puis l'apprentissage (en pas sur la durée totale réglée dans l'onglet Algorithme).",
            "status": "État du calcul et temps restant estimé.",
            "m_eps": "Nombre d'épisodes (essais complets) terminés depuis le début de l'entraînement.",
            "m_steps": "Nombre de pas simulés depuis le début de l'apprentissage, à comparer à la durée réglée dans l'onglet Algorithme.",
            "m_last": "Récompense totale du dernier épisode terminé.",
            "m_best": "Meilleure récompense totale obtenue sur un épisode depuis le début.",
            "m_base": "Récompense moyenne de la politique de référence (régulateur classique du projet, sinon actions aléatoires). Objectif : faire mieux.",
            "chk_norm": "Ramène chaque signal à l'échelle 0–1 pour comparer des grandeurs d'ordres de grandeur très différents.",
            "p_reward": "Récompense totale de chaque épisode (points) et moyenne glissante sur 10 épisodes (courbe jaune). La ligne rouge pointillée est la référence. Une courbe qui monte puis se stabilise est bon signe.",
            "p_obs": "Observations (ce que l'agent voit) pendant l'épisode en cours. Cliquez sur un nom de la légende pour masquer ou afficher une courbe.",
            "p_act": "Actions (ce que l'agent commande) pendant l'épisode en cours. Cliquez sur un nom de la légende pour masquer ou afficher une courbe.",
            "p_rew_step": "Récompense obtenue à chaque pas de l'épisode en cours.",
            "logbox": "Journal détaillé de l'entraînement : démarrage, évaluation de référence, une ligne par épisode (vitesse, temps restant). Les erreurs s'affichent aussi dans le bandeau en haut de la fenêtre.",
        })
        self._reset_train_state()
        return page

    def _reset_train_state(self):
        self.xs, self.rewards = [], []
        self.c_reward.setData([], [])
        self.c_smooth.setData([], [])
        self.base_line.hide()
        self.progress.setValue(0)
        self.status.setText("Prêt.")
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
            if looks_like_error(line):  # une erreur écrite dans le journal doit aussi se voir ailleurs que dans le journal
                self.report_error(summarize(line), str(text))

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
        self._log("Lancement de l'entraînement...")
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
        self.status.setText("Démarrage de l'environnement...")

    def _on_progress(self, stage, fraction):
        if stage == "baseline":
            self._phase_base = True
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
            self.status.setText(f"Évaluation de référence : {fraction * 100:.0f} %")

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("Arrêt...")

    def _on_meta(self, obs_names, act_names):
        self.p_obs.set_names(obs_names)
        self.p_act.set_names(act_names)

    def _on_baseline(self, mean, lbl):
        self.base_line.setValue(mean)
        self.base_line.show()
        self.m_base.set(f"{mean:.2f}")
        self._log(f"Référence tracée en rouge : {lbl}")

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
            self.status.setText(f"Épisode {n} - reste ~{fmt_duration(left)}")
        else:
            self.status.setText(f"Épisode {n}")
        self._log(f"Épisode {n} | {length} pas | récompense {reward:.2f} | meilleure {max(self.rewards):.2f}{extra}")
        self._reset_trace()

    def _on_state(self, s):
        if self._t_learn is None:  # premier pas d'apprentissage : fin de la phase de preparation
            self._t_learn, self._steps_learn0 = time.monotonic(), s["timesteps"] - 1
            self._log(f"Apprentissage démarré (préparation : {fmt_duration(self._t_learn - self._t_run)}).")
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
        self._log(message + f"  (durée totale : {fmt_duration(time.monotonic() - self._t_run)})")
        if message.startswith("ERREUR"):
            self.status.setText("Erreur : " + summarize(message))
            self.report_error("Échec de l'entraînement : " + summarize(message), message)
        else:
            self.status.setText("Terminé." if message.startswith("Terminé") else "Arrêté.")
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.plist.setEnabled(True)
        self._refresh_runs()
        self.distill.set_project(self.project)
        self.identify.set_project(self.project)
        self.evaltab.set_project(self.project)
        self._update_dashboard()
