"""Onglet « Evaluation » : rejouer un modele entraine (ou un modele extrait) et suivre observations/actions en direct."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QMessageBox, QPlainTextEdit, QSpinBox,
                               QVBoxLayout, QWidget)

from ..evaluation import MODEL_SEP, identified_choices, policy_choices
from .live_plots import MODEL_SUFFIX, SeriesPlot
from .notify import looks_like_error, summarize
from .runner import EvalRunner, RuntimeMissing
from .theme import C
from .widgets import Card, Kpi, button, fmt_run_date, label, tips

SPEEDS = [("Maximale", 0.0), ("Rapide (100 pas/s)", 0.01), ("Normale (30 pas/s)", 0.033), ("Lente (10 pas/s)", 0.1)]


class EvalTab(QWidget):
    busy = Signal(bool)
    error = Signal(str, str)  # (résumé, détail) : une erreur à montrer dans le bandeau de la fenêtre principale

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings, self.project, self.runner = settings, None, None
        self.returns = []
        self.model_names, self.model_errors = [], []

        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._fill_policies)
        self.policy_combo = QComboBox()
        self.f_episodes = QSpinBox(); self.f_episodes.setRange(1, 1000); self.f_episodes.setValue(5)
        self.f_seed = QSpinBox(); self.f_seed.setRange(0, 1_000_000); self.f_seed.setValue(1000)
        self.model_combo = QComboBox()
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
        form.addWidget(label("Modèle identifié à comparer", "sectionLabel"), 2, 0, 1, 2)
        form.addWidget(self.model_combo, 3, 0, 1, 5)
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
        self.k_err = Kpi("Écart du modèle")
        kpis = QHBoxLayout()
        for k in (self.k_ep, self.k_step, self.k_ret, self.k_mean, self.k_err):
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
        self.logbox.setFixedHeight(110)
        tips(self, {
            "run_combo": "Entraînement à rejouer (seuls ceux qui ont un modèle enregistré sont proposés).",
            "policy_combo": "Qui pilote le système : l'agent entraîné, un modèle mathématique extrait de cet entraînement (onglet Modèle mathématique), "
                            "ou des paliers d'actions aléatoires qui ne demandent aucun entraînement (pratique pour comparer un modèle identifié).",
            "model_combo": "Facultatif : rejoue un modèle identifié (onglet Identification) à côté du vrai système, avec les mêmes actions. "
                           "Ses sorties sont tracées en pointillé, de la même couleur que le signal réel. Modèle dynamique : simulation libre depuis "
                           "l'observation de départ. Modèle en régime permanent : y = f(u) appliqué aux actions courantes ; il réagit sans délai, "
                           "alors que le vrai système met du temps à s'établir, d'où un écart pendant les transitions.",
            "k_err": "Écart moyen entre le modèle identifié et le système, par épisode : erreur quadratique moyenne divisée par l'écart-type du signal réel "
                     "(0 = identique ; proche de 1 = pas mieux qu'une valeur constante). Le détail par signal est dans le journal.",
            "f_episodes": "Nombre d'épisodes (essais complets) à rejouer.",
            "f_seed": "Fixe les situations de départ : la même graine redonne les mêmes départs, ce qui permet de comparer deux politiques sur les mêmes cas.",
            "speed": "Vitesse de la lecture. « Maximale » va le plus vite possible ; les vitesses plus lentes permettent de suivre les courbes en direct.",
            "btn_start": "Rejoue les épisodes avec la politique choisie (sans bruit d'exploration) et trace les courbes en direct.",
            "btn_stop": "Interrompt l'évaluation en cours.",
            "status": "État de l'évaluation.",
            "k_ep": "Épisode en cours sur le nombre demandé.",
            "k_step": "Numéro du pas dans l'épisode en cours.",
            "k_ret": "Somme des récompenses de l'épisode en cours.",
            "k_mean": "Moyenne des retours des épisodes terminés.",
            "p_obs": "Observations (ce que l'agent voit) pendant l'épisode en cours ; en pointillé, les sorties du modèle identifié s'il y en a un. "
                     "Cliquez sur la légende pour masquer ou afficher une courbe.",
            "p_act": "Actions (ce que l'agent commande) pendant l'épisode en cours. Cliquez sur la légende pour masquer ou afficher une courbe.",
            "p_cum": "Récompense cumulée depuis le début de l'épisode en cours.",
            "p_ret": "Retour (somme des récompenses) de chaque épisode terminé.",
            "chk_norm": "Ramène chaque signal à l'échelle 0–1 pour comparer des grandeurs très différentes.",
            "logbox": "Un résumé par épisode terminé. Les erreurs s'affichent aussi dans le bandeau en haut de la fenêtre.",
        })

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
        self.refresh_models()
        self.btn_start.setEnabled(project is not None and self.runner is None)  # les paliers d'actions n'exigent aucun entraînement
        self.status.setText("Prêt." if runs else "Aucun entraînement avec modèle : seuls les paliers d'actions aléatoires sont proposés.")

    def refresh_models(self):
        """Recharge la liste des modèles identifiés du projet (appelée aussi à la fin d'une identification)."""
        current = self.model_combo.currentData()
        self.model_combo.clear()
        self.model_combo.addItem("Aucun", None)
        for mid, txt in (identified_choices(self.project) if self.project else []):
            self.model_combo.addItem(txt, mid)
        i = self.model_combo.findData(current)
        self.model_combo.setCurrentIndex(max(i, 0))

    def _fill_policies(self, *_):
        self.policy_combo.clear()
        d = self.run_combo.currentData()
        from pathlib import Path

        for pid, txt in policy_choices(Path(d) if d else None):
            self.policy_combo.addItem(txt, pid)

    # -- evaluation ------------------------------------------------------------------------
    def _reset_views(self):
        self.returns = []
        for p in (self.p_obs, self.p_act, self.p_cum):
            p.reset()
        self.bars.setOpts(x=[], height=[])
        for k in (self.k_ep, self.k_step, self.k_ret, self.k_mean, self.k_err):
            k.set("-")
        self.logbox.clear()
        self._episode = 0
        self.model_errors = []

    def _start(self):
        d, pid = self.run_combo.currentData(), self.policy_combo.currentData()
        if not pid or not self.project or (pid != "steps" and not d):
            return
        from pathlib import Path

        ident = self.model_combo.currentData()
        identified = tuple(ident.split(MODEL_SEP, 1)) if ident else None

        self._reset_views()
        delay = SPEEDS[self.speed.currentIndex()][1]
        try:
            self.runner = EvalRunner(self.project, self.settings, Path(d).name if d and pid != "steps" else "", pid,
                                     self.f_episodes.value(), self.f_seed.value(), delay, self, identified=identified)
            self.model_names = []
            self.runner.model_names.connect(lambda names: setattr(self, "model_names", names))
            self.runner.meta.connect(self._on_meta)
            self.runner.step.connect(self._on_step)
            self.runner.episode_done.connect(self._on_episode)
            self.runner.log.connect(self._on_log)
            self.runner.finished.connect(self._on_finished)
            self.runner.start()
        except RuntimeMissing as e:
            self.runner = None
            QMessageBox.warning(self, "Composants de calcul", str(e))
            return
        for w in (self.btn_start, self.run_combo, self.policy_combo, self.model_combo):
            w.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Évaluation en cours ...")
        self.busy.emit(True)

    def _stop(self):
        if self.runner:
            self.runner.stop()
            self.status.setText("Arrêt...")

    def _on_meta(self, obs_names, act_names):
        # le signal `model_names` est émis avant `meta` (même ligne du flux) : les courbes du modèle suivent celles du système
        self.p_obs.set_names(list(obs_names) + [n + MODEL_SUFFIX for n in self.model_names])
        self.p_act.set_names(act_names)

    def _on_step(self, s):
        if s["episode"] != self._episode:  # nouvel episode : on repart d'un graphique vide
            self._episode = s["episode"]
            for p in (self.p_obs, self.p_act, self.p_cum):
                p.reset()
            self.k_ep.set(f"{s['episode']} / {self.f_episodes.value()}")
        model = s.get("model")
        self.p_obs.append(s["step"], s["obs"] if model is None else np.r_[s["obs"], model])
        self.p_act.append(s["step"], s["action"])
        self.p_cum.append(s["step"], [s["total"]])
        self.k_step.set(str(s["step"]))
        self.k_ret.set(f"{s['total']:.2f}")

    def _on_episode(self, ev):
        self.returns.append(ev["reward"])
        self.bars.setOpts(x=np.arange(1, len(self.returns) + 1), height=self.returns)
        self.k_mean.set(f"{np.mean(self.returns):.2f}")
        self.logbox.appendPlainText(f"Épisode {ev['n']} : retour {ev['reward']:.3f}, {ev['length']} pas"
                                    + (" (arrêt anticipé)" if ev.get("terminated") else ""))
        err = ev.get("model_error")
        if err:
            mean = float(np.nanmean([e["nrmse"] for e in err.values()]))
            self.model_errors.append(mean)
            self.k_err.set(f"{np.mean(self.model_errors):.3f}")
            self.logbox.appendPlainText("   écart du modèle : " + ", ".join(f"{n} {e['nrmse']:.3f}" for n, e in err.items()))

    def _on_log(self, text):
        self.logbox.appendPlainText(text.rstrip())
        if looks_like_error(text):
            self.error.emit(summarize(text), text)

    def _on_finished(self, message):
        self.runner = None
        for w in (self.btn_start, self.run_combo, self.policy_combo, self.model_combo):
            w.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.busy.emit(False)
        if message.startswith("ERREUR"):
            self.logbox.appendPlainText(message)
            self.status.setText("Échec de l'évaluation : " + summarize(message))
            self.error.emit("Échec de l'évaluation : " + summarize(message), message)
        else:
            self.status.setText("Évaluation terminée." if message.startswith("Terminé") else "Évaluation arrêtée.")
