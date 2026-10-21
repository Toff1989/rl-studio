"""Fenêtre « Convertir un modèle Simulink » : prépare un modèle existant pour RL Studio et crée le projet.

Quatre étapes : modèle (analyse), actions, observations et variables initiales, réglages (pas, récompense).
Le modèle d'origine n'est jamais modifié : le worker (`convert-model`) en enregistre une copie convertie dans un nouveau
projet, puis la teste à vide. La logique (validation, project.yaml) est dans rlstudio/convert.py.
"""
import json
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..convert import sanitize_name, validate_conversion
from .explorer import SimulinkExplorer
from .runner import WorkerJob
from .theme import C
from .widgets import Card, button, form_row, header_tips, label, tip, tips

STEPS = ["Modèle", "Actions", "Observations", "Réglages"]

COL_TIPS = {
    "use": "Décochez pour ne pas utiliser cette ligne.",
    "name": "Nom donné à ce signal dans le projet (lettres, chiffres et _ ; utilisable dans la récompense).",
    "source": "Origine du signal dans le modèle Simulink.",
    "kind": "Type de l'entrée du modèle.",
    "default": "Valeur de l'action avant que l'agent ne commande (valeur initiale).",
    "low": "Valeur minimale (borne basse).",
    "high": "Valeur maximale (borne haute).",
    "port": "Numéro du port de sortie du bloc (1 pour le premier).",
    "variable": "Variable du modèle qui sera tirée au hasard au début de chaque épisode.",
    "value": "Valeur actuelle de la variable dans le modèle.",
}

class RowTable(QTableWidget):
    """Table de lignes cochables. `cols` : [(cle, titre, genre)], genre = "check" | "text" | "float" | "ro"."""

    def __init__(self, cols, height=(150, 260)):
        super().__init__(0, len(cols))
        self.cols = cols
        self.setHorizontalHeaderLabels([c[1] for c in cols])
        header_tips(self, [COL_TIPS.get(c[0]) for c in cols])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(32)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setShowGrid(False)
        self.setMinimumHeight(height[0])
        widths = {"use": 60, "name": 130, "source": 250, "kind": 130, "default": 100, "low": 80, "high": 80, "port": 50,
                  "variable": 140, "value": 110}
        for j, (key, _, _) in enumerate(cols[:-1]):
            self.setColumnWidth(j, widths.get(key, 100))
        self.meta = []  # donnees associees a chaque ligne (non affichees)

    def add_row(self, values: dict, meta=None, checked=True):
        row = self.rowCount()
        self.insertRow(row)
        for j, (key, _, kind) in enumerate(self.cols):
            if kind == "check":
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            else:
                v = values.get(key)
                item = QTableWidgetItem("" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)))
                if kind == "ro":
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if COL_TIPS.get(key):
                item.setToolTip(COL_TIPS[key])
            self.setItem(row, j, item)
        self.meta.append(meta or {})

    def remove_selected(self):
        for idx in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(idx)
            del self.meta[idx]

    def rows(self, only_checked=True):
        """[(valeurs, meta)] ; leve ValueError si un nombre est mal ecrit."""
        out = []
        for r in range(self.rowCount()):
            vals = {}
            use = True
            for j, (key, _, kind) in enumerate(self.cols):
                it = self.item(r, j)
                if kind == "check":
                    use = it.checkState() == Qt.Checked
                elif kind == "float":
                    try:
                        vals[key] = float(it.text().replace(",", "."))
                    except ValueError:
                        raise ValueError(f"Ligne {r + 1} : « {it.text()} » n'est pas un nombre ({self.cols[j][1]}).")
                else:
                    vals[key] = it.text().strip()
            if use or not only_checked:
                out.append((vals, self.meta[r]))
        return out


class ConvertDialog(QDialog):
    def __init__(self, parent, settings, store):
        super().__init__(parent)
        self.settings, self.store = settings, store
        self.created_slug = None
        self.analysis = None
        self._job = None
        self._spec_file = None
        self.setWindowTitle("Convertir un modèle Simulink")
        self.resize(1000, 740)

        self.steps_lbl = QLabel()
        self.steps_lbl.setTextFormat(Qt.RichText)
        self.stack = QStackedWidget()
        for page in (self._page_model(), self._page_actions(), self._page_observations(), self._page_settings()):
            self.stack.addWidget(page)
        self.btn_back = button("Précédent", None, self._back, tip="Revient à l'étape précédente.")
        self.btn_next = button("Suivant", "primary", self._next, tip="Passe à l'étape suivante. Sur la dernière étape, crée le projet converti et le teste à vide.")
        self.btn_cancel = button("Annuler", "ghost", self.reject, tip="Ferme l'assistant sans rien créer.")
        self.status = label("", "muted", wrap=True, tip="Progression de l'analyse ou de la conversion.")
        foot = QHBoxLayout()
        foot.addWidget(self.btn_cancel)
        foot.addWidget(self.status, 1)
        foot.addWidget(self.btn_back)
        foot.addWidget(self.btn_next)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)
        lay.addWidget(label("Convertir un modèle Simulink", "h1"))
        lay.addWidget(label("Prépare un modèle existant pour RL Studio. Le fichier d'origine n'est pas modifié : "
                            "une copie convertie est enregistrée dans un nouveau projet.", "hint", wrap=True))
        lay.addWidget(self.steps_lbl)
        lay.addWidget(self.stack, 1)
        lay.addLayout(foot)
        self._go(0)

    # ------------------------------------------------------------------ pages
    def _page_model(self):
        self.f_model = QLineEdit()
        self.f_model.setPlaceholderText("Modèle Simulink (.slx ou .mdl)")
        self.f_init = QLineEdit()
        self.f_init.setPlaceholderText("Optionnel : script .m ou fichier .mat qui définit les variables du modèle")
        self.btn_analyze = button("Analyser le modèle", "primary", self._analyze,
                                  tip="Ouvre le modèle dans MATLAB pour repérer ses entrées, ses sorties, son solveur et les variables manquantes (quelques dizaines de secondes).")
        tips(self, {
            "f_model": "Fichier du modèle Simulink à convertir (.slx ou .mdl). Il n'est jamais modifié : une copie est créée dans le nouveau projet.",
            "f_init": "Facultatif : script .m ou fichier .mat qui définit les variables dont dépend le modèle. Il est exécuté tel quel dans MATLAB avant l'analyse.",
        })
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        r1.addWidget(self.f_model, 1)
        r1.addWidget(button("Parcourir...", None, lambda: self._browse(self.f_model, "Modèle Simulink", "Simulink (*.slx *.mdl)"),
                            tip="Choisit le fichier du modèle Simulink."))
        r2.addWidget(self.f_init, 1)
        r2.addWidget(button("Parcourir...", None, lambda: self._browse(self.f_init, "Script d'initialisation",
                                                                        "MATLAB (*.m *.mat)"),
                            tip="Choisit le script .m ou le fichier .mat d'initialisation."))
        pick = Card("Modèle à convertir", "Le script d'initialisation est exécuté tel quel dans MATLAB avant l'analyse : "
                                          "indique-le si le modèle utilise des variables définies hors du .slx.")
        pick.body.addLayout(r1)
        pick.body.addLayout(r2)
        pick.body.addWidget(self.btn_analyze, 0, Qt.AlignLeft)
        self.summary = label("", wrap=True)
        self.summary.setTextFormat(Qt.RichText)
        self.warnings = QPlainTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setObjectName("mono")
        self.warnings.setMaximumHeight(110)
        self.warnings.setPlaceholderText("Les avertissements de l'analyse s'affichent ici.")
        self.removable = QListWidget()
        self.removable.setMaximumHeight(120)
        tips(self, {
            "summary": "Résumé de l'analyse : entrées, sorties, constantes, solveur.",
            "warnings": "Avertissements de l'analyse (variables manquantes, signaux de bus non pris en charge…).",
            "removable": "Blocs d'affichage (scopes…) ou liens cassés qui seront retirés de la copie pour accélérer les simulations.",
        })
        res = Card("Résultat de l'analyse")
        res.body.addWidget(self.summary)
        res.body.addWidget(self.warnings)
        res.body.addWidget(label("Blocs à retirer de la copie (affichages, liens cassés) :", "hint"))
        res.body.addWidget(self.removable)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        v.addWidget(pick)
        v.addWidget(res)
        v.addStretch(1)
        return w

    def _page_actions(self):
        self.t_actions = RowTable([("use", "Utiliser", "check"), ("name", "Nom", "text"), ("source", "Source", "ro"),
                                   ("kind", "Type", "ro"), ("default", "Valeur init.", "float"),
                                   ("low", "Min", "float"), ("high", "Max", "float")])
        row = QHBoxLayout()
        row.addWidget(button("Ajouter un paramètre de bloc...", None, self._explore,
                             tip="Ouvre l'arbre des blocs du modèle pour choisir un paramètre numérique que l'agent pilotera (ex. le gain d'un bloc Gain)."))
        row.addWidget(button("Retirer la ligne", "ghost", self.t_actions.remove_selected, tip="Retire du tableau la ligne sélectionnée."))
        self.t_actions.setToolTip("Actions proposées à partir des entrées du modèle. Décochez « Utiliser » pour en écarter une ; double-cliquez pour changer un nom ou des bornes.")
        row.addStretch(1)
        card = Card("Actions : ce que l'agent commande",
                    "Chaque entrée du modèle devient une constante pilotable par l'agent ; un paramètre de bloc "
                    "(ex. le gain d'un bloc Gain) est modifié en direct. Le min et le max bornent l'action.")
        card.body.addWidget(self.t_actions, 1)
        card.body.addLayout(row)
        return card

    def _page_observations(self):
        self.t_obs = RowTable([("use", "Utiliser", "check"), ("name", "Nom", "text"), ("source", "Source", "ro"),
                               ("port", "Port", "float")])
        orow = QHBoxLayout()
        orow.addWidget(button("Ajouter un signal...", None, self._explore,
                              tip="Ouvre l'arbre des blocs du modèle pour choisir un signal supplémentaire à observer."))
        orow.addWidget(button("Retirer la ligne", "ghost", self.t_obs.remove_selected, tip="Retire du tableau la ligne sélectionnée."))
        self.t_obs.setToolTip("Observations proposées à partir des sorties du modèle. Décochez « Utiliser » pour en écarter une.")
        orow.addStretch(1)
        cobs = Card("Observations : ce que l'agent voit",
                    "Les sorties du modèle sont proposées ; tu peux ajouter n'importe quel signal. Une sortie est "
                    "remplacée par le bloc qui l'alimente.")
        cobs.body.addWidget(self.t_obs, 1)
        cobs.body.addLayout(orow)
        self.t_init = RowTable([("use", "Utiliser", "check"), ("variable", "Variable", "ro"), ("value", "Valeur actuelle", "ro"),
                                ("low", "Min", "float"), ("high", "Max", "float")])
        cinit = Card("Variables initiales aléatoires (optionnel)",
                     "Variables du modèle tirées au hasard à chaque épisode (ex. angle de départ). Les variables définies "
                     "hors du .slx sont de toute façon copiées dans le modèle converti.")
        self.t_init.setToolTip("Variables du modèle tirées au hasard dans [Min, Max] au début de chaque épisode, pour varier les situations de départ. Cochez celles à utiliser.")
        cinit.body.addWidget(self.t_init, 1)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        v.addWidget(cobs, 3)
        v.addWidget(cinit, 2)
        return w

    def _page_settings(self):
        self.f_name = QLineEdit()
        self.s_step = QDoubleSpinBox()
        self.s_step.setDecimals(4)
        self.s_step.setRange(0.0001, 1e6)
        self.s_step.setValue(1.0)
        self.s_step.setSuffix(" s")
        self.s_episode = QDoubleSpinBox()
        self.s_episode.setDecimals(2)
        self.s_episode.setRange(0.01, 1e9)
        self.s_episode.setValue(100.0)
        self.s_episode.setSuffix(" s")
        self.c_solver = QComboBox()
        self.c_solver.addItem("Pas fixe (recommandé : l'état se conserve d'un pas de décision à l'autre)", "fixed")
        self.c_solver.addItem("Garder le solveur du modèle", "keep")
        self.s_solver = QDoubleSpinBox()
        self.s_solver.setDecimals(6)
        self.s_solver.setRange(1e-6, 1e6)
        self.s_solver.setValue(0.01)
        self.s_solver.setSuffix(" s")
        self.f_reward = QLineEdit()
        self.f_reward.setPlaceholderText("ex. -(y - 1.0)**2 - 0.01*u**2")
        self.f_term = QLineEdit()
        self.f_term.setPlaceholderText("optionnel, ex. abs(y) > 50")
        self.names_hint = label("", "hint", wrap=True)
        self.check_lbl = label("", wrap=True)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setSpacing(10)
        form_row(form, "Nom du projet", self.f_name, "Nom du nouveau projet qui contiendra la copie convertie du modèle.")
        form_row(form, "Pas de décision", self.s_step,
                 "Temps simulé entre deux décisions de l'agent. Plus il est petit, plus l'agent réagit finement, mais plus l'entraînement est long.")
        form_row(form, "Durée d'un épisode", self.s_episode, "Temps simulé d'un essai complet, du départ à la fin (sauf arrêt anticipé).")
        form_row(form, "Solveur", self.c_solver,
                 "Pas fixe : recommandé, car l'état du modèle se conserve d'un pas de décision à l'autre. « Garder le solveur du modèle » conserve ses réglages d'origine.")
        form_row(form, "Pas du solveur", self.s_solver, "Pas de calcul interne du solveur à pas fixe. Il doit diviser le pas de décision.")
        c1 = Card("Réglages", "Le pas de décision est le temps simulé entre deux actions de l'agent.")
        c1.body.addLayout(form)
        form2 = QFormLayout()
        form2.setSpacing(10)
        form_row(form2, "Récompense", self.f_reward,
                 "Expression calculée à chaque pas : plus elle est grande, mieux c'est. Utilisez les noms des actions et des observations, plus t, dt et step.")
        form_row(form2, "Arrêt anticipé", self.f_term,
                 "Condition facultative qui termine l'épisode dès qu'elle est vraie (ex. abs(y) > 50).")
        tips(self, {"names_hint": "Noms utilisables dans la récompense et l'arrêt anticipé.",
                    "check_lbl": "Résultat de la vérification de la configuration."})
        c2 = Card("Récompense", "Expression calculée à chaque pas : plus elle est grande, mieux c'est.")
        c2.body.addLayout(form2)
        c2.body.addWidget(self.names_hint)
        c2.body.addWidget(self.check_lbl)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        v.addWidget(c1)
        v.addWidget(c2)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ navigation
    def _go(self, i):
        self.stack.setCurrentIndex(i)
        accent, muted = C.get("accent", "#4f7cf0"), C.get("muted", "#888888")
        parts = []
        for k, name in enumerate(STEPS):
            col = accent if k == i else muted
            weight = "bold" if k == i else "normal"
            parts.append(f'<span style="color:{col}; font-weight:{weight}">{k + 1}  {name}</span>')
        self.steps_lbl.setText("&nbsp;&nbsp;&rsaquo;&nbsp;&nbsp;".join(parts))
        self.btn_back.setEnabled(i > 0 and self._job is None)
        last = i == len(STEPS) - 1
        self.btn_next.setText("Convertir et créer le projet" if last else "Suivant")
        self.btn_next.setEnabled(self._job is None and (i > 0 or self.analysis is not None))
        if i == 3:
            self._prepare_settings()

    def _back(self):
        self._go(self.stack.currentIndex() - 1)

    def _next(self):
        i = self.stack.currentIndex()
        try:
            if i == 1:
                self._actions_spec()
            elif i == 2:
                self._obs_spec()
                self._initial_spec()
        except ValueError as e:
            QMessageBox.warning(self, "Convertir un modèle", str(e))
            return
        if i == len(STEPS) - 1:
            self._convert()
        else:
            self._go(i + 1)

    def _browse(self, field, title, flt):
        start = field.text() or (self.f_model.text() if field is self.f_init else "") or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, flt)
        if path:
            field.setText(path)
            if field is self.f_model:
                self.analysis = None
                self.summary.setText("")
                self._go(0)

    def _busy(self, on, text=""):
        if on:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
        self.status.setText(text)
        for w in (self.btn_analyze, self.btn_back, self.btn_next):
            w.setEnabled(not on)

    # ------------------------------------------------------------------ analyse
    def _analyze(self):
        model = self.f_model.text().strip()
        if not model or not Path(model).is_file():
            QMessageBox.warning(self, "Convertir un modèle", "Choisis d'abord un fichier de modèle Simulink.")
            return
        args = ["analyze-model", "--model", model]
        if self.f_init.text().strip():
            args += ["--init", self.f_init.text().strip()]
        self._busy(True, "Analyse du modèle dans MATLAB (quelques dizaines de secondes)...")
        self._job = WorkerJob(self.settings, args, self)
        self._job.done.connect(self._analyzed)
        self._job.start()

    def _analyzed(self, res, err):
        self._job = None
        self._busy(False)
        if res is None:
            self.status.setText("")
            QMessageBox.critical(self, "Analyse impossible", err or "Erreur inconnue.")
            self._go(0)
            return
        self.analysis = res["analysis"]
        self._fill(res["analysis"], res["proposal"])
        self._go(0)

    def _fill(self, a, proposal):
        n_in, n_out = len(a.get("inports") or []), len(a.get("outports") or [])
        n_const, n_var = len(a.get("constants") or []), len(a.get("variables") or [])
        sol = a.get("solver") or {}
        self.summary.setText(
            f"<b>{a.get('model', '')}</b> : {n_in} entrée(s), {n_out} sortie(s), {n_const} constante(s) numérique(s), "
            f"{n_var} variable(s) utilisée(s).<br>Solveur : {sol.get('type', '?')} ({sol.get('name', '?')}), "
            f"pas {sol.get('step', '?')}.")
        self.warnings.setPlainText("\n".join(a.get("warnings") or []) or "Aucun avertissement.")
        self.removable.clear()
        for b in a.get("removable") or []:
            it = QListWidgetItem(f"{b['path']}   ({b['type']})")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, b["path"])
            self.removable.addItem(it)
        # actions : entrees convertibles (proposees) puis constantes numeriques (non cochees : a choisir)
        self.t_actions.setRowCount(0)
        self.t_actions.meta = []
        for act in proposal.get("actions") or []:
            self.t_actions.add_row({"name": act["name"], "source": act["path"], "kind": "Entrée du modèle",
                                    "default": float(act.get("default", 0.0)), "low": float(act["low"]),
                                    "high": float(act["high"])}, meta=act)
        taken = {x["name"] for x in proposal.get("actions") or []} | {x["name"] for x in proposal.get("observations") or []}
        for c in a.get("constants") or []:
            v = float(c["value"])
            span = max(1.0, abs(v))
            base = sanitize_name(c["name"], "param")
            name, k = base, 2
            while name in taken:
                name, k = f"{base}_{k}", k + 1
            taken.add(name)
            self.t_actions.add_row({"name": name, "source": c["path"], "kind": "Constante (Value)", "default": v,
                                    "low": v - span, "high": v + span},
                                   meta={"kind": "param", "path": c["path"], "param": "Value"}, checked=False)
        # observations : sorties observables
        self.t_obs.setRowCount(0)
        self.t_obs.meta = []
        outs = {o["path"]: o for o in a.get("outports") or []}
        for o in proposal.get("observations") or []:
            src = outs.get(o["path"], {}).get("source", o["path"])
            self.t_obs.add_row({"name": o["name"], "source": f"sortie {Path(o['path']).name}  (bloc {src})",
                                "port": float(o.get("port", 1))}, meta={"path": o["path"]})
        # variables initiales : scalaires numeriques
        self.t_init.setRowCount(0)
        self.t_init.meta = []
        for v in a.get("variables") or []:
            if v.get("scalar") and v.get("value") is not None:
                val = float(v["value"])
                span = max(0.1, abs(val) * 0.2)
                self.t_init.add_row({"variable": v["name"], "value": val, "low": val - span, "high": val + span},
                                    meta={"in_base": bool(v.get("in_base")), "in_model": bool(v.get("in_model"))}, checked=False)
        self.f_name.setText(a.get("model", "") + " (RL)")
        self.s_step.setValue(self._guess_step(sol))
        self.s_solver.setValue(self._guess_solver_step(sol))
        self.c_solver.setCurrentIndex(1 if sol.get("type") == "Fixed-step" else 0)

    @staticmethod
    def _guess_step(sol):
        try:
            return max(float(sol["step"]) * 10, 0.01) if sol.get("type") == "Fixed-step" else 1.0
        except (KeyError, TypeError, ValueError):
            return 1.0

    @staticmethod
    def _guess_solver_step(sol):
        try:
            return float(sol["step"]) if sol.get("type") == "Fixed-step" else 0.01
        except (KeyError, TypeError, ValueError):
            return 0.01

    # ------------------------------------------------------------------ explorateur de blocs
    def _explore(self):
        model = self.f_model.text().strip()
        if not model or not Path(model).is_file():
            return
        self._busy(True, "Lecture des blocs du modèle...")
        self._job = WorkerJob(self.settings, ["inspect", "--model", model], self)
        self._job.done.connect(self._explored)
        self._job.start()

    def _explored(self, res, err):
        self._job = None
        self._busy(False)
        self._go(self.stack.currentIndex())
        if res is None:
            QMessageBox.critical(self, "MATLAB", f"Impossible d'inspecter le modèle :\n{err}")
            return
        dlg = SimulinkExplorer(self, res["blocks"])
        dlg.on_observation = self._add_obs_row
        dlg.on_action = self._add_action_row
        dlg.exec()

    def _add_obs_row(self, o):
        self.t_obs.add_row({"name": sanitize_name(o["name"], "signal"), "source": f"bloc {o['block']}", "port": float(o["port"])},
                           meta={"path": o["block"]})

    def _add_action_row(self, a):
        v = float(a["high"]) / 2 if a["high"] else 0.0
        self.t_actions.add_row({"name": sanitize_name(a["name"], "param"), "source": f"{a['block']} / {a['param']}",
                                "kind": "Paramètre de bloc", "default": v, "low": float(a["low"]), "high": float(a["high"])},
                               meta={"kind": "param", "path": a["block"], "param": a["param"]})
        self.stack.setCurrentIndex(1)

    # ------------------------------------------------------------------ lecture des tables -> specification
    def _actions_spec(self):
        out = []
        for vals, meta in self.t_actions.rows():
            out.append({"name": vals["name"], "kind": meta.get("kind", "inport"), "path": meta.get("path", ""),
                        "param": meta.get("param", "Value"), "default": vals["default"], "low": vals["low"], "high": vals["high"]})
        return out

    def _obs_spec(self):
        return [{"name": vals["name"], "path": meta["path"], "port": int(round(vals["port"]))}
                for vals, meta in self.t_obs.rows()]

    def _initial_spec(self):
        return [{"variable": vals["variable"], "low": vals["low"], "high": vals["high"]} for vals, _ in self.t_init.rows()]

    def _embedded_variables(self):
        """Variables du workspace de base utilisees par le modele : copiees dans la copie convertie (modele autonome)."""
        names = {v["name"] for v in (self.analysis or {}).get("variables") or [] if v.get("in_base") and not v.get("in_model")}
        names |= {i["variable"] for i in self._initial_spec()}
        return sorted(names)

    def _prepare_settings(self):
        names = [a["name"] for a in self._actions_spec()] + [o["name"] for o in self._obs_spec()]
        self.names_hint.setText("Variables utilisables : " + (", ".join(names) or "(aucune)")
                                + ", t (temps), dt, step. Fonctions : abs, min, max, exp, sqrt, sin, cos, tanh, clip, where.")
        if not self.f_reward.text().strip():
            obs = [o["name"] for o in self._obs_spec()]
            if obs:
                self.f_reward.setText(f"-({obs[0]} - 1.0)**2")
        self.check_lbl.setText("")

    def _spec(self):
        return {
            "project_name": self.f_name.text().strip(), "model_path": self.f_model.text().strip(),
            "init_path": self.f_init.text().strip(), "actions": self._actions_spec(), "observations": self._obs_spec(),
            "initial": self._initial_spec(), "variables": self._embedded_variables(),
            "remove": [self.removable.item(i).data(Qt.UserRole) for i in range(self.removable.count())
                       if self.removable.item(i).checkState() == Qt.Checked],
            "step_s": self.s_step.value(), "episode_s": self.s_episode.value(), "solver": self.c_solver.currentData(),
            "solver_step": self.s_solver.value(), "reward": self.f_reward.text().strip(), "terminate": self.f_term.text().strip(),
        }

    # ------------------------------------------------------------------ conversion
    def _convert(self):
        try:
            spec = self._spec()
        except ValueError as e:
            QMessageBox.warning(self, "Convertir un modèle", str(e))
            return
        errs = validate_conversion(spec)
        if errs:
            self.check_lbl.setText("<br>".join("&bull;  " + e for e in errs))
            self.check_lbl.setProperty("tone", "bad")
            self.check_lbl.style().unpolish(self.check_lbl)
            self.check_lbl.style().polish(self.check_lbl)
            return
        self.check_lbl.setText("")
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(spec, tmp, ensure_ascii=False)
        tmp.close()
        self._spec_file = tmp.name
        out_dir = self.store.root / self.store.free_slug(spec["project_name"])
        self._busy(True, "Conversion et test à vide dans MATLAB (une à deux minutes)...")
        self._job = WorkerJob(self.settings, ["convert-model", "--spec", tmp.name, "--out", str(out_dir)], self)
        self._job.done.connect(self._converted)
        self._job.start()

    def _converted(self, res, err):
        self._job = None
        self._busy(False)
        if self._spec_file:
            Path(self._spec_file).unlink(missing_ok=True)
            self._spec_file = None
        self._go(self.stack.currentIndex())
        if res is None:
            QMessageBox.critical(self, "Conversion impossible", err or "Erreur inconnue.")
            return
        self.created_slug = res["slug"]
        check = res.get("check") or {}
        lines = []
        if res.get("warnings"):
            lines += ["Avertissements :"] + [f"  - {w}" for w in res["warnings"]] + [""]
        if check.get("ok"):
            lines += [f"Projet créé et testé : {check['episode_steps']} pas par épisode, "
                      f"environ {check['ms_per_step']:.0f} ms de simulation par pas.",
                      "", "Observations après le test à vide :"] + [f"  {k} = {v}" for k, v in check["obs"].items()]
            QMessageBox.information(self, "Conversion terminée", "\n".join(lines))
        else:
            lines += ["Le projet est créé, mais le test à vide a échoué :", check.get("error", "erreur inconnue"), "",
                      "Corrige la configuration dans l'onglet Environnement du projet."]
            QMessageBox.warning(self, "Conversion terminée avec une erreur", "\n".join(lines))
        self.accept()

    def reject(self):
        if self._job is not None:  # un calcul MATLAB est en cours : on ne ferme pas a moitie
            return
        super().reject()
