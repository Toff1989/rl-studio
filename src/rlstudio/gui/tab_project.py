"""Onglet Projet (fiche, indicateurs)."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLineEdit, QPlainTextEdit, QVBoxLayout,
)

from .widgets import Card, Kpi, button, column, fmt_run_date, label, scroll_page


class ProjectTab:
    """Methodes de MainWindow (onglet projet)."""

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
