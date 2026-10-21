"""Onglet Résultats : liste des entraînements du projet."""
import json
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QPlainTextEdit, QWidget,
)

from .theme import C
from .widgets import Card, button, fmt_run_date, label, tips


class RunsTab:
    """Méthodes de MainWindow (onglet Résultats)."""

    def _tab_runs(self):
        self.runs_list = QListWidget()
        self.runs_list.setObjectName("runs")
        self.runs_list.currentItemChanged.connect(self._show_run)
        self.runs_empty = label("Aucun entraînement pour ce projet.\nLance-en un depuis l'onglet Entraînement.",
                                "muted", wrap=True)
        self.runs_empty.setAlignment(Qt.AlignCenter)
        left = Card("Entraînements")
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
        tips(self, {
            "runs_list": "Les entraînements du projet, du plus récent au plus ancien. Cliquez sur l'un d'eux pour voir ses détails.",
            "runs_empty": "Aucun entraînement n'a encore été lancé pour ce projet.",
            "run_title": "Date de l'entraînement sélectionné.",
            "run_info": "Modèle enregistré ou non, et récompense moyenne de la politique de référence.",
            "run_cfg": "Configuration exacte utilisée pour cet entraînement (copie du fichier project.yaml de l'époque).",
            "btn_run_dir": "Affiche dans l'Explorateur le dossier de cet entraînement (modèle, configuration, résultats d'extraction).",
        })
        detail = Card()
        detail.body.addWidget(self.run_title)
        detail.body.addWidget(self.run_info)
        detail.body.addWidget(label("Configuration utilisée", "sectionLabel"))
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
                    sub.append(f"référence {json.loads(base.read_text(encoding='utf-8'))['mean']:.2f}")
                except Exception:
                    pass
            sub.append("modèle ✓" if (d / "model.zip").exists() else "sans modèle")
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
        rows.append(f"Modèle : <b>{'enregistre' if has_model else 'absent (entraînement interrompu ou en erreur)'}</b>")
        base = os.path.join(d, "baseline.json")
        if os.path.exists(base):
            b = json.loads(open(base, encoding="utf-8").read())
            rows.append(f"Référence ({b.get('label', '?')}) : <b>{b['mean']:.3f}</b> ± {b['std']:.3f}")
        rows.append(f"<span style='color:{C['muted']}'>{d}</span>")
        self.run_info.setText("<br>".join(rows))
        cfg = os.path.join(d, "project.yaml")
        self.run_cfg.setPlainText(open(cfg, encoding="utf-8").read() if os.path.exists(cfg) else "")

    def _open_run_folder(self):
        it = self.runs_list.currentItem()
        if it and sys.platform == "win32":
            os.startfile(it.data(Qt.UserRole))
