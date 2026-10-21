"""Onglet « Modele mathematique » : extraire d'un run une loi de commande lisible (PID, table, arbre, equation)."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..distill import METHOD_LABELS, METHODS
from ..distill.export import distill_dir, load_report
from .runner import DistillRunner, RuntimeMissing, WorkerJob
from .theme import C
from .widgets import Card, button, fmt_run_date, label

COLUMNS = ["Modèle", "Complexité", "R2 (états de l'expert)", "Retour", "Score", "Menés à terme", ""]


class DistillTab(QWidget):
    busy = Signal(bool)  # un calcul est en cours : le parent bloque le changement de projet

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project = None
        self.report = None
        self.runner = None
        self.job = None
        self._build()

    # -- construction ----------------------------------------------------------------------
    def _build(self):
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._load_report)
        self.method_boxes = {}
        meth = QHBoxLayout()
        for m in METHODS:
            cb = QCheckBox(METHOD_LABELS[m])
            cb.setChecked(True)
            self.method_boxes[m] = cb
            meth.addWidget(cb)
        meth.addStretch()

        self.f_episodes = QSpinBox(); self.f_episodes.setRange(2, 500); self.f_episodes.setValue(20)
        self.f_noise = QDoubleSpinBox(); self.f_noise.setRange(0, 100); self.f_noise.setValue(10); self.f_noise.setSuffix(" %")
        self.f_dagger = QSpinBox(); self.f_dagger.setRange(0, 10); self.f_dagger.setValue(3)
        self.f_dagger_eps = QSpinBox(); self.f_dagger_eps.setRange(1, 200); self.f_dagger_eps.setValue(10)
        self.f_eval = QSpinBox(); self.f_eval.setRange(1, 200); self.f_eval.setValue(10)
        self.f_margin = QDoubleSpinBox(); self.f_margin.setRange(0, 50); self.f_margin.setValue(5); self.f_margin.setSuffix(" %")
        self.f_dims = QSpinBox(); self.f_dims.setRange(1, 2); self.f_dims.setValue(2)
        self.f_integ = QLineEdit(); self.f_integ.setPlaceholderText("ex : angle, position (action intégrale d'un PID)")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        fields = [("Épisodes de collecte", self.f_episodes), ("Bruit d'action", self.f_noise), ("Tours de DAgger", self.f_dagger),
                  ("Épisodes / tour", self.f_dagger_eps), ("Épisodes d'évaluation", self.f_eval), ("Tolérance", self.f_margin),
                  ("Dimensions de la table", self.f_dims)]
        for i, (txt, w) in enumerate(fields):
            grid.addWidget(label(txt, "sectionLabel"), (i // 4) * 2, i % 4)
            grid.addWidget(w, (i // 4) * 2 + 1, i % 4)
        grid.addWidget(label("Observations intégrées (PID)", "sectionLabel"), 2, 3)
        grid.addWidget(self.f_integ, 3, 3)

        self.btn_start = button("Extraire le modèle", "primary", self._start)
        self.btn_stop = button("Arrêter", None, self._stop)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p %")
        self.status = label("", "muted", wrap=True)
        self.stage = label("", wrap=True)
        self.timing = label("", "muted")
        self.logbox = QPlainTextEdit()
        self.logbox.setObjectName("mono")
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(5000)  # journal borne : il ne ralentit pas l'affichage
        self.logbox.setFixedHeight(120)
        self.logbox.setPlaceholderText("Journal de l'extraction")
        self.logbox.hide()
        self._t0, self._fraction = None, 0.0
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)
        row = QHBoxLayout()
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addWidget(self.progress, 1)

        cfg = Card("Extraction", "Rejoue la politique apprise, ajuste des modèles simples sur ses décisions (avec DAgger pour "
                   "éviter la dérive), puis les compare en boucle fermée. Le modèle le plus simple qui reste dans la tolérance est recommandé.")
        top = QHBoxLayout()
        top.addWidget(label("Run", "sectionLabel"))
        top.addWidget(self.run_combo, 1)
        cfg.body.addLayout(top)
        cfg.body.addWidget(label("Familles de modèles", "sectionLabel"))
        cfg.body.addLayout(meth)
        cfg.body.addLayout(grid)
        cfg.body.addLayout(row)
        cfg.body.addWidget(self.stage)
        cfg.body.addWidget(self.timing)
        cfg.body.addWidget(self.logbox)
        cfg.body.addWidget(self.status)

        # resultats
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_candidate)
        self.table.setMinimumHeight(180)

        self.pareto = self._plot("Simplicité / performance", "Complexité (paramètres, échelle log)", "Score (0 = aléatoire, 1 = politique)")
        self.scatter = self._plot("Fidélité des actions", "Action de la politique", "Action du modèle")
        plots = QHBoxLayout()
        plots.addWidget(self.pareto)
        plots.addWidget(self.scatter)

        self.equations = QPlainTextEdit()
        self.equations.setObjectName("mono")
        self.equations.setReadOnly(True)
        self.equations.setMinimumHeight(150)
        self.notes = label("", "hint", wrap=True)
        self.btn_copy_py = button("Copier Python", None, lambda: self._copy("policy.py"))
        self.btn_copy_m = button("Copier MATLAB", None, lambda: self._copy("policy.m"))
        self.btn_save = button("Enregistrer...", None, self._save_as)
        self.btn_slx = button("Exporter vers Simulink...", None, self._export_simulink)
        self.btn_open = button("Ouvrir le dossier", None, self._open_dir)
        actions = QHBoxLayout()
        for b in (self.btn_copy_py, self.btn_copy_m, self.btn_save, self.btn_slx, self.btn_open):
            actions.addWidget(b)
        actions.addStretch()

        res = Card("Résultats")
        self.summary = label("Aucune extraction pour ce run.", "muted", wrap=True)
        res.body.addWidget(self.summary)
        res.body.addWidget(self.table)
        res.body.addLayout(plots)
        res.body.addWidget(label("Équations / description du modèle sélectionné", "sectionLabel"))
        res.body.addWidget(self.equations)
        res.body.addLayout(actions)
        res.body.addWidget(self.notes)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(28, 20, 28, 20)
        lay.setSpacing(14)
        lay.addWidget(cfg)
        lay.addWidget(res, 1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        from .widgets import scroll_page  # noqa: PLC0415
        outer.addWidget(scroll_page(inner))
        self._set_result_buttons(False)
        self.restyle()

    def _plot(self, title, xl, yl):
        pw = pg.PlotWidget()
        pw.setObjectName("plot")
        pw._title, pw._xl, pw._yl = title, xl, yl
        pw.showGrid(x=True, y=True, alpha=0.12)
        pw.setMenuEnabled(False)
        pw.setMinimumHeight(240)
        return pw

    def restyle(self):
        if "surface" not in C:  # theme pas encore applique
            return
        for pw in (self.pareto, self.scatter):
            pw.setBackground(C["surface"])
            pw.setTitle(pw._title, color=C["text"], size="11pt")
            pw.setLabel("bottom", pw._xl, color=C["muted"])
            pw.setLabel("left", pw._yl, color=C["muted"])
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setPen(C["border2"])
                pw.getAxis(ax).setTextPen(C["muted"])
        self._draw_plots()

    # -- projet / runs ---------------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        runs = [d for d in (project.list_runs() if project else []) if (d / "model.zip").exists()]
        for d in runs:
            self.run_combo.addItem(fmt_run_date(d.name), str(d))
        self.run_combo.blockSignals(False)
        self.btn_start.setEnabled(bool(runs) and self.runner is None)
        if not runs:
            self._clear_results("Aucun run avec modèle : entraîne le projet d'abord.")
        else:
            self.run_combo.setCurrentIndex(0)
            self._load_report()

    def _run_dir(self):
        data = self.run_combo.currentData()
        return Path(data) if data else None

    def _clear_results(self, text):
        self.report = None
        self.table.setRowCount(0)
        self.equations.clear()
        self.notes.setText("")
        self.summary.setText(text)
        self._set_result_buttons(False)
        self._draw_plots()

    def _load_report(self, *_):
        d = self._run_dir()
        rep = load_report(d) if d else None
        if rep is None:
            self._clear_results("Aucune extraction pour ce run." if d else "Aucun run avec modèle.")
            return
        self.report = rep
        self._fill_results()

    # -- affichage -------------------------------------------------------------------------
    def _fill_results(self):
        rep = self.report
        ex = rep["expert"]
        self.summary.setText(
            f"Politique apprise ({rep['algo']}) : retour moyen <b>{ex['return_mean']:.2f}</b> ± {ex['return_std']:.2f}, "
            f"politique aléatoire : {rep['random_return']:.2f}. Extraction du {rep['created']} ({rep['duration_s']:.0f} s).")
        self.summary.setTextFormat(Qt.RichText)
        cands = sorted(rep["candidates"], key=lambda c: c["complexity"])
        self.table.setRowCount(len(cands))
        for r, c in enumerate(cands):
            vals = [c["label"], str(c["complexity"]), f"{c['r2_expert']:.3f}", f"{c['return_mean']:.2f}",
                    f"{c['score']:.3f}", f"{c['completed_rate'] * 100:.0f} %",
                    "recommande" if c["id"] == rep["recommended"] else ""]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter) if k < 6 else None
                it.setData(Qt.UserRole, c["id"])
                self.table.setItem(r, k, it)
        self.notes.setText("\n".join("• " + n for n in rep["notes"]))
        self._set_result_buttons(True)
        rec_row = next((r for r in range(self.table.rowCount()) if self.table.item(r, 0).data(Qt.UserRole) == rep["recommended"]), 0)
        self.table.selectRow(rec_row)
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

    def _draw_plots(self):
        self.pareto.clear()
        self.scatter.clear()
        if not self.report:
            return
        if "accent" not in C:
            return
        rep, sel = self.report, self._candidate()
        xs = [np.log10(max(c["complexity"], 1)) for c in rep["candidates"]]
        ys = [c["score"] for c in rep["candidates"]]
        brushes = [pg.mkBrush(C["success"] if c["id"] == rep["recommended"] else C["accent"]) for c in rep["candidates"]]
        item = pg.ScatterPlotItem(xs, ys, size=[16 if sel and c["id"] == sel["id"] else 10 for c in rep["candidates"]],
                                  brush=brushes, pen=pg.mkPen(C["text"], width=1))
        self.pareto.addItem(item)
        thr = 1.0 - rep["options"]["margin"]
        self.pareto.addItem(pg.InfiniteLine(pos=thr, angle=0, pen=pg.mkPen(C["warning"], style=Qt.DashLine, width=1.5)))
        for c, x, y in zip(rep["candidates"], xs, ys):
            t = pg.TextItem(c["id"], color=C["muted"], anchor=(0, 1))
            t.setPos(x, y)
            self.pareto.addItem(t)
        if sel and sel.get("scatter"):
            pts = np.asarray(sel["scatter"])
            lo, hi = float(pts.min()), float(pts.max())
            self.scatter.addItem(pg.PlotDataItem([lo, hi], [lo, hi], pen=pg.mkPen(C["warning"], style=Qt.DashLine, width=1.5)))
            self.scatter.addItem(pg.ScatterPlotItem(pts[:, 0], pts[:, 1], size=6, brush=pg.mkBrush(C["accent"]), pen=None))

    def _set_result_buttons(self, on):
        for b in (self.btn_copy_py, self.btn_copy_m, self.btn_save, self.btn_slx, self.btn_open):
            b.setEnabled(on)

    # -- extraction ------------------------------------------------------------------------
    def options(self):
        return {"methods": [m for m, cb in self.method_boxes.items() if cb.isChecked()], "episodes": self.f_episodes.value(),
                "noise": self.f_noise.value() / 100.0, "dagger_iters": self.f_dagger.value(),
                "dagger_episodes": self.f_dagger_eps.value(), "eval_episodes": self.f_eval.value(),
                "margin": self.f_margin.value() / 100.0, "table_dims": self.f_dims.value(),
                "integrators": [s.strip() for s in self.f_integ.text().split(",") if s.strip()]}

    def _start(self):
        d = self._run_dir()
        opts = self.options()
        if not d or not self.project:
            return
        if not opts["methods"]:
            QMessageBox.information(self, "Modèle mathématique", "Coche au moins une famille de modèles.")
            return
        try:
            self.runner = DistillRunner(self.project, self.settings, d.name, opts, self)
            self.runner.log.connect(self._on_log)
            self.runner.progress.connect(self._on_progress)
            self.runner.finished.connect(self._on_finished)
            self.runner.start()
        except RuntimeMissing as e:
            self.runner = None
            QMessageBox.warning(self, "Composants manquants", str(e))
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.run_combo.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("")
        self.logbox.clear()
        self.logbox.show()
        self.stage.setText("Démarrage du processus de calcul (chargement de PyTorch, quelques secondes) ...")
        self._t0, self._fraction = time.monotonic(), 0.0
        self._tick()
        self.clock.start()
        self.busy.emit(True)

    def _stop(self):
        if self.runner:
            self.runner.stop()
            self.stage.setText("Arrêt...")

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
        text = f"Écoulé : {self._fmt(elapsed)}"
        if self._fraction > 0.03:
            text += f"   ·   reste environ {self._fmt(elapsed * (1 - self._fraction) / self._fraction)}"
        self.timing.setText(text)

    def _on_finished(self, message):
        self.runner = None
        self.clock.stop()
        elapsed = time.monotonic() - self._t0 if self._t0 else 0
        self._t0 = None
        self.btn_stop.setEnabled(False)
        self.run_combo.setEnabled(True)
        self.btn_start.setEnabled(self.run_combo.count() > 0)
        self.busy.emit(False)
        self.stage.setText("")
        self.timing.setText(f"Durée : {self._fmt(elapsed)}")
        if message.startswith("ERREUR"):
            self.logbox.appendPlainText(message)
            self.status.setText("Échec de l'extraction : voir le journal ci-dessus.")
        else:
            self.status.setText("Extraction terminée." if message.startswith("Terminé") else "Extraction arrêtée.")
            self.progress.setValue(1000 if message.startswith("Terminé") else self.progress.value())
            self._load_report()

    # -- exports ---------------------------------------------------------------------------
    def _file(self, name):
        c = self._candidate()
        d = self._run_dir()
        return (distill_dir(d) / c["id"] / name) if c and d else None

    def _copy(self, name):
        f = self._file(name)
        if f and f.exists():
            QApplication.clipboard().setText(f.read_text(encoding="utf-8"))
            self.status.setText(f"{name} copie dans le presse-papiers.")

    def _save_as(self):
        c = self._candidate()
        if not c:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Enregistrer le modèle", f"{c['id']}.py",
                                                "Python (*.py);;MATLAB (*.m);;Équations (*.txt);;JSON (*.json)")
        if not target:
            return
        src = {".py": "policy.py", ".m": "policy.m", ".txt": "equations.txt", ".json": "surrogate.json"}.get(Path(target).suffix.lower())
        f = self._file(src) if src else None
        if f is None or not f.exists():
            QMessageBox.warning(self, "Enregistrer", "Extension non prise en charge (.py, .m, .txt ou .json).")
            return
        text = f.read_text(encoding="utf-8")
        if src == "policy.m":  # MATLAB exige que le nom de fonction == nom de fichier
            text = text.replace("policy", Path(target).stem)
        Path(target).write_text(text, encoding="utf-8")
        self.status.setText(f"Enregistré : {target}")

    def _export_simulink(self):
        c = self._candidate()
        d = self._run_dir()
        if not c or not d or not self.project:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Contrôleur Simulink", f"controleur_{c['id'].replace('-', '_')}.slx",
                                                "Modèle Simulink (*.slx)")
        if not target:
            return
        args = ["export-simulink", "--project", str(self.project.path), "--run", d.name, "--candidate", c["id"], "--out", target]
        self.job = WorkerJob(self.settings, args, self)
        self.job.done.connect(lambda res, err: self.status.setText(f"Modèle Simulink créé : {res['path']}" if res else f"ERREUR : {err}"))
        self.btn_slx.setEnabled(False)
        self.job.done.connect(lambda *_: self.btn_slx.setEnabled(True))
        self.status.setText("Génération du modèle Simulink (démarrage de MATLAB) ...")
        self.job.start()

    def _open_dir(self):
        d = self._run_dir()
        if d and sys.platform == "win32":
            target = distill_dir(d)
            if target.exists():
                os.startfile(str(target))
