"""Fenetre Parametres : reglages generaux de l'application (rien de tout ceci n'est dans la fenetre principale)."""
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QStackedWidget, QVBoxLayout,
    QWidget,
)

from .. import components, hardware, runtime
from ..paths import LICENSE_FILE, LICENSES_DIR, LOGS_DIR, NOTICES_FILE, USER_DIR
from ..settings import Settings
from . import theme
from .runner import ToolProcess, WorkerJob
from .widgets import Card, button, label, scroll_page, column

class ThemeEditor(QDialog):
    """Edition d'un theme personnalise avec apercu en direct sur toute l'application."""

    FONTS = ["Segoe UI", "Consolas", "Cascadia Mono", "Arial", "Verdana", "Georgia", "Trebuchet MS"]

    def __init__(self, parent, base, name, reserved, existing_custom, on_preview):
        super().__init__(parent)
        self.setWindowTitle("Personnaliser le theme")
        self.setMinimumWidth(440)
        self.reserved, self.existing_custom, self.on_preview = reserved, existing_custom, on_preview
        self.vals = {k: base[k] for k in theme.BASE_KEYS}
        self.deleted = False
        self.result_name = self.result_theme = None

        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Mon theme")
        self.font_box = QComboBox()
        self.font_box.setEditable(True)
        self.font_box.addItems(self.FONTS)
        self.font_box.setCurrentText(base.get("font", "Segoe UI"))
        self.font_box.currentTextChanged.connect(lambda _: self._changed())

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        form.addRow("Nom du theme", self.name_edit)
        self.swatches = {}
        for key in theme.BASE_KEYS:
            b = QPushButton()
            b.setObjectName("swatch")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self._pick(k))
            self.swatches[key] = b
            self._paint(key)
            form.addRow(theme.BASE_LABELS[key], b)
        form.addRow("Police", self.font_box)

        self.msg = label("", "hint", wrap=True)
        btns = QHBoxLayout()
        self.btn_delete = button("Supprimer ce theme", "danger", self._delete)
        self.btn_delete.setVisible(name in existing_custom)
        btns.addWidget(self.btn_delete)
        btns.addStretch()
        btns.addWidget(button("Annuler", "ghost", self.reject))
        btns.addWidget(button("Enregistrer", "primary", self._accept))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)
        lay.addWidget(label("Personnaliser le theme", "cardTitle"))
        lay.addWidget(label("Les changements s'appliquent en direct. Les nuances (bordures, texte attenue...) "
                            "sont deduites des couleurs choisies.", "hint", wrap=True))
        lay.addLayout(form)
        lay.addWidget(self.msg)
        lay.addLayout(btns)

    def _paint(self, key):
        c = self.vals[key]
        fg = "#000000" if theme.luminance(c) > 0.55 else "#ffffff"
        self.swatches[key].setText(c.upper())
        self.swatches[key].setStyleSheet(f"background: {c}; color: {fg};")

    def _pick(self, key):
        col = QColorDialog.getColor(QColor(self.vals[key]), self, theme.BASE_LABELS[key])
        if col.isValid():
            self.vals[key] = col.name()
            self._paint(key)
            self._changed()

    def theme_dict(self):
        return {**self.vals, "font": self.font_box.currentText().strip() or "Segoe UI"}

    def _changed(self):
        self.on_preview(self.theme_dict())

    def _accept(self):
        name = self.name_edit.text().strip()
        if not name:
            self.msg.setText("Donne un nom au theme.")
            return
        if name in self.reserved:
            self.msg.setText(f"« {name} » est un theme integre : choisis un autre nom.")
            return
        self.result_name, self.result_theme = name, self.theme_dict()
        self.accept()

    def _delete(self):
        self.deleted = True
        self.accept()



class _Scanner(QObject):
    done = Signal(object)

    def start(self):
        threading.Thread(target=lambda: self.done.emit(hardware.scan()), daemon=True).start()


class LogDialog(QDialog):
    """Execute un outil console et affiche sa sortie en direct."""

    def __init__(self, parent, title, args):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(780, 520)
        self.code = None
        self.status = label("Execution en cours...", "muted")
        self.out = QPlainTextEdit()
        self.out.setObjectName("mono")
        self.out.setReadOnly(True)
        self.btn_close = button("Fermer", "primary", self.accept)
        self.btn_close.setEnabled(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)
        lay.addWidget(label(title, "cardTitle"))
        lay.addWidget(self.status)
        lay.addWidget(self.out, 1)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self.proc = ToolProcess(args, self)
        self.proc.line.connect(self.out.appendPlainText)
        self.proc.finished.connect(self._finished)

    def start(self):
        self.proc.start()

    def _finished(self, code):
        self.code = code
        self.btn_close.setEnabled(True)
        self.status.setText("Termine." if code == 0 else f"Termine avec des avertissements ou des erreurs (code {code}).")

    def reject(self):
        if self.code is None:
            return  # pas de fermeture pendant l'installation
        super().reject()


PAGES = ["Apparence", "General", "MATLAB / Simulink", "Calcul (CPU / GPU)", "Composants", "A propos"]


class SettingsDialog(QDialog):
    def __init__(self, parent, settings: Settings, apply_theme, themes, theme_name):
        super().__init__(parent)
        self.setWindowTitle("Parametres")
        self.resize(940, 640)
        self.settings = settings
        self.apply_theme, self.all_themes = apply_theme, themes  # callbacks de la fenetre principale
        self.initial_theme = theme_name
        self.hw = dict(settings.get("hardware") or {})
        self.env_info = {}
        self.projects_dir_changed = False

        self.nav = QListWidget()
        self.nav.setFixedWidth(210)
        self.nav.addItems(PAGES)
        self.nav.setStyleSheet("QListWidget::item { padding: 9px 12px; }")
        self.stack = QStackedWidget()
        for build in (self._page_appearance, self._page_general, self._page_matlab, self._page_compute,
                      self._page_components, self._page_about):
            self.stack.addWidget(build())
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.btn_ok = button("OK", "primary", self._ok)
        btns = QHBoxLayout()
        btns.addStretch()
        btns.addWidget(button("Annuler", "ghost", self.reject))
        btns.addWidget(self.btn_ok)
        body = QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 14)
        lay.addLayout(body, 1)
        btns.setContentsMargins(20, 0, 20, 0)
        lay.addLayout(btns)

        self._refresh_hw_labels()
        self._refresh_env_info()

    # -- pages --------------------------------------------------------------------------
    def _page_appearance(self):
        self.theme_combo = QComboBox()
        self._fill_themes()
        self.theme_combo.currentTextChanged.connect(lambda n: n and self.apply_theme(n))
        card = Card("Theme", "Le theme s'applique immediatement ; Annuler revient au theme precedent.")
        row = QHBoxLayout()
        row.addWidget(self.theme_combo, 1)
        row.addWidget(button("Personnaliser...", None, self._edit_theme))
        card.body.addLayout(row)
        return scroll_page(column(card), 720)

    def _fill_themes(self):
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        self.theme_combo.addItems(list(self.all_themes()))
        self.theme_combo.setCurrentText(self.parent().theme_name)
        self.theme_combo.blockSignals(False)

    def _edit_theme(self):
        win = self.parent()
        custom = dict(self.settings.get("custom_themes") or {})
        previous = win.theme_name
        base = self.all_themes().get(previous, theme.BUILTIN[theme.DEFAULT_THEME])
        dlg = ThemeEditor(self, base, previous if previous in custom else "", set(theme.BUILTIN), custom,
                          lambda t: self.apply_theme(None, base=t))
        accepted = dlg.exec() == QDialog.Accepted
        if accepted and dlg.deleted:
            custom.pop(previous, None)
            self.settings.set("custom_themes", custom)
            self.apply_theme(theme.DEFAULT_THEME, save=True)
        elif accepted and dlg.result_name:
            custom[dlg.result_name] = dlg.result_theme
            self.settings.set("custom_themes", custom)
            self.apply_theme(dlg.result_name, save=True)
        else:
            self.apply_theme(previous)
        self._fill_themes()

    def _page_general(self):
        self.projects_edit = QLineEdit(str(self.settings.projects_dir))
        card = Card("Dossier des projets", "Chaque projet est un dossier contenant sa configuration, son environnement et ses resultats.")
        row = QHBoxLayout()
        row.addWidget(self.projects_edit, 1)
        row.addWidget(button("Parcourir...", None, self._browse_projects))
        row.addWidget(button("Ouvrir", "ghost", lambda: self._open(self.projects_edit.text())))
        card.body.addLayout(row)
        data = Card("Donnees de l'application", "Reglages, composants de calcul et journaux (propres a chaque utilisateur).")
        data.body.addWidget(label(str(USER_DIR), "muted"))
        data.body.addLayout(self._row(button("Ouvrir le dossier", None, lambda: self._open(USER_DIR))))
        return scroll_page(column(card, data), 720)

    def _page_matlab(self):
        self.ml_enabled = QCheckBox("Activer l'integration MATLAB / Simulink")
        self.ml_enabled.setChecked(bool(self.settings.get("matlab_enabled")))
        self.ml_combo = QComboBox()
        self.ml_combo.currentIndexChanged.connect(self._on_matlab_choice)
        self.ml_path = QLineEdit(self.settings.get("matlab_root") or "")
        self.ml_path.setPlaceholderText("Detection automatique (version la plus recente)")
        self.ml_info = label("", "hint", wrap=True)
        self.ml_test = label("", "statusbox", wrap=True)
        self.ml_test.hide()
        install = Card("Installation de MATLAB", "Choisis l'installation a utiliser ; les fichiers du Engine Python doivent etre presents.")
        install.body.addWidget(self.ml_enabled)
        install.body.addWidget(self.ml_combo)
        row = QHBoxLayout()
        row.addWidget(self.ml_path, 1)
        row.addWidget(button("Parcourir...", None, self._browse_matlab))
        install.body.addLayout(row)
        install.body.addWidget(self.ml_info)
        install.body.addLayout(self._row(button("Detecter a nouveau", None, self._rescan),
                                         button("Tester MATLAB", "primary", self._test_matlab)))
        install.body.addWidget(self.ml_test)
        self.ml_engine = label("", "hint", wrap=True)
        engine = Card("MATLAB Engine pour Python", "Installe automatiquement dans les composants de calcul, avec la version qui correspond a ton MATLAB.")
        engine.body.addWidget(self.ml_engine)
        self._fill_matlab_combo()
        return scroll_page(column(install, engine), 720)

    def _page_compute(self):
        self.gpu_enabled = QCheckBox("Autoriser l'entrainement sur GPU (CUDA)")
        self.gpu_enabled.setChecked(bool(self.settings.get("gpu_enabled")))
        self.gpu_info = label("", wrap=True)
        self.torch_info = label("", "hint", wrap=True)
        gpu = Card("Acceleration GPU", "Si desactive, tous les entrainements tournent sur CPU. Pour un projet, 'auto' choisit le GPU quand il est disponible.")
        gpu.body.addWidget(self.gpu_enabled)
        gpu.body.addWidget(self.gpu_info)
        gpu.body.addWidget(self.torch_info)
        gpu.body.addLayout(self._row(button("Scanner le materiel", None, self._rescan)))
        return scroll_page(column(gpu), 720)

    def _page_components(self):
        self.rt_status = label("", wrap=True)
        self.rt_versions = label("", "hint", wrap=True)
        self.rt_manual = QLineEdit(self.settings.get("python_runtime") or "")
        self.rt_manual.setPlaceholderText("Automatique (composants installes par l'application)")
        card = Card("Composants de calcul", "Python, PyTorch et MATLAB Engine : ce qui execute les entrainements. L'installation choisit les versions adaptees a ton materiel.")
        card.body.addWidget(self.rt_status)
        card.body.addWidget(self.rt_versions)
        card.body.addLayout(self._row(button("Installer / reparer les composants...", "primary", self._provision),
                                      button("Verifier", None, self._refresh_env_info)))
        manual = Card("Python de calcul personnalise (avance)", "Remplace le runtime installe par un autre interpreteur Python (ex. un venv de developpement).")
        row = QHBoxLayout()
        row.addWidget(self.rt_manual, 1)
        row.addWidget(button("Parcourir...", None, self._browse_python))
        manual.body.addLayout(row)
        logs = Card("Journaux")
        logs.body.addLayout(self._row(button("Ouvrir le dossier des journaux", None, lambda: self._open(LOGS_DIR))))
        return scroll_page(column(card, manual, logs), 720)

    def _page_about(self):
        from .. import __version__

        card = Card("RL Studio", f"Version {__version__}")
        for name, value in [("Reglages", self.settings.path), ("Projets", self.settings.projects_dir),
                            ("Composants", USER_DIR / "runtime"), ("Journaux", LOGS_DIR)]:
            lb = label(f"<b>{name}</b> : {value}", wrap=True)
            lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
            card.body.addWidget(lb)
        legal = Card("Licences", "Logiciel libre. Les composants tiers gardent leur propre licence.")
        legal.body.addWidget(label("<b>RL Studio</b> - licence MIT, \u00a9 2026 Christophe Barras", wrap=True))
        qt = label(
            "L'interface utilise <b>Qt 6</b> (via PySide6) sous licence <b>GNU LGPL v3</b>, en liaison dynamique : "
            "les bibliotheques Qt sont des fichiers separes que vous pouvez remplacer (dossier <i>_internal\\PySide6</i>). "
            "Le code source de Qt et PySide6 est disponible aupres de The Qt Company.", wrap=True)
        qt.setTextFormat(Qt.RichText)
        legal.body.addWidget(qt)
        legal.body.addLayout(self._row(
            button("Licence de RL Studio", None, lambda: self._open(LICENSE_FILE)),
            button("Composants tiers", None, lambda: self._open(NOTICES_FILE)),
            button("Textes des licences (LGPL, GPL)", None, lambda: self._open(LICENSES_DIR)),
            button("Sources de Qt", None, lambda: QDesktopServices.openUrl(
                QUrl("https://download.qt.io/archive/qt/6.8/6.8.3/single/")))))
        return scroll_page(column(card, legal), 720)

    @staticmethod
    def _row(*widgets):
        row = QHBoxLayout()
        for w in widgets:
            row.addWidget(w)
        row.addStretch()
        return row

    @staticmethod
    def _open(path):
        if sys.platform == "win32" and Path(path).exists():
            os.startfile(path)

    # -- actions ------------------------------------------------------------------------
    def _browse_projects(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier des projets", self.projects_edit.text())
        if d:
            self.projects_edit.setText(d)

    def _browse_python(self):
        f, _ = QFileDialog.getOpenFileName(self, "Python de calcul", "", "Python (python.exe python)")
        if f:
            self.rt_manual.setText(f)

    def _browse_matlab(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier d'installation de MATLAB (ex. C:/Program Files/MATLAB/R2026b)",
                                             self.ml_path.text() or "C:/Program Files/MATLAB")
        if d:
            self.ml_path.setText(d)
            self._show_matlab_info(hardware.inspect_matlab(d))

    def _fill_matlab_combo(self):
        self.ml_combo.blockSignals(True)
        self.ml_combo.clear()
        self.ml_combo.addItem("Detection automatique", "")
        current = self.ml_path.text().strip()
        for m in self.hw.get("matlab") or []:
            flag = "" if m.get("complete") else "  (incomplet)"
            self.ml_combo.addItem(f"{m.get('release') or '?'}  -  {m['root']}{flag}", m["root"])
        idx = self.ml_combo.findData(current)
        self.ml_combo.setCurrentIndex(max(idx, 0))
        self.ml_combo.blockSignals(False)
        self._show_matlab_info(self._selected_matlab())

    def _selected_matlab(self):
        root = self.ml_path.text().strip()
        installs = self.hw.get("matlab") or []
        if root:
            return next((m for m in installs if Path(m["root"]) == Path(root)), None) or hardware.inspect_matlab(root)
        return next((m for m in installs if m.get("complete")), None)

    def _on_matlab_choice(self, _i):
        self.ml_path.setText(self.ml_combo.currentData() or "")
        self._show_matlab_info(self._selected_matlab())

    def _show_matlab_info(self, m):
        if not m:
            self.ml_info.setText("Aucune installation MATLAB valide trouvee a cet emplacement.")
            return
        py = ", ".join(m.get("engine_python_versions") or []) or "inconnu"
        ok = "complete" if m.get("complete") else "INCOMPLETE (matlab.exe ou Engine Python manquant)"
        self.ml_info.setText(f"{m.get('release') or '?'} - version {m.get('version')} - installation {ok}. "
                             f"Simulink : {'oui' if m.get('simulink') else 'non'}. Python supporte par l'Engine : {py}.")

    def _rescan(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._scanner = _Scanner()
        self._scanner.done.connect(self._scan_done)
        self._scanner.start()

    def _scan_done(self, hw):
        QApplication.restoreOverrideCursor()
        self.hw = hw
        self.settings.set("hardware", hw)
        self._fill_matlab_combo()
        self._refresh_hw_labels()

    def _refresh_hw_labels(self):
        gpus = [g for g in self.hw.get("gpus") or [] if g.get("vendor") == "NVIDIA"]
        if gpus:
            g = gpus[0]
            self.gpu_info.setText(f"<b>{g['name']}</b>  -  {g.get('memory_gb') or '?'} Go  -  pilote {g.get('driver')}  -  "
                                  f"CUDA max du pilote : {self.hw.get('cuda_driver') or 'inconnu'}")
        else:
            self.gpu_info.setText("Aucune carte NVIDIA detectee : les entrainements utiliseront le CPU." if self.hw
                                  else "Materiel non scanne : clique sur « Scanner le materiel ».")

    def _refresh_env_info(self):
        self.rt_status.setText(runtime.status(self.settings)["message"])
        job = WorkerJob(self.settings, ["env-info"], self)
        job.done.connect(self._env_info_done)
        self.rt_versions.setText("Verification des composants...")
        job.start()

    def _env_info_done(self, res, err):
        if res is None:
            self.rt_versions.setText("Verification impossible : " + err)
            self.torch_info.setText("")
            self.ml_engine.setText("Etat inconnu (composants de calcul indisponibles).")
            return
        self.env_info = res
        yes = lambda v: v or "non installe"  # noqa: E731
        self.rt_versions.setText(
            f"Python {res.get('python')}  -  PyTorch {yes(res.get('torch'))}  -  Stable-Baselines3 {yes(res.get('stable_baselines3'))}"
            f"  -  Gymnasium {yes(res.get('gymnasium'))}")
        if res.get("torch"):
            gpu = f"GPU utilisable : {res['gpu']} (CUDA {res.get('cuda_version')})" if res.get("cuda_available") \
                else "PyTorch est en version CPU (ou aucun GPU utilisable)."
            self.torch_info.setText(f"PyTorch {res['torch']}. {gpu}")
        else:
            self.torch_info.setText("PyTorch n'est pas installe.")
        self.ml_engine.setText(f"matlabengine {res['matlabengine']} installe." if res.get("matlabengine")
                               else "matlabengine n'est pas installe dans les composants de calcul.")

    def _test_matlab(self):
        self._apply_to_settings()
        self.ml_test.setProperty("tone", "")
        self.ml_test.setText("Demarrage de MATLAB (jusqu'a 30 s)...")
        self.ml_test.show()
        job = WorkerJob(self.settings, ["matlab-test"], self)
        job.done.connect(self._matlab_test_done)
        job.start()

    def _matlab_test_done(self, res, err):
        tone = "ok" if res else "bad"
        if res:
            text = (f"MATLAB {res['version']} demarre en {res['startup_s']} s. Simulink : "
                    f"{'licence valide' if res['simulink'] else 'licence ABSENTE'}.")
            if not res["simulink"]:
                tone = "bad"
        else:
            text = "Echec : " + err
        self.ml_test.setProperty("tone", tone)
        self.ml_test.style().unpolish(self.ml_test)
        self.ml_test.style().polish(self.ml_test)
        self.ml_test.setText(text)

    def _provision(self):
        self._apply_to_settings()
        dlg = LogDialog(self, "Installation des composants de calcul", ["--provision"])
        dlg.start()
        dlg.exec()
        self.settings.reload()
        self.hw = dict(self.settings.get("hardware") or {})
        self.ml_path.setText(self.settings.get("matlab_root") or "")
        self._fill_matlab_combo()
        self._refresh_hw_labels()
        self._refresh_env_info()

    # -- validation ---------------------------------------------------------------------
    def _apply_to_settings(self):
        self.settings.update({
            "matlab_enabled": self.ml_enabled.isChecked(),
            "matlab_root": self.ml_path.text().strip(),
            "gpu_enabled": self.gpu_enabled.isChecked(),
            "python_runtime": self.rt_manual.text().strip(),
        })

    def _ok(self):
        old = str(self.settings.projects_dir)
        new = self.projects_edit.text().strip()
        if new and new != old:
            try:
                Path(new).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, "Dossier des projets", f"Dossier inutilisable :\n{e}")
                return
            self.settings.set("projects_dir", new)
            self.projects_dir_changed = True
        self._apply_to_settings()
        self.settings.set("theme", self.parent().theme_name)
        self.accept()

    def reject(self):
        self.apply_theme(self.initial_theme)  # annule l'apercu du theme
        super().reject()
