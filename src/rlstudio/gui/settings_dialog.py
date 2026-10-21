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
from .guide import GuideView
from .notify import looks_like_error, summarize
from .widgets import Card, button, column, label, scroll_page, tip, tips

class ThemeEditor(QDialog):
    """Édition d'un thème personnalisé avec aperçu en direct sur toute l'application."""

    FONTS = ["Segoe UI", "Consolas", "Cascadia Mono", "Arial", "Verdana", "Georgia", "Trebuchet MS"]

    def __init__(self, parent, base, name, reserved, existing_custom, on_preview):
        super().__init__(parent)
        self.setWindowTitle("Personnaliser le thème")
        self.setMinimumWidth(440)
        self.reserved, self.existing_custom, self.on_preview = reserved, existing_custom, on_preview
        self.vals = {k: base[k] for k in theme.BASE_KEYS}
        self.deleted = False
        self.result_name = self.result_theme = None

        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Mon thème")
        self.font_box = QComboBox()
        self.font_box.setEditable(True)
        self.font_box.addItems(self.FONTS)
        self.font_box.setCurrentText(base.get("font", "Segoe UI"))
        self.font_box.currentTextChanged.connect(lambda _: self._changed())

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        form.addRow("Nom du thème", self.name_edit)
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
        tips(self, {"name_edit": "Nom sous lequel le thème sera enregistré (différent de Sombre, Clair et Matrix).",
                    "font_box": "Police de toute l'application. Vous pouvez saisir le nom d'une police installée sur votre ordinateur."})
        for key, b in self.swatches.items():
            b.setToolTip(f"Cliquez pour choisir la couleur « {theme.BASE_LABELS[key]} » ; l'aperçu est immédiat sur toute la fenêtre.")

        self.msg = label("", "hint", wrap=True)
        btns = QHBoxLayout()
        self.btn_delete = button("Supprimer ce thème", "danger", self._delete, tip="Supprime ce thème personnalisé et revient au thème Sombre.")
        self.btn_delete.setVisible(name in existing_custom)
        btns.addWidget(self.btn_delete)
        btns.addStretch()
        btns.addWidget(button("Annuler", "ghost", self.reject, tip="Abandonne les modifications du thème."))
        btns.addWidget(button("Enregistrer", "primary", self._accept, tip="Enregistre ce thème et l'applique."))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)
        lay.addWidget(label("Personnaliser le thème", "cardTitle"))
        lay.addWidget(label("Les changements s'appliquent en direct. Les nuances (bordures, texte atténué...) "
                            "sont déduites des couleurs choisies.", "hint", wrap=True))
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
            self.msg.setText("Donne un nom au thème.")
            return
        if name in self.reserved:
            self.msg.setText(f"« {name} » est un thème intégré : choisis un autre nom.")
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
    """Exécute un outil console et affiche sa sortie en direct."""

    def __init__(self, parent, title, args):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(780, 520)
        self.code = None
        self.status = label("Exécution en cours...", "muted", wrap=True)
        self.out = QPlainTextEdit()
        self.out.setObjectName("mono")
        self.out.setReadOnly(True)
        self.btn_close = button("Fermer", "primary", self.accept, tip="Disponible une fois l'opération terminée (elle ne peut pas être interrompue).")
        self.btn_close.setEnabled(False)
        self.out.setToolTip("Sortie détaillée de l'opération en cours.")
        self._problems = []
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
        self.proc.line.connect(self._on_line)
        self.proc.finished.connect(self._finished)

    def start(self):
        self.proc.start()

    def _on_line(self, line):
        self.out.appendPlainText(line)
        if looks_like_error(line):
            self._problems.append(line)

    def _finished(self, code):
        self.code = code
        self.btn_close.setEnabled(True)
        if code == 0 and not self._problems:
            self.status.setText("Terminé.")
            return
        # une erreur écrite dans la sortie ne doit pas passer inaperçue : on la remonte au-dessus du journal
        problem = summarize(self._problems[-1]) if self._problems else f"code de sortie {code}"
        self.status.setText(f"Terminé avec des avertissements ou des erreurs (code {code}) : {problem}")
        self.status.setObjectName("statusbox")
        self.status.setProperty("tone", "bad")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def reject(self):
        if self.code is None:
            return  # pas de fermeture pendant l'installation
        super().reject()


PAGES = ["Apparence", "Général", "MATLAB / Simulink", "Calcul (CPU / GPU)", "Composants", "Guide d'utilisation", "À propos"]
PAGE_TIPS = [
    "Thème de couleurs de l'application.",
    "Dossier des projets et dossier des données de l'application.",
    "Choix de l'installation de MATLAB utilisée pour les projets Simulink.",
    "Utilisation de la carte graphique NVIDIA pour l'entraînement.",
    "Python, PyTorch et MATLAB Engine : ce qui exécute les calculs.",
    "Mode d'emploi de RL Studio, avec recherche.",
    "Version, emplacements des fichiers et licences.",
]


class SettingsDialog(QDialog):
    def __init__(self, parent, settings: Settings, apply_theme, themes, theme_name):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.resize(940, 640)
        self.settings = settings
        self.apply_theme, self.all_themes = apply_theme, themes  # callbacks de la fenetre principale
        self.initial_theme = theme_name
        self.hw = dict(settings.get("hardware") or {})
        self.env_info = {}
        self.projects_dir_changed = False

        self.nav = QListWidget()
        self.nav.setFixedWidth(210)
        self.nav.setToolTip("Catégories de paramètres.")
        self.nav.addItems(PAGES)
        for i, text in enumerate(PAGE_TIPS):
            self.nav.item(i).setToolTip(text)
        self.nav.setStyleSheet("QListWidget::item { padding: 9px 12px; }")
        self.stack = QStackedWidget()
        for build in (self._page_appearance, self._page_general, self._page_matlab, self._page_compute,
                      self._page_components, self._page_guide, self._page_about):
            self.stack.addWidget(build())
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.btn_ok = button("OK", "primary", self._ok, tip="Enregistre les paramètres et ferme la fenêtre.")
        btns = QHBoxLayout()
        btns.addStretch()
        btns.addWidget(button("Annuler", "ghost", self.reject, tip="Ferme la fenêtre sans enregistrer (le thème revient à celui d'avant)."))
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
        card = Card("Thème", "Le thème s'applique immédiatement ; Annuler revient au thème précédent.")
        row = QHBoxLayout()
        row.addWidget(tip(self.theme_combo, "Thème de couleurs : Sombre, Clair, Matrix, ou un thème que vous avez créé."), 1)
        row.addWidget(button("Personnaliser...", None, self._edit_theme,
                             tip="Crée ou modifie un thème : choix des couleurs et de la police, avec aperçu en direct."))
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
        card = Card("Dossier des projets", "Chaque projet est un dossier contenant sa configuration, son environnement et ses résultats.")
        row = QHBoxLayout()
        row.addWidget(tip(self.projects_edit, "Dossier qui contient un sous-dossier par projet. Le changer ne déplace pas les projets existants."), 1)
        row.addWidget(button("Parcourir...", None, self._browse_projects, tip="Choisit le dossier des projets."))
        row.addWidget(button("Ouvrir", "ghost", lambda: self._open(self.projects_edit.text()), tip="Affiche ce dossier dans l'Explorateur de fichiers."))
        card.body.addLayout(row)
        data = Card("Données de l'application", "Réglages, composants de calcul et journaux (propres à chaque utilisateur).")
        data.body.addWidget(label(str(USER_DIR), "muted"))
        data.body.addLayout(self._row(button("Ouvrir le dossier", None, lambda: self._open(USER_DIR),
                                             tip="Affiche le dossier des réglages, des composants de calcul et des journaux.")))
        return scroll_page(column(card, data), 720)

    def _page_matlab(self):
        self.ml_enabled = QCheckBox("Activer l'intégration MATLAB / Simulink")
        self.ml_enabled.setChecked(bool(self.settings.get("matlab_enabled")))
        self.ml_combo = QComboBox()
        self.ml_combo.currentIndexChanged.connect(self._on_matlab_choice)
        self.ml_path = QLineEdit(self.settings.get("matlab_root") or "")
        self.ml_path.setPlaceholderText("Détection automatique (version la plus récente)")
        self.ml_info = label("", "hint", wrap=True)
        self.ml_test = label("", "statusbox", wrap=True)
        self.ml_test.hide()
        install = Card("Installation de MATLAB", "Choisis l'installation à utiliser ; les fichiers du Engine Python doivent être présents.")
        tips(self, {
            "ml_enabled": "Décochez pour travailler sans MATLAB : les projets Simulink ne pourront alors pas être entraînés.",
            "ml_combo": "Installations de MATLAB trouvées sur cet ordinateur. « Détection automatique » prend la plus récente et complète.",
            "ml_path": "Dossier d'installation de MATLAB (ex. C:/Program Files/MATLAB/R2026b). Vide = détection automatique.",
            "ml_info": "Version détectée, présence de Simulink et versions de Python acceptées par MATLAB Engine.",
            "ml_test": "Résultat du test de démarrage de MATLAB.",
        })
        install.body.addWidget(self.ml_enabled)
        install.body.addWidget(self.ml_combo)
        row = QHBoxLayout()
        row.addWidget(self.ml_path, 1)
        row.addWidget(button("Parcourir...", None, self._browse_matlab, tip="Choisit manuellement le dossier d'installation de MATLAB."))
        install.body.addLayout(row)
        install.body.addWidget(self.ml_info)
        install.body.addLayout(self._row(button("Détecter à nouveau", None, self._rescan,
                                                tip="Relance la recherche des installations de MATLAB et de la carte graphique."),
                                         button("Tester MATLAB", "primary", self._test_matlab,
                                                tip="Démarre MATLAB pour vérifier qu'il fonctionne et que la licence Simulink est valide (jusqu'à 30 s).")))
        install.body.addWidget(self.ml_test)
        self.ml_engine = label("", "hint", wrap=True)
        self.ml_engine.setToolTip("Le MATLAB Engine est le pont qui permet à Python de piloter MATLAB et Simulink.")
        engine = Card("MATLAB Engine pour Python", "Installe automatiquement dans les composants de calcul, avec la version qui correspond à ton MATLAB.")
        engine.body.addWidget(self.ml_engine)
        self._fill_matlab_combo()
        return scroll_page(column(install, engine), 720)

    def _page_compute(self):
        self.gpu_enabled = QCheckBox("Autoriser l'entraînement sur GPU (CUDA)")
        self.gpu_enabled.setChecked(bool(self.settings.get("gpu_enabled")))
        self.gpu_info = label("", wrap=True)
        self.torch_info = label("", "hint", wrap=True)
        gpu = Card("Accélération GPU", "Si désactivé, tous les entraînements tournent sur CPU. Pour un projet, 'auto' choisit le GPU quand il est disponible.")
        tips(self, {
            "gpu_enabled": "Autorise l'entraînement sur la carte graphique NVIDIA pour les projets réglés sur « cuda » ou « auto ». Décoché : tout tourne sur le processeur.",
            "gpu_info": "Carte graphique NVIDIA détectée, pilote et version de CUDA prise en charge.",
            "torch_info": "Version de PyTorch installée et utilisabilité du GPU.",
        })
        gpu.body.addWidget(self.gpu_enabled)
        gpu.body.addWidget(self.gpu_info)
        gpu.body.addWidget(self.torch_info)
        gpu.body.addLayout(self._row(button("Scanner le matériel", None, self._rescan,
                                            tip="Détecte à nouveau la carte graphique, la mémoire et les installations de MATLAB.")))
        return scroll_page(column(gpu), 720)

    def _page_components(self):
        self.rt_status = label("", wrap=True)
        self.rt_versions = label("", "hint", wrap=True)
        self.rt_manual = QLineEdit(self.settings.get("python_runtime") or "")
        self.rt_manual.setPlaceholderText("Automatique (composants installés par l'application)")
        card = Card("Composants de calcul", "Python, PyTorch et MATLAB Engine : ce qui exécute les entraînements. L'installation choisit les versions adaptées à ton matériel.")
        tips(self, {
            "rt_status": "Les composants de calcul sont-ils prêts ? Sinon, utilisez « Installer / réparer ».",
            "rt_versions": "Versions de Python, PyTorch, Stable-Baselines3 et Gymnasium trouvées dans les composants de calcul.",
            "rt_manual": "Avancé : chemin d'un python.exe à utiliser à la place des composants installés par l'application (ex. un environnement de développement). Vide = automatique.",
        })
        card.body.addWidget(self.rt_status)
        card.body.addWidget(self.rt_versions)
        card.body.addLayout(self._row(button("Installer / réparer les composants...", "primary", self._provision,
                                             tip="Installe ou répare Python, PyTorch (version adaptée à votre carte graphique) et MATLAB Engine. Nécessite une connexion Internet."),
                                      button("Vérifier", None, self._refresh_env_info, tip="Relit les versions installées.")))
        manual = Card("Python de calcul personnalisé (avancé)", "Remplace le runtime installé par un autre interpréteur Python (ex. un venv de développement).")
        row = QHBoxLayout()
        row.addWidget(self.rt_manual, 1)
        row.addWidget(button("Parcourir...", None, self._browse_python, tip="Choisit un interpréteur Python (python.exe)."))
        manual.body.addLayout(row)
        logs = Card("Journaux")
        logs.body.addLayout(self._row(button("Ouvrir le dossier des journaux", None, lambda: self._open(LOGS_DIR),
                                             tip="Affiche les fichiers journaux détaillés (utiles pour comprendre un problème d'installation).")))
        return scroll_page(column(card, manual, logs), 720)

    def _page_guide(self):
        self.guide_view = GuideView()
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(20, 18, 20, 8)
        lay.addWidget(self.guide_view)
        return holder

    def _page_about(self):
        from .. import __version__

        card = Card("RL Studio", f"Version {__version__}")
        for name, value in [("Réglages", self.settings.path), ("Projets", self.settings.projects_dir),
                            ("Composants", USER_DIR / "runtime"), ("Journaux", LOGS_DIR)]:
            lb = label(f"<b>{name}</b> : {value}", wrap=True)
            lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
            card.body.addWidget(lb)
        legal = Card("Licences", "Logiciel libre. Les composants tiers gardent leur propre licence.")
        legal.body.addWidget(label("<b>RL Studio</b> - licence MIT, \u00a9 2026 Christophe Barras", wrap=True))
        qt = label(
            "L'interface utilise <b>Qt 6</b> (via PySide6) sous licence <b>GNU LGPL v3</b>, en liaison dynamique : "
            "les bibliothèques Qt sont des fichiers séparés que vous pouvez remplacer (dossier <i>_internal\\PySide6</i>). "
            "Le code source de Qt et PySide6 est disponible auprès de The Qt Company.", wrap=True)
        qt.setTextFormat(Qt.RichText)
        legal.body.addWidget(qt)
        legal.body.addLayout(self._row(
            button("Licence de RL Studio", None, lambda: self._open(LICENSE_FILE), tip="Ouvre le texte de la licence MIT de RL Studio."),
            button("Composants tiers", None, lambda: self._open(NOTICES_FILE), tip="Ouvre la liste des composants tiers et de leurs licences."),
            button("Textes des licences (LGPL, GPL)", None, lambda: self._open(LICENSES_DIR), tip="Ouvre le dossier des textes complets des licences."),
            button("Sources de Qt", None, lambda: QDesktopServices.openUrl(
                QUrl("https://download.qt.io/archive/qt/6.8/6.8.3/single/")), tip="Ouvre dans le navigateur la page de téléchargement du code source de Qt.")))
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
        self.ml_combo.addItem("Détection automatique", "")
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
            self.ml_info.setText("Aucune installation MATLAB valide trouvée à cet emplacement.")
            return
        py = ", ".join(m.get("engine_python_versions") or []) or "inconnu"
        ok = "complète" if m.get("complete") else "INCOMPLÈTE (matlab.exe ou Engine Python manquant)"
        self.ml_info.setText(f"{m.get('release') or '?'} - version {m.get('version')} - installation {ok}. "
                             f"Simulink : {'oui' if m.get('simulink') else 'non'}. Python supporté par l'Engine : {py}.")

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
            self.gpu_info.setText("Aucune carte NVIDIA détectée : les entraînements utiliseront le CPU." if self.hw
                                  else "Matériel non scanné : clique sur « Scanner le matériel ».")

    def _refresh_env_info(self):
        self.rt_status.setText(runtime.status(self.settings)["message"])
        job = WorkerJob(self.settings, ["env-info"], self)
        job.done.connect(self._env_info_done)
        self.rt_versions.setText("Vérification des composants...")
        job.start()

    def _env_info_done(self, res, err):
        if res is None:
            self.rt_versions.setText("Vérification impossible : " + err)
            self.torch_info.setText("")
            self.ml_engine.setText("État inconnu (composants de calcul indisponibles).")
            return
        self.env_info = res
        yes = lambda v: v or "non installé"  # noqa: E731
        self.rt_versions.setText(
            f"Python {res.get('python')}  -  PyTorch {yes(res.get('torch'))}  -  Stable-Baselines3 {yes(res.get('stable_baselines3'))}"
            f"  -  Gymnasium {yes(res.get('gymnasium'))}")
        if res.get("torch"):
            gpu = f"GPU utilisable : {res['gpu']} (CUDA {res.get('cuda_version')})" if res.get("cuda_available") \
                else "PyTorch est en version CPU (ou aucun GPU utilisable)."
            self.torch_info.setText(f"PyTorch {res['torch']}. {gpu}")
        else:
            self.torch_info.setText("PyTorch n'est pas installé.")
        self.ml_engine.setText(f"matlabengine {res['matlabengine']} installé." if res.get("matlabengine")
                               else "matlabengine n'est pas installé dans les composants de calcul.")

    def _test_matlab(self):
        self._apply_to_settings()
        self.ml_test.setProperty("tone", "")
        self.ml_test.setText("Démarrage de MATLAB (jusqu'à 30 s)...")
        self.ml_test.show()
        job = WorkerJob(self.settings, ["matlab-test"], self)
        job.done.connect(self._matlab_test_done)
        job.start()

    def _matlab_test_done(self, res, err):
        tone = "ok" if res else "bad"
        if res:
            text = (f"MATLAB {res['version']} démarré en {res['startup_s']} s. Simulink : "
                    f"{'licence valide' if res['simulink'] else 'licence ABSENTE'}.")
            if not res["simulink"]:
                tone = "bad"
        else:
            text = "Échec : " + err
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
