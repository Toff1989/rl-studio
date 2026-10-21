"""Interface graphique : fenêtre principale (assemblage des onglets et des actions)."""
import sys

from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtCore import QSize, QTimer
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow, QMenu, QMessageBox, QTabWidget, QVBoxLayout,
    QWidget,
)

from ..paths import ICON_FILE
from ..project import ProjectStore
from ..settings import Settings
from . import theme
from .. import runtime
from ..pyversion import support_warning
from .distill_tab import DistillTab
from .identify_tab import IdentifyTab
from .eval_tab import EvalTab
from .settings_dialog import SettingsDialog
from .guide import TAB_SECTIONS, GuideDialog
from .notify import ErrorBanner, install_excepthook
from .widgets import button, label
from .project_actions import ProjectActions
from .tab_project import ProjectTab
from .tab_env import EnvTab
from .tab_algo import AlgoTab
from .tab_train import TrainTab
from .tab_runs import RunsTab

class MainWindow(ProjectActions, ProjectTab, EnvTab, AlgoTab, TrainTab, RunsTab, QMainWindow):
    def __init__(self, store=None, settings=None):
        super().__init__()
        self.settings = settings or Settings()
        self._plots = []
        self.theme_name = theme.DEFAULT_THEME
        self._apply_theme(self.settings.get("theme", theme.DEFAULT_THEME))
        self.store = store or ProjectStore(root=self.settings.projects_dir)
        self.project = None
        self.dirty = False
        self._loading = False
        self.worker = None
        self._rew_timer = QTimer(self)
        self._rew_timer.setSingleShot(True)
        self._rew_timer.setInterval(40)
        self._rew_timer.timeout.connect(self._flush_rew)
        self.setWindowTitle("RL Studio")
        self.resize(1440, 900)
        if ICON_FILE.exists():  # icone de la fenetre, de la barre des taches et des boites de dialogue
            self._icon = QIcon(str(ICON_FILE))
            QApplication.instance().setWindowIcon(self._icon)
            self.setWindowIcon(self._icon)
        self._build_ui()
        install_excepthook(self.banner)
        warning = support_warning()
        if warning:
            self.report_error(warning, level="warning")
        QShortcut(QKeySequence("F1"), self, activated=self.open_guide)
        self.setAcceptDrops(True)
        self._refresh_list()

    def _all_themes(self):
        return {**theme.BUILTIN, **(self.settings.get("custom_themes") or {})}

    def _apply_theme(self, name, save=False, base=None):
        """Applique un theme par nom (et le memorise si save) ou directement un dict `base` (apercu)."""
        if base is None:
            themes = self._all_themes()
            if name not in themes:
                name = theme.DEFAULT_THEME
            base = themes[name]
            self.theme_name = name
            if save:
                self.settings.set("theme", name)
        theme.apply(QApplication.instance(), base)
        if self._plots:
            self._style_plots()
            if self.runs_list.currentItem():
                self._show_run(self.runs_list.currentItem(), None)

    def report_error(self, summary, detail="", level="error"):
        """Affiche une erreur dans le bandeau du haut de la fenêtre (visible quel que soit l'onglet)."""
        self.banner.report(summary, detail, level)

    def open_guide(self, *_):
        """Guide d'utilisation (non modal, ouvert à la section de l'onglet affiché)."""
        if getattr(self, "_guide", None) is None:
            self._guide = GuideDialog(self)
        self._guide.show_section(TAB_SECTIONS.get(self.tabs.tabText(self.tabs.currentIndex())))

    def _open_settings(self, page=0):
        if self.worker:
            QMessageBox.information(self, "Paramètres", "Arrête l'entraînement en cours avant de modifier les paramètres.")
            return
        page = page if isinstance(page, int) and not isinstance(page, bool) else 0
        dlg = SettingsDialog(self, self.settings, self._apply_theme, self._all_themes, self.theme_name)
        dlg.nav.setCurrentRow(page)
        if dlg.exec() == QDialog.Accepted and dlg.projects_dir_changed:
            if self.dirty:
                self._save()
            self.store = ProjectStore(root=self.settings.projects_dir)
            self.project = None
            self._set_dirty(False)
            self._refresh_list()
        self._update_runtime_warning()

    def _update_runtime_warning(self):
        self.btn_runtime_warn.setVisible(not runtime.status(self.settings)["ok"])

    def _build_ui(self):
        # barre laterale
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(280)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 20, 0, 16)
        sl.setSpacing(10)
        head = QHBoxLayout()
        head.setContentsMargins(20, 0, 10, 0)
        head.setSpacing(10)
        if ICON_FILE.exists():
            logo = QLabel()
            logo.setPixmap(QIcon(str(ICON_FILE)).pixmap(QSize(36, 36)))
            logo.setFixedSize(36, 36)
            head.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("RL Studio", "appTitle"))
        titles.addWidget(label("Apprentissage par renforcement", "muted"))
        head.addLayout(titles, 1)
        sl.addLayout(head)
        newrow = QHBoxLayout()
        newrow.setContentsMargins(16, 6, 16, 0)
        newrow.addWidget(button("+  Nouveau projet", "primary", self._new,
                                tip="Crée un projet à partir d'un modèle de départ (Python, Simulink, pendule, chauffage, ventilation…)."))
        sl.addLayout(newrow)
        improw = QHBoxLayout()
        improw.setContentsMargins(16, 0, 16, 0)
        improw.addWidget(button("Importer un projet (.rls)...", "ghost", self._import_dialog,
                                tip="Ouvre un projet partagé (fichier .rls). Vous pouvez aussi glisser-déposer le fichier dans la fenêtre. "
                                    "N'importez que des projets de personnes de confiance : ils contiennent du code."))
        sl.addLayout(improw)
        convrow = QHBoxLayout()
        convrow.setContentsMargins(16, 0, 16, 0)
        convrow.addWidget(button("Convertir un modèle Simulink...", "ghost", self._convert_dialog,
                                 tip="Prépare un modèle Simulink existant pour l'apprentissage (assistant en 4 étapes, MATLAB requis). "
                                     "Le fichier d'origine n'est jamais modifié : une copie est créée dans un nouveau projet."))
        sl.addLayout(convrow)
        lab = label("PROJETS", "sectionLabel", tip="Les projets du dossier configuré dans Paramètres → Général.")
        lab.setContentsMargins(20, 8, 0, 0)
        sl.addWidget(lab)
        self.plist = QListWidget()
        self.plist.setToolTip("Vos projets. Cliquez sur l'un d'eux pour l'ouvrir (impossible pendant un calcul en cours).")
        self.plist.currentItemChanged.connect(self._on_select)
        sl.addWidget(self.plist, 1)
        self.btn_runtime_warn = button("⚠  Composants de calcul manquants", "danger", lambda: self._open_settings(4),
                                       tip="Python, PyTorch ou MATLAB Engine ne sont pas installés : cliquez pour ouvrir Paramètres → Composants et les installer.")
        wrow = QHBoxLayout()
        wrow.setContentsMargins(12, 0, 12, 0)
        wrow.addWidget(self.btn_runtime_warn)
        sl.addLayout(wrow)
        self.btn_actions = button("Actions du projet  ▾", "ghost",
                                  tip="Dupliquer, renommer, exporter, ouvrir le dossier ou supprimer le projet sélectionné.")
        menu = QMenu(self.btn_actions)
        menu.setToolTipsVisible(True)
        self.act_dup = menu.addAction("Dupliquer", self._duplicate)
        self.act_dup.setToolTip("Crée une copie du projet (réglages, environnement et résultats).")
        self.act_ren = menu.addAction("Renommer...", self._rename)
        self.act_ren.setToolTip("Change le nom du projet (et de son dossier).")
        self.act_dir = menu.addAction("Ouvrir le dossier", self._open_folder)
        self.act_dir.setToolTip("Affiche le dossier du projet dans l'Explorateur de fichiers.")
        self.act_exp = menu.addAction("Exporter (.rls)...", self._export_dialog)
        self.act_exp.setToolTip("Enregistre le projet dans un fichier .rls unique pour le partager ou le sauvegarder.")
        menu.addSeparator()
        self.act_del = menu.addAction("Supprimer...", self._delete)
        self.act_del.setToolTip("Déplace le projet dans la corbeille projects/.trash/ : il peut être récupéré.")
        self.btn_actions.setMenu(menu)
        arow = QHBoxLayout()
        arow.setContentsMargins(12, 0, 12, 0)
        arow.addWidget(self.btn_actions)
        sl.addLayout(arow)
        self.btn_guide = button("?  Guide d'utilisation", "ghost", self.open_guide,
                                tip="Ouvre le guide d'utilisation (touche F1), à la section de l'onglet affiché.")
        grow = QHBoxLayout()
        grow.setContentsMargins(12, 0, 12, 0)
        grow.addWidget(self.btn_guide)
        sl.addLayout(grow)
        self.btn_settings = button("⚙  Paramètres", "ghost", self._open_settings,
                                   tip="Apparence, dossier des projets, MATLAB, GPU, composants de calcul et guide d'utilisation.")
        srow = QHBoxLayout()
        srow.setContentsMargins(12, 0, 12, 0)
        srow.addWidget(self.btn_settings)
        sl.addLayout(srow)
        self._update_runtime_warning()

        # zone principale : barre du haut + onglets
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        top = QWidget()
        top.setObjectName("topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(28, 18, 28, 10)
        tl.setSpacing(12)
        self.top_title = label("", "h1", tip="Nom du projet ouvert.")
        self.top_badge = label("", "badge", tip="Type du projet : Python (environnement écrit en code) ou Simulink (modèle décrit sans code).")
        self.lbl_dirty = label("●  Modifications non enregistrées", "dirty",
                               tip="Des champs ont été modifiés : cliquez sur Enregistrer. L'entraînement enregistre aussi avant de démarrer.")
        self.lbl_dirty.hide()
        self.btn_save = button("Enregistrer", "primary", self._save,
                               tip="Enregistre les modifications du projet (fichier project.yaml). Actif seulement s'il y a des modifications.")
        tl.addWidget(self.top_title)
        tl.addWidget(self.top_badge)
        tl.addStretch(1)
        tl.addWidget(self.lbl_dirty)
        tl.addWidget(self.btn_save)
        ml.addWidget(top)
        self.banner = ErrorBanner()
        ml.addWidget(self.banner)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._tab_general(), "Projet")
        self.tabs.addTab(self._tab_env(), "Environnement")
        self.tabs.addTab(self._tab_algo(), "Algorithme")
        self.tabs.addTab(self._tab_train(), "Entraînement")
        self.tabs.addTab(self._tab_runs(), "Résultats")
        self.distill = DistillTab(self.settings)
        self.identify = IdentifyTab(self.settings)
        self.evaltab = EvalTab(self.settings)
        for tab in (self.distill, self.identify, self.evaltab):
            tab.busy.connect(lambda b: self.plist.setEnabled(not b))
            tab.error.connect(self.report_error)
        self.tabs.insertTab(4, self.evaltab, "Évaluation")
        self.tabs.addTab(self.distill, "Modèle mathématique")
        self.tabs.addTab(self.identify, "Identification")
        for i, tip_text in enumerate([
                "Fiche du projet : nom, description et indicateurs.",
                "Le système à piloter : fichier Python ou modèle Simulink, observations, actions, récompense.",
                "L'algorithme d'apprentissage (SAC, PPO, TD3), la durée de l'entraînement et ses réglages.",
                "Lance l'entraînement et suit les courbes en direct.",
                "Rejoue un entraînement terminé pour voir l'agent à l'œuvre.",
                "La liste des entraînements du projet et leur configuration.",
                "Extrait de l'agent une loi de commande lisible (PID, table, arbre, équation).",
                "Remplace un modèle par un modèle réduit (équations ou réseau), sans entraînement."]):
            self.tabs.setTabToolTip(i, tip_text)
        ml.addWidget(self.tabs, 1)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(side)
        root.addWidget(main, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._set_dirty(False)

    def closeEvent(self, event):
        if self.dirty and not self._ask_save():
            event.ignore()
            return
        if self.worker:
            self.worker.stop()
            if not self.worker.proc.waitForFinished(8000):
                self.worker.proc.kill()
        super().closeEvent(event)



def main(open_file=None):
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    open_file = open_file or next((a for a in sys.argv[1:] if a.lower().endswith(".rls")), None)
    if open_file:
        QTimer.singleShot(200, lambda: win.import_file(open_file))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
