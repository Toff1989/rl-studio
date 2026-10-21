"""Guide d'utilisation : lecteur du fichier docs/guide.md (sommaire, recherche), dans une fenêtre ou une page des Paramètres."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextDocument
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QTextBrowser, QVBoxLayout, QWidget

from ..paths import GUIDE_FILE
from .widgets import button, label

# onglet de la fenêtre principale -> début du titre de la section du guide
TAB_SECTIONS = {
    "Projet": "5.1", "Environnement": "5.2", "Algorithme": "5.3", "Entraînement": "5.4", "Évaluation": "5.5",
    "Résultats": "5.6", "Modèle mathématique": "5.7", "Identification": "5.8",
}
MISSING = "# Guide introuvable\n\nLe fichier du guide d'utilisation est absent de cette installation (docs/guide.md)."


def load_guide_text() -> str:
    try:
        return GUIDE_FILE.read_text(encoding="utf-8")
    except OSError:
        return MISSING


class GuideView(QWidget):
    """Sommaire à gauche, texte à droite, recherche en haut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setMarkdown(load_guide_text())
        self.browser.setToolTip("Guide d'utilisation de RL Studio. Utilisez le sommaire à gauche ou la recherche pour naviguer.")
        self.toc = QListWidget()
        self.toc.setFixedWidth(260)
        self.toc.setWordWrap(True)
        self.toc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.toc.setToolTip("Sommaire du guide : cliquez sur une section pour y aller.")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher dans le guide…")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Tapez un mot puis Entrée : le guide se place sur la première occurrence ; Entrée de nouveau pour la suivante.")
        self.search.returnPressed.connect(self.find_next)
        self.status = label("", "muted")
        self._headings = []  # (position, niveau, texte)
        self._index_headings()
        self.toc.itemClicked.connect(lambda it: self._scroll_to(it.data(Qt.UserRole)))

        left = QVBoxLayout()
        left.addWidget(self.search)
        left.addWidget(self.toc, 1)
        left.addWidget(self.status)
        side = QWidget()
        side.setLayout(left)
        side.setFixedWidth(260)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(side)
        lay.addWidget(self.browser, 1)

    def _index_headings(self):
        block = self.browser.document().begin()
        while block.isValid():
            level = block.blockFormat().headingLevel()
            text = block.text().strip()
            if level and text:
                self._headings.append((block.position(), level, text))
                if level <= 3:
                    it = QListWidgetItem("\u00a0\u00a0\u00a0" * (level - 1) + text)
                    it.setData(Qt.UserRole, block.position())
                    self.toc.addItem(it)
            block = block.next()

    def _scroll_to(self, position):
        cursor = QTextCursor(self.browser.document())
        cursor.setPosition(position)
        self.browser.setTextCursor(cursor)
        bar = self.browser.verticalScrollBar()
        bar.setValue(bar.value() + self.browser.cursorRect().top())  # le titre vient en haut de la zone visible

    def goto_section(self, prefix: str) -> bool:
        """Se place sur le titre qui commence par `prefix` (ex. « 5.2 »)."""
        for pos, _level, text in self._headings:
            if text.startswith(prefix):
                self._scroll_to(pos)
                return True
        return False

    def find_next(self):
        text = self.search.text().strip()
        if not text:
            return
        doc = self.browser.document()
        cursor = doc.find(text, self.browser.textCursor())
        if cursor.isNull():
            cursor = doc.find(text)  # reprise au début
        if cursor.isNull():
            self.status.setText("Aucun résultat.")
            return
        self.status.setText("")
        self.browser.setTextCursor(cursor)
        self.browser.ensureCursorVisible()


class GuideDialog(QDialog):
    """Fenêtre du guide, non modale : elle reste ouverte à côté de l'application (y compris pendant un calcul)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guide d'utilisation")
        self.resize(1080, 720)
        self.setModal(False)
        self.view = GuideView()
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Fermer", "primary", self.close, tip="Ferme le guide (F1 le rouvre)."))
        lay.addLayout(row)

    def show_section(self, prefix=None):
        self.show()
        self.raise_()
        self.activateWindow()
        if prefix:
            self.view.goto_section(prefix)
