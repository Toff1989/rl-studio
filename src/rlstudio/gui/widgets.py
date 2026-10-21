"""Composants d'interface reutilisables."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

def tip(widget, text):
    """Infobulle (survol) sur un widget ; renvoie le widget pour pouvoir chaîner."""
    if text:
        widget.setToolTip(text)
    return widget


def tips(owner, mapping):
    """Infobulles en lot : {nom d'attribut de `owner`: texte}. Une faute de frappe dans le nom échoue tout de suite."""
    for name, text in mapping.items():
        tip(getattr(owner, name), text)


def header_tips(table, texts):
    """Infobulles des en-têtes de colonnes d'un tableau (liste alignée sur les colonnes ; texte vide = aucune)."""
    for j, text in enumerate(texts):
        item = table.horizontalHeaderItem(j)
        if item is not None and text:
            item.setToolTip(text)


def form_row(form, text, widget, tooltip):
    """Ligne de formulaire dont l'étiquette et le champ portent la même infobulle."""
    form.addRow(tip(QLabel(text), tooltip), tip(widget, tooltip))


def button(text, variant=None, slot=None, tip=None):
    b = QPushButton(text)
    if variant:
        b.setProperty("variant", variant)
    if slot:
        b.clicked.connect(slot)
    b.setCursor(Qt.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    return b


def label(text="", name=None, wrap=False, tip=None):
    lb = QLabel(text)
    if tip:
        lb.setToolTip(tip)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


class Card(QFrame):
    """Bloc arrondi avec titre optionnel ; remplir via `.body` (QVBoxLayout)."""

    def __init__(self, title=None, hint=None, tip=None):
        super().__init__()
        self.setObjectName("card")
        if tip:
            self.setToolTip(tip)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 16)
        self.body.setSpacing(10)
        if title:
            self.body.addWidget(label(title, "cardTitle"))
        if hint:
            self.body.addWidget(label(hint, "hint", wrap=True))


class Kpi(QFrame):
    def __init__(self, title, tip=None):
        super().__init__()
        self.setObjectName("card")
        if tip:
            self.setToolTip(tip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        lay.addWidget(label(title.upper(), "kpiTitle"))
        self.value = label("-", "kpiValue", wrap=True)
        lay.addWidget(self.value)

    def set(self, text):
        self.value.setText(text)


def scroll_page(inner: QWidget, max_width=None):
    """Page defilante avec marges ; `max_width` evite les champs etires sur grand ecran."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(28, 24, 28, 24)
    if max_width:
        inner.setMaximumWidth(max_width)
        row.addWidget(inner, 100)
        row.addStretch(1)
    else:
        row.addWidget(inner)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(holder)
    return area


def column(*items, spacing=16):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for it in items:
        lay.addWidget(it) if isinstance(it, QWidget) else lay.addLayout(it)
    lay.addStretch(1)
    return w


def fmt_run_date(name: str) -> str:
    """AAAAMMJJ-HHMMSS -> JJ/MM/AAAA HH:MM"""
    if len(name) >= 13 and name[8] == "-":
        return f"{name[6:8]}/{name[4:6]}/{name[0:4]}  {name[9:11]}:{name[11:13]}"
    return name
