"""Composants d'interface reutilisables."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

def button(text, variant=None, slot=None):
    b = QPushButton(text)
    if variant:
        b.setProperty("variant", variant)
    if slot:
        b.clicked.connect(slot)
    b.setCursor(Qt.PointingHandCursor)
    return b


def label(text="", name=None, wrap=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


class Card(QFrame):
    """Bloc arrondi avec titre optionnel ; remplir via `.body` (QVBoxLayout)."""

    def __init__(self, title=None, hint=None):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 16)
        self.body.setSpacing(10)
        if title:
            self.body.addWidget(label(title, "cardTitle"))
        if hint:
            self.body.addWidget(label(hint, "hint", wrap=True))


class Kpi(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("card")
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
