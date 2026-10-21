"""Onglet Algorithme."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QPlainTextEdit, QSpinBox,
)

from ..training import ALGOS
from .widgets import Card, column, scroll_page


class AlgoTab:
    """Methodes de MainWindow (onglet algorithme)."""

    def _tab_algo(self):
        self.a_name = QComboBox()
        self.a_name.addItems(ALGOS)
        self.a_steps = QSpinBox()
        self.a_steps.setRange(100, 100_000_000)
        self.a_steps.setSingleStep(10_000)
        self.a_steps.setGroupSeparatorShown(True)
        self.a_steps.setSuffix(" pas")
        self.a_seed = QSpinBox()
        self.a_seed.setRange(0, 99999)
        self.a_device = QComboBox()
        self.a_device.addItems(["cpu", "cuda", "auto"])
        self.a_base = QSpinBox()
        self.a_base.setRange(0, 100)
        self.a_base.setSuffix(" episode(s)")
        self.a_kwargs = QPlainTextEdit()
        self.a_kwargs.setObjectName("mono")
        self.a_kwargs.setPlaceholderText("learning_rate: 0.0003\nbatch_size: 256")
        self.a_kwargs.setMinimumHeight(180)

        base = Card("Entrainement", "Les petits reseaux MLP sont generalement plus rapides sur CPU que sur GPU.")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)
        form.addRow("Algorithme", self.a_name)
        form.addRow("Duree", self.a_steps)
        form.addRow("Graine aleatoire", self.a_seed)
        form.addRow("Peripherique", self.a_device)
        form.addRow("Evaluation de reference", self.a_base)
        base.body.addLayout(form)
        hyper = Card("Hyperparametres", "YAML transmis a Stable-Baselines3 (optionnel). Laisse vide pour les valeurs par defaut.")
        hyper.body.addWidget(self.a_kwargs)

        self.a_name.currentIndexChanged.connect(self._mark_dirty)
        self.a_device.currentIndexChanged.connect(self._mark_dirty)
        for s in (self.a_steps, self.a_seed, self.a_base):
            s.valueChanged.connect(self._mark_dirty)
        self.a_kwargs.textChanged.connect(self._mark_dirty)
        return scroll_page(column(base, hyper), 720)
