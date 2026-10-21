"""Onglet Algorithme : choix et réglages de l'apprentissage."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QPlainTextEdit, QSpinBox,
)

from ..training import ALGOS
from .widgets import Card, column, form_row, scroll_page, tips


class AlgoTab:
    """Méthodes de MainWindow (onglet Algorithme)."""

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

        base = Card("Entraînement", "Les petits réseaux MLP sont généralement plus rapides sur CPU que sur GPU.")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)
        form_row(form, "Algorithme", self.a_name,
                 "Méthode d'apprentissage. SAC : polyvalent et économe en essais (conseillé pour débuter). TD3 : proche de SAC. "
                 "PPO : stable mais demande beaucoup plus de pas.")
        form_row(form, "Durée", self.a_steps,
                 "Nombre total de pas de simulation de l'apprentissage. Plus c'est long, meilleur est souvent l'agent, mais plus l'attente est grande. "
                 "Commencez court pour vérifier que tout fonctionne.")
        form_row(form, "Graine aléatoire", self.a_seed,
                 "Fixe le hasard de l'entraînement : la même graine redonne le même résultat. Changez-la pour vérifier que le résultat ne dépend pas de la chance.")
        form_row(form, "Périphérique", self.a_device,
                 "Où se font les calculs : cpu (processeur), cuda (carte graphique NVIDIA) ou auto (carte graphique si elle est disponible et autorisée dans les Paramètres). "
                 "Pour de petits réseaux, le processeur est souvent aussi rapide.")
        form_row(form, "Évaluation de référence", self.a_base,
                 "Nombre d'épisodes joués avant l'apprentissage avec la politique de référence du projet (sinon des actions aléatoires). "
                 "Leur récompense moyenne est tracée en rouge pour juger les progrès. 0 = pas de référence.")
        base.body.addLayout(form)
        hyper = Card("Hyperparamètres", "YAML transmis à Stable-Baselines3 (optionnel). Laisse vide pour les valeurs par défaut.")
        hyper.body.addWidget(self.a_kwargs)
        tips(self, {"a_kwargs": "Réglages avancés de Stable-Baselines3 au format YAML (ex. learning_rate: 0.0003). Laissez vide pour les valeurs par défaut."})

        self.a_name.currentIndexChanged.connect(self._mark_dirty)
        self.a_device.currentIndexChanged.connect(self._mark_dirty)
        for s in (self.a_steps, self.a_seed, self.a_base):
            s.valueChanged.connect(self._mark_dirty)
        self.a_kwargs.textChanged.connect(self._mark_dirty)
        return scroll_page(column(base, hyper), 720)
