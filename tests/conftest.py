"""Niveaux de dépendances des tests.

Le noyau (numpy, pyyaml) suffit à la plupart des tests. Deux marqueurs désignent ceux qui ont besoin de plus ; ils sont
ignorés (avec la raison affichée) quand les paquets manquent, au lieu d'échouer :

- `runtime` : gymnasium, stable-baselines3 et torch (extra `runtime` + PyTorch) : entraînement, extraction, évaluation ;
- `gui`     : PySide6 et pyqtgraph (extra `gui`) : fenêtres et graphiques.

Voir README, section Développement : `pip install -e ".[gui,runtime,dev]"` exécute tout.
"""
import importlib.util

import pytest

REQUIRES = {
    "runtime": ("gymnasium", "stable_baselines3", "torch"),
    "gui": ("PySide6", "pyqtgraph"),
}


def missing_modules(group):
    return [m for m in REQUIRES[group] if importlib.util.find_spec(m) is None]


def pytest_configure(config):
    config.addinivalue_line("markers", "runtime: nécessite gymnasium, stable-baselines3 et torch")
    config.addinivalue_line("markers", "gui: nécessite PySide6 et pyqtgraph")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for group in REQUIRES:
            if item.get_closest_marker(group):
                missing = missing_modules(group)
                if missing:
                    item.add_marker(pytest.mark.skip(reason=f"dépendance absente ({group}) : {', '.join(missing)}"))
