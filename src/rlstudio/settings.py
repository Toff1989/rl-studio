"""Reglages generaux de l'application (fichier JSON par utilisateur)."""
import json
from pathlib import Path

from .paths import USER_DIR, default_projects_dir

DEFAULTS = {
    "theme": "Sombre",
    "custom_themes": {},
    "projects_dir": "",       # vide = dossier par defaut
    "matlab_enabled": True,
    "matlab_root": "",        # vide = detection automatique de la version la plus recente
    "gpu_enabled": True,      # False : tous les entrainements sont forces sur CPU
    "python_runtime": "",     # vide = runtime installe par l'installateur (ou l'interpreteur courant en dev)
    "hardware": {},           # derniere detection du materiel
    "plan": {},               # derniers composants installes
}


class Settings:
    def __init__(self, path=None):
        self.path = Path(path) if path else USER_DIR / "settings.json"
        self.data = {}
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def reload(self):
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default if default is not None else DEFAULTS.get(key)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def update(self, values: dict):
        self.data.update(values)
        self.save()

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # un echec d'ecriture ne doit jamais empecher d'utiliser l'application

    @property
    def projects_dir(self) -> Path:
        return Path(self.get("projects_dir") or default_projects_dir())
