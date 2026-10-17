"""Projets RL : un dossier = un projet (project.yaml + fichiers + runs/)."""
import copy
import re
import shutil
import time
from pathlib import Path

import yaml

from .paths import TEMPLATES_DIR, default_projects_dir

DEFAULT_ALGO = {
    "name": "SAC",          # SAC | PPO | TD3
    "total_timesteps": 100_000,
    "seed": 0,
    "device": "cpu",
    "baseline_episodes": 3,
    "kwargs": {},
}


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
    return s or "projet"


class Project:
    def __init__(self, path):
        self.path = Path(path)
        self.cfg = {}
        self.reload()

    @property
    def slug(self):
        return self.path.name

    @property
    def name(self):
        return self.cfg.get("name", self.slug)

    @property
    def kind(self):
        return self.cfg.get("kind", "python")

    @property
    def runs_dir(self):
        return self.path / "runs"

    def reload(self):
        with open(self.path / "project.yaml", encoding="utf-8-sig") as fh:
            self.cfg = yaml.safe_load(fh) or {}
        algo = copy.deepcopy(DEFAULT_ALGO)
        algo.update(self.cfg.get("algo") or {})
        self.cfg["algo"] = algo
        self.cfg.setdefault("env", {})

    def save(self):
        text = yaml.safe_dump(self.cfg, sort_keys=False, allow_unicode=True)
        (self.path / "project.yaml").write_text(text, encoding="utf-8")

    def list_runs(self):
        if not self.runs_dir.exists():
            return []
        return sorted((d for d in self.runs_dir.iterdir() if d.is_dir()), reverse=True)

    def new_run_dir(self) -> Path:
        d = self.runs_dir / time.strftime("%Y%m%d-%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        return d


class ProjectStore:
    def __init__(self, root=None, templates=None):
        self.root = Path(root or default_projects_dir())
        self.templates = Path(templates or TEMPLATES_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- lecture ------------------------------------------------------------------------
    def list(self):
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and (d / "project.yaml").exists():
                out.append(Project(d))
        return out

    def get(self, slug) -> Project:
        return Project(self.root / slug)

    def list_templates(self):
        """[(id, nom, description)] ; chaque template est un dossier avec un project.yaml."""
        out = []
        for d in sorted(self.templates.iterdir()):
            f = d / "project.yaml"
            if f.exists():
                cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                out.append((d.name, cfg.get("template_name", cfg.get("name", d.name)),
                            cfg.get("template_description", cfg.get("description", ""))))
        return out

    # -- ecriture -----------------------------------------------------------------------
    def _free_slug(self, name):
        base, slug, i = slugify(name), slugify(name), 2
        while (self.root / slug).exists():
            slug, i = f"{base}-{i}", i + 1
        return slug

    def free_slug(self, name) -> str:
        return self._free_slug(name)

    def create(self, name, template="python_blank") -> Project:
        name = name.strip()
        if not name:
            raise ValueError("Le nom du projet est vide.")
        src = self.templates / template
        if not src.is_dir():
            raise ValueError(f"Template inconnu : {template}")
        dst = self.root / self._free_slug(name)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("runs", "__pycache__"))
        p = Project(dst)
        p.cfg["name"] = name
        p.cfg["created"] = time.strftime("%Y-%m-%d %H:%M")
        p.cfg.pop("template_name", None)
        p.cfg.pop("template_description", None)
        p.save()
        return p

    def duplicate(self, project: Project, new_name=None, with_runs=False) -> Project:
        new_name = new_name or f"{project.name} (copie)"
        dst = self.root / self._free_slug(new_name)
        ignore = ["__pycache__"] + ([] if with_runs else ["runs"])
        shutil.copytree(project.path, dst, ignore=shutil.ignore_patterns(*ignore))
        p = Project(dst)
        p.cfg["name"] = new_name
        p.cfg["created"] = time.strftime("%Y-%m-%d %H:%M")
        p.save()
        return p

    def rename(self, project: Project, new_name):
        """Change le nom affiche ; le dossier ne bouge pas (les chemins des runs restent valides)."""
        if not new_name.strip():
            raise ValueError("Le nom du projet est vide.")
        project.cfg["name"] = new_name.strip()
        project.save()

    def delete(self, project: Project) -> Path:
        """Deplace le projet dans .trash/ (recuperable) au lieu de l'effacer."""
        trash = self.root / ".trash"
        trash.mkdir(exist_ok=True)
        dst = trash / f"{project.slug}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(project.path), str(dst))
        return dst
