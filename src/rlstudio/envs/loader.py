"""Construit l'environnement Gymnasium d'un projet (Python custom ou Simulink generique)."""
import importlib.util
import sys
from pathlib import Path

from ..project import Project


def load_module(project: Project, filename: str):
    path = (project.path / filename).resolve()
    if project.path.resolve() not in path.parents:  # un projet importe ne doit pas executer du code hors de son dossier
        raise ValueError(f"Fichier hors du dossier du projet : {filename}")
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    name = f"rlstudio_project_{project.slug}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))  # permet les imports relatifs entre fichiers du projet
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def make_env(project: Project, **overrides):
    env_cfg = project.cfg["env"]
    if project.kind == "python":
        entry = env_cfg.get("entry", "env.py:make_env")
        filename, _, func = entry.partition(":")
        module = load_module(project, filename)
        params = {**(env_cfg.get("params") or {}), **overrides}
        return getattr(module, func or "make_env")(**params)
    if project.kind == "simulink":
        from .simulink import SimulinkEnv

        return SimulinkEnv(project.path, env_cfg, **overrides)
    raise ValueError(f"Type de projet inconnu : {project.kind!r}")


def make_baseline(project: Project, env):
    """Politique de reference definie par le projet (`baseline_policy(env)` dans son env.py)."""
    if project.kind != "python":
        return None
    filename = project.cfg["env"].get("entry", "env.py:make_env").partition(":")[0]
    module = load_module(project, filename)
    fn = getattr(module, "baseline_policy", None)
    return fn(env) if fn else None
