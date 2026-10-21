"""Choix de l'interpreteur de calcul et de son environnement (cote interface)."""
import os
import sys
from pathlib import Path

from . import components
from .paths import FROZEN, LIB_DIR, MATLAB_DIR, RESOURCE_DIR, USER_DIR
from .settings import Settings


def worker_python(settings: Settings) -> Path | None:
    """Python qui execute les entrainements : reglage manuel > runtime installe > interpreteur courant (dev)."""
    manual = settings.get("python_runtime")
    if manual and Path(manual).exists():
        return Path(manual)
    installed = components.runtime_python()
    if installed:
        return installed
    return None if FROZEN else Path(sys.executable)


def status(settings: Settings) -> dict:
    py = worker_python(settings)
    if py is None:
        return {"ok": False, "python": None, "kind": "missing",
                "message": "Composants de calcul non installés (Paramètres > Composants)."}
    kind = "manuel" if settings.get("python_runtime") else ("installe" if components.runtime_python() else "developpement")
    return {"ok": True, "python": str(py), "kind": kind, "message": f"Python de calcul : {py}"}


def worker_env(settings: Settings) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(LIB_DIR), env.get("PYTHONPATH", "")]))
    env["RLSTUDIO_RESOURCES"] = str(RESOURCE_DIR)
    env["RLSTUDIO_LIB"] = str(LIB_DIR)  # le Python embarqué ignore PYTHONPATH (voir components.SITECUSTOMIZE)
    env["RLSTUDIO_HOME"] = str(USER_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"  # le calcul ne depend que de son propre runtime
    root = settings.get("matlab_root")
    if settings.get("matlab_enabled") and root:
        env["RLSTUDIO_MATLAB_ROOT"] = root
        env["PATH"] = os.pathsep.join([str(Path(root) / "bin" / "win64"), env.get("PATH", "")])
    return env


def tools_command(args: list[str]) -> tuple[str, list[str]]:
    """Commande des outils console (scan, provision) : exe dedie une fois gele, module sinon."""
    if FROZEN:
        return str(Path(sys.executable).with_name("rlstudio-tools.exe")), list(args)
    return sys.executable, ["-m", "rlstudio.launcher", *args]
