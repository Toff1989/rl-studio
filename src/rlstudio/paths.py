"""Chemins de l'application, en developpement comme une fois gelee par PyInstaller.

Disposition de l'application installee (identique au dossier `dist/rlstudio` du build) :

    rlstudio.exe, rlstudio-tools.exe   interface et outils console
    resources/{templates,matlab,docs}  modeles de projets, scripts MATLAB et guide d'utilisation
    lib/rlstudio/                      code Python pur, utilise par le runtime de calcul
    vendor/                            Python embarqué, get-pip.py, requirements-runtime.txt
"""
import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else REPO_ROOT

RESOURCE_DIR = Path(os.environ.get("RLSTUDIO_RESOURCES") or (APP_DIR / "resources" if FROZEN else REPO_ROOT))
TEMPLATES_DIR = RESOURCE_DIR / "templates"
MATLAB_DIR = RESOURCE_DIR / "matlab"
GUIDE_FILE = RESOURCE_DIR / "docs" / "guide.md" if FROZEN else REPO_ROOT / "docs" / "guide.md"
VENDOR_DIR = APP_DIR / "vendor" if FROZEN else REPO_ROOT / "packaging" / "vendor"
ICON_FILE = RESOURCE_DIR / "icon.ico" if FROZEN else REPO_ROOT / "packaging" / "icon.ico"
# mentions legales (copiees a cote de l'exe par build.ps1 ; a la racine du depot en developpement)
LICENSE_FILE = APP_DIR / "LICENSE"
NOTICES_FILE = APP_DIR / "THIRD_PARTY_NOTICES.md"
LICENSES_DIR = APP_DIR / "licenses"
# code Python pur vu par le runtime de calcul (PYTHONPATH)
LIB_DIR = APP_DIR / "lib" if FROZEN else REPO_ROOT / "src"


def _user_dir() -> Path:
    if os.environ.get("RLSTUDIO_HOME"):
        return Path(os.environ["RLSTUDIO_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "RLStudio"
    return Path.home() / ".local" / "share" / "rlstudio"


# reglages, runtime de calcul, journaux : un dossier par utilisateur
USER_DIR = _user_dir()
RUNTIME_DIR = USER_DIR / "runtime"
LOGS_DIR = USER_DIR / "logs"


def default_projects_dir() -> Path:
    if os.environ.get("RLSTUDIO_PROJECTS"):
        return Path(os.environ["RLSTUDIO_PROJECTS"])
    return Path.home() / "Documents" / "RLStudio" / "projects" if FROZEN else REPO_ROOT / "projects"
