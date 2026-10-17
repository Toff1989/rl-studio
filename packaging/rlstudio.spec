# -*- mode: python ; coding: utf-8 -*-
# PyInstaller : une analyse, deux executables (interface sans console + outils console) dans le meme dossier.
# Le calcul (PyTorch, Stable-Baselines3, MATLAB Engine) n'est PAS embarque : il vit dans le runtime
# installe par l'installateur selon le materiel (voir src/rlstudio/components.py).
from pathlib import Path

ROOT = Path(SPECPATH).parent
ICON = str(ROOT / "packaging" / "icon.ico")

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    hiddenimports=["rlstudio.gui.app", "rlstudio.gui.settings_dialog", "rlstudio.gui.runner"],
    excludes=["torch", "torchvision", "stable_baselines3", "gymnasium", "matlab", "matplotlib", "tkinter",
              "scipy", "pandas", "IPython", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia"],
    noarchive=False,
)
# Qt est utilise sous LGPL v3 : on n'embarque QUE ce que l'interface utilise. En particulier Qt Virtual Keyboard
# (GPL v3 / commerciale seulement) est tire par defaut via le plugin platforminputcontexts. Voir
# THIRD_PARTY_NOTICES.md et packaging/check_bundle.py (liste blanche verifiee a chaque build).
DROP = ("qt6virtualkeyboard", "qtvirtualkeyboardplugin", "plugins/platforminputcontexts",
        "qt6quick", "qt6qml", "qt6pdf", "qpdf.dll",
        "plugins/tls/", "plugins/networkinformation/", "plugins/generic/")


def keep(entry):
    dest = entry[0].replace("\\", "/").lower()
    return not any(k in dest for k in DROP)


a.binaries = [e for e in a.binaries if keep(e)]
a.datas = [e for e in a.datas if keep(e)]
pyz = PYZ(a.pure)

gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="rlstudio", console=False, icon=ICON)
tools = EXE(pyz, a.scripts, [], exclude_binaries=True, name="rlstudio-tools", console=True, icon=ICON)

COLLECT(gui, tools, a.binaries, a.datas, strip=False, upx=False, name="rlstudio")
