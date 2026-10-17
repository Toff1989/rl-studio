"""Controle de conformite d'un build PyInstaller : `python packaging/check_bundle.py dist/rlstudio`.

Qt est utilise sous licence LGPL v3. Certains modules Qt ne sont disponibles qu'en GPL v3 / licence commerciale
(documentation Qt : Virtual Keyboard, Charts, Data Visualization...). Le controle est a LISTE BLANCHE : tout module
Qt non liste ici fait echouer le build, pour obliger a verifier sa licence avant de l'embarquer.
"""
import re
import sys
from pathlib import Path

ALLOWED_QT_MODULES = {"Core", "Gui", "Widgets", "Network", "OpenGL", "OpenGLWidgets", "Svg", "Test"}  # LGPL v3
ALLOWED_PLUGIN_DIRS = {"platforms", "styles", "imageformats", "iconengines"}
REQUIRED_FILES = ["LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/LGPL-3.0.txt", "licenses/GPL-3.0.txt"]


def check(bundle) -> list[str]:
    bundle = Path(bundle)
    problems = []
    pyside = next((p for p in bundle.rglob("PySide6") if p.is_dir() and (p / "plugins").exists() or
                   any(p.glob("Qt6*.dll"))), None)
    if pyside is None:
        return [f"PySide6 introuvable dans {bundle}"]

    for dll in sorted(pyside.glob("Qt6*.dll")):
        module = re.sub(r"^Qt6", "", dll.stem)
        if module not in ALLOWED_QT_MODULES:
            problems.append(f"module Qt non autorise : {dll.name} (verifier sa licence, puis l'ajouter a la liste blanche)")
    for pyd in sorted(pyside.glob("Qt*.pyd")):
        if pyd.stem[2:] not in ALLOWED_QT_MODULES:
            problems.append(f"module PySide6 non autorise : {pyd.name}")
    plugins = pyside / "plugins"
    if plugins.exists():
        for d in sorted(p for p in plugins.iterdir() if p.is_dir()):
            if d.name not in ALLOWED_PLUGIN_DIRS:
                problems.append(f"plugin Qt non autorise : plugins/{d.name}")
    # liaison dynamique : les bibliotheques Qt doivent rester des fichiers remplacables (LGPL v3)
    if not (pyside / "Qt6Core.dll").is_file():
        problems.append("Qt6Core.dll absente en tant que fichier separe : Qt doit rester remplacable par l'utilisateur")
    for rel in REQUIRED_FILES:
        if not (bundle / rel).is_file():
            problems.append(f"fichier de licence manquant : {rel}")
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bundle = Path(argv[0] if argv else "dist/rlstudio")
    problems = check(bundle)
    if problems:
        print("CONTROLE DES LICENCES : ECHEC")
        for p in problems:
            print(" -", p)
        return 1
    print(f"Controle des licences OK ({bundle}) : modules Qt LGPL uniquement, textes de licence presents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
