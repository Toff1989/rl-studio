"""Point d'entree PyInstaller (rlstudio.exe et rlstudio-tools.exe)."""
import sys

from rlstudio.launcher import main

if __name__ == "__main__":
    sys.exit(main())
