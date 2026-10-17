"""rlstudio-scan.exe : detection du materiel, bibliotheque standard uniquement (execute par l'installateur).

    rlstudio-scan.exe --out hw.json     ecrit hw.json (donnees) et hw.txt (resume lisible)
"""
import argparse
import io
import json
import os
import sys

from rlstudio import hardware

if sys.stdout is None:
    sys.stdout = io.StringIO()

ap = argparse.ArgumentParser()
ap.add_argument("--out")
args = ap.parse_args()
hw = hardware.scan()
if args.out:
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(hw, fh, indent=2, ensure_ascii=False)
    with open(os.path.splitext(args.out)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(hardware.summary(hw))
print(hardware.summary(hw))
