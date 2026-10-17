"""Point d'entree de l'application (et des outils console de l'installateur).

  rlstudio.exe                 interface graphique
  rlstudio.exe --selftest      demarre l'interface hors ecran et quitte (verification du build)
  rlstudio-tools.exe --scan --out hw.json          detecte le materiel
  rlstudio-tools.exe --provision [--hardware hw.json] [--no-gpu] [--no-matlab] [--dry-run]
"""
import argparse
import io
import json
import os
import sys


def _ensure_streams():
    """Un exe 'windowed' n'a pas de stdout/stderr : evite les crashs sur print()."""
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


def run_scan(out=None) -> dict:
    from . import hardware

    hw = hardware.scan()
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(hw, fh, indent=2, ensure_ascii=False)
        with open(os.path.splitext(out)[0] + ".txt", "w", encoding="utf-8") as fh:
            fh.write(hardware.summary(hw))
    print(hardware.summary(hw))
    return hw


def run_provision(args) -> int:
    from . import components
    from .paths import LOGS_DIR, RUNTIME_DIR
    from .settings import Settings

    settings = Settings()
    if args.hardware:
        with open(args.hardware, encoding="utf-8") as fh:
            hw = json.load(fh)
    else:
        print("Detection du materiel...")
        hw = run_scan()
    plan = components.make_plan(hw, gpu_enabled=not args.no_gpu and bool(settings.get("gpu_enabled")),
                                matlab_enabled=not args.no_matlab and bool(settings.get("matlab_enabled")),
                                matlab_root=settings.get("matlab_root") or "")
    print("\n--- Plan d'installation ---\n" + components.plan_summary(plan) + "\n")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logfile = open(LOGS_DIR / "provision.log", "a", encoding="utf-8")

    def log(line):
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    try:
        result = components.provision(plan, RUNTIME_DIR, log, skip=args.skip or (), dry_run=args.dry_run)
    except Exception as e:
        log(f"ERREUR : {e}")
        return 2
    finally:
        logfile.close()
    if not args.dry_run:
        update = {"hardware": hw, "plan": plan}
        if args.no_gpu:
            update["gpu_enabled"] = False
        if plan["matlab"].get("root") and not settings.get("matlab_root"):
            update["matlab_root"] = plan["matlab"]["root"]
        settings.update(update)
    print("\nTermine." if not result["errors"] else "\nTermine avec des avertissements : " + "; ".join(result["errors"]))
    return 0 if not result["errors"] else 1


def run_selftest() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from .gui.app import MainWindow
    from .project import ProjectStore
    from .settings import Settings
    import tempfile

    tmp = tempfile.mkdtemp()
    app = QApplication(sys.argv[:1])
    win = MainWindow(ProjectStore(root=os.path.join(tmp, "p")), Settings(os.path.join(tmp, "s.json")))
    win.show()
    app.processEvents()
    win.close()
    print("selftest OK")
    return 0


def main(argv=None):
    _ensure_streams()
    ap = argparse.ArgumentParser(prog="rlstudio")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--provision", action="store_true")
    ap.add_argument("--hardware")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-matlab", action="store_true")
    ap.add_argument("--skip", nargs="*", choices=["base", "torch", "matlab"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pause-on-error", action="store_true", help="attend une touche si l'installation echoue")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("file", nargs="?", help="fichier .rls a importer (double-clic dans l'Explorateur)")
    args = ap.parse_args(argv)
    if args.scan:
        run_scan(args.out)
        return 0
    if args.provision:
        code = run_provision(args)
        if code and args.pause_on_error and sys.stdin is not None and sys.stdin.isatty():
            input("\nAppuyez sur Entree pour fermer... (les composants pourront etre reinstalles depuis Parametres)")
        return code
    if args.selftest:
        return run_selftest()
    from .gui.app import main as gui_main

    gui_main(open_file=args.file if args.file and args.file.lower().endswith(".rls") else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
