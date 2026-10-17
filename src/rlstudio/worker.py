"""Processus de calcul : `python -m rlstudio.worker <commande>`.

Execute dans le runtime (PyTorch, Stable-Baselines3, MATLAB Engine), separement de l'interface.
Protocole : une ligne JSON par evenement sur stdout ; la commande `stop` arrive sur stdin.

  train        --project DIR [--settings FILE]   entraine un projet
  inspect      --model FILE                      blocs d'un modele Simulink
  matlab-test                                    demarre MATLAB et verifie Simulink
  env-info                                       versions installees (Python, PyTorch, CUDA...)
  evaluate     --project DIR --run NOM [--policy agent|distill:ID]   rejoue des episodes en direct
  distill      --project DIR --run NOM [--options JSON]   extrait un modele mathematique d'un run
  export-simulink --project DIR --run NOM --candidate ID --out FILE   controleur extrait -> modele Simulink
"""
import argparse
import json
import sys
import threading
import time
import traceback


def emit(**event):
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _listen_for_stop(flag: threading.Event):
    for line in sys.stdin:
        if line.strip() == "stop":
            flag.set()
            return


def resolve_device(requested: str, gpu_enabled: bool, log) -> str:
    if not gpu_enabled:
        return "cpu"
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if requested == "auto":
        return "cuda" if cuda else "cpu"
    if requested == "cuda" and not cuda:
        log("CUDA indisponible : entrainement sur CPU.")
        return "cpu"
    return requested


def cmd_train(args):
    import numpy as np

    from .project import Project
    from .settings import Settings
    from .training import train

    settings = Settings(args.settings) if args.settings else Settings()
    project = Project(args.project)
    algo = project.cfg["algo"]
    algo["device"] = resolve_device(algo.get("device", "cpu"), bool(settings.get("gpu_enabled")),
                                    lambda m: emit(type="log", text=m))
    stop = threading.Event()
    threading.Thread(target=_listen_for_stop, args=(stop,), daemon=True).start()
    last = [0.0]

    def on_state(s):
        now = time.monotonic()
        if now - last[0] < 0.05:  # au plus ~20 mises a jour par seconde
            return
        last[0] = now
        emit(type="state", timesteps=int(s["timesteps"]), obs=np.asarray(s["obs"], dtype=float).round(5).tolist(),
             action=np.atleast_1d(s["action"]).astype(float).round(5).tolist(), reward=float(s["reward"]))

    run_dir = train(
        project,
        on_episode=lambda n, r, length, steps: emit(type="episode", n=n, reward=r, length=length, steps=steps),
        on_state=on_state,
        on_baseline=lambda mean, label: emit(type="baseline", mean=mean, label=label),
        on_meta=lambda m: emit(type="meta", **m),
        should_stop=stop.is_set,
        log=lambda m: emit(type="log", text=m),
    )
    emit(type="done", run_dir=str(run_dir), stopped=stop.is_set())


def _run_dir(project, name):
    d = project.runs_dir / name
    if not d.is_dir():
        raise FileNotFoundError(f"run introuvable : {name}")
    return d


def cmd_distill(args):
    from .distill import DistillOptions, run_distillation
    from .distill.collect import Stopped
    from .project import Project

    project = Project(args.project)
    options = DistillOptions.from_dict(json.loads(args.options) if args.options else None)
    stop = threading.Event()
    threading.Thread(target=_listen_for_stop, args=(stop,), daemon=True).start()
    try:
        report = run_distillation(project, _run_dir(project, args.run), options, log=lambda m: emit(type="log", text=m),
                                  progress=lambda stage, frac: emit(type="progress", stage=stage, fraction=round(frac, 4)),
                                  should_stop=stop.is_set)
    except Stopped:
        emit(type="done", stopped=True)
        return
    emit(type="done", stopped=False, recommended=report["recommended"])


def cmd_export_simulink(args):
    from .distill.export import build_simulink_controller, load_candidate
    from .project import Project

    project = Project(args.project)
    run_dir = _run_dir(project, args.run)
    out = build_simulink_controller(load_candidate(run_dir, args.candidate), args.out, log=lambda m: emit(type="log", text=m))
    emit(type="result", path=str(out))


def cmd_evaluate(args):
    import numpy as np

    from .evaluation import run_evaluation
    from .project import Project

    project = Project(args.project)
    stop = threading.Event()
    threading.Thread(target=_listen_for_stop, args=(stop,), daemon=True).start()
    last = [0.0]

    def on_step(s):
        now = time.monotonic()
        if args.delay <= 0 and now - last[0] < 0.03:  # sans ralentissement : ~30 mises a jour par seconde
            return
        last[0] = now
        emit(type="step", episode=s["episode"], step=s["step"], obs=np.asarray(s["obs"]).round(5).tolist(),
             action=np.asarray(s["action"]).round(5).tolist(), reward=round(s["reward"], 5), total=round(s["total"], 5))

    summary = run_evaluation(project, _run_dir(project, args.run), args.policy, args.episodes, args.seed, args.delay,
                             on_meta=lambda m: emit(type="meta", **m), on_step=on_step,
                             on_episode=lambda e: emit(type="episode", **e), should_stop=stop.is_set,
                             log=lambda m: emit(type="log", text=m))
    emit(type="done", stopped=stop.is_set(), **summary)


def cmd_inspect(args):
    from .envs.simulink import MatlabSession

    session = MatlabSession()
    try:
        emit(type="result", blocks=session.inspect(args.model))
    finally:
        session.close()


def cmd_matlab_test(_args):
    import matlab.engine

    t = time.time()
    eng = matlab.engine.start_matlab()
    try:
        version = eng.version()
        simulink = bool(eng.eval("license('test','Simulink')", nargout=1))
        emit(type="result", version=version, simulink=simulink, startup_s=round(time.time() - t, 1))
    finally:
        eng.quit()


def cmd_env_info(_args):
    info = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("numpy", "gymnasium", "stable_baselines3", "yaml"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        info.update(torch=None, cuda_available=False, cuda_version=None, gpu=None, torch_error=str(e)[:200])
    try:
        from importlib.metadata import version

        info["matlabengine"] = version("matlabengine")
    except Exception:
        info["matlabengine"] = None
    emit(type="result", **info)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="rlstudio.worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--project", required=True)
    p.add_argument("--settings")
    p = sub.add_parser("inspect")
    p.add_argument("--model", required=True)
    sub.add_parser("matlab-test")
    sub.add_parser("env-info")
    p = sub.add_parser("distill")
    p.add_argument("--project", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--options")
    p = sub.add_parser("evaluate")
    p.add_argument("--project", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--policy", default="agent")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--delay", type=float, default=0.0, help="pause par pas (s), pour suivre l'episode a l'oeil")
    p = sub.add_parser("export-simulink")
    p.add_argument("--project", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    handlers = {"train": cmd_train, "inspect": cmd_inspect, "matlab-test": cmd_matlab_test, "env-info": cmd_env_info,
                "distill": cmd_distill, "evaluate": cmd_evaluate, "export-simulink": cmd_export_simulink}
    try:
        handlers[args.cmd](args)
    except Exception as e:
        emit(type="error", text=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
