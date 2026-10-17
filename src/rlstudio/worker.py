"""Processus de calcul : `python -m rlstudio.worker <commande>`.

Execute dans le runtime (PyTorch, Stable-Baselines3, MATLAB Engine), separement de l'interface.
Protocole : une ligne JSON par evenement sur stdout ; la commande `stop` arrive sur stdin.

  train        --project DIR [--settings FILE]   entraine un projet
  inspect      --model FILE                      blocs d'un modele Simulink
  matlab-test                                    demarre MATLAB et verifie Simulink
  env-info                                       versions installees (Python, PyTorch, CUDA...)
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
        should_stop=stop.is_set,
        log=lambda m: emit(type="log", text=m),
    )
    emit(type="done", run_dir=str(run_dir), stopped=stop.is_set())


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
    args = ap.parse_args(argv)
    handlers = {"train": cmd_train, "inspect": cmd_inspect, "matlab-test": cmd_matlab_test, "env-info": cmd_env_info}
    try:
        handlers[args.cmd](args)
    except Exception as e:
        emit(type="error", text=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
