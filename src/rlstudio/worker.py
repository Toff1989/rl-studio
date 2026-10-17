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
  analyze-model --model FILE [--init FILE]     analyse un modele Simulink existant (conversion)
  convert-model --spec FILE --out DIR          cree un projet depuis une copie convertie du modele, puis le teste a vide
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

    last_progress = [0.0]

    def on_progress(stage, fraction):
        now = time.monotonic()
        if fraction < 1.0 and now - last_progress[0] < 0.25:  # au plus ~4 mises a jour par seconde
            return
        last_progress[0] = now
        emit(type="progress", stage=stage, fraction=round(float(fraction), 4))

    run_dir = train(
        project,
        on_episode=lambda n, r, length, steps: emit(type="episode", n=n, reward=r, length=length, steps=steps),
        on_state=on_state,
        on_baseline=lambda mean, label: emit(type="baseline", mean=mean, label=label),
        on_meta=lambda m: emit(type="meta", **m),
        on_progress=on_progress,
        on_run=lambda d: emit(type="run", run_dir=str(d)),
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


def cmd_analyze_model(args):
    from .convert import default_spec
    from .envs.simulink import MatlabSession

    session = MatlabSession()
    try:
        emit(type="log", text="Analyse du modele...")
        analysis = session.analyze(args.model, args.init or "")
        emit(type="result", analysis=analysis, proposal=default_spec(analysis))
    finally:
        session.close()


def _check_environment(project_dir, env_cfg, session, steps=3):
    """Test a vide : reset + quelques pas a l'action mediane ; renvoie un dict {ok, ...} sans lever d'exception."""
    import numpy as np

    from .envs.simulink import SimulinkEnv

    env = SimulinkEnv(project_dir, env_cfg, session=session)
    try:
        t0 = time.time()
        obs, _ = env.reset(seed=0)
        mid = (env.action_space.low + env.action_space.high) / 2.0
        reward, ms = 0.0, []
        for _ in range(steps):
            t = time.time()
            obs, r, term, trunc, _ = env.step(mid.astype(np.float32))
            ms.append((time.time() - t) * 1000)
            reward += r
            if term or trunc:
                break
        return {"ok": True, "obs": {n: round(float(v), 5) for n, v in zip(env.obs_names, obs)}, "reward": round(float(reward), 5),
                "ms_per_step": round(float(np.mean(ms)), 1), "episode_steps": env.n_steps, "first_reset_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def cmd_convert_model(args):
    import shutil
    from pathlib import Path

    from .convert import matlab_spec, validate_conversion, write_project
    from .envs.simulink import MatlabSession

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    errs = validate_conversion(spec)
    if errs:
        raise ValueError("Specification invalide :\n- " + "\n- ".join(errs))
    out_dir = Path(args.out)
    if out_dir.exists():
        raise FileExistsError(f"Le dossier existe deja : {out_dir}")
    out_dir.mkdir(parents=True)
    session = MatlabSession()
    try:
        try:
            model_name = Path(spec["model_path"]).stem
            emit(type="log", text="Conversion du modele (copie : l'original n'est pas modifie)...")
            result = session.convert(spec["model_path"], out_dir / f"{model_name}.slx", matlab_spec(spec), spec.get("init_path") or "")
            cfg = write_project(out_dir, spec, result)
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)  # pas de projet a moitie cree
            raise
        emit(type="log", text="Test a vide : un episode court sur le modele converti...")
        check = _check_environment(out_dir, cfg["env"], session)
        emit(type="result", slug=out_dir.name, project_dir=str(out_dir), warnings=result.get("warnings") or [], check=check)
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
    p = sub.add_parser("analyze-model")
    p.add_argument("--model", required=True)
    p.add_argument("--init")
    p = sub.add_parser("convert-model")
    p.add_argument("--spec", required=True, help="fichier JSON de la specification (voir convert.py)")
    p.add_argument("--out", required=True, help="dossier du nouveau projet (ne doit pas exister)")
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
                "analyze-model": cmd_analyze_model, "convert-model": cmd_convert_model,
                "distill": cmd_distill, "evaluate": cmd_evaluate, "export-simulink": cmd_export_simulink}
    try:
        handlers[args.cmd](args)
    except Exception as e:
        emit(type="error", text=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
