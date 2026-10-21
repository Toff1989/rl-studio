"""Planification et installation des composants de calcul (runtime Python, PyTorch, MATLAB Engine).

Le runtime est un Python separe de l'interface : l'interface est gelee par PyInstaller alors que
la bonne build de PyTorch (CUDA) et le matlabengine dependent du materiel de l'utilisateur.
Bibliotheque standard uniquement (execute par l'installateur et par la fenetre Parametres).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from . import hardware
from .paths import FROZEN, RUNTIME_DIR, VENDOR_DIR

BUNDLED_PYTHON = "3.12"  # version du Python embarqué dans l'installateur
# index PyTorch par ordre decroissant : (suffixe, version CUDA correspondante)
TORCH_INDEXES = [("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6))]
TORCH_URL = "https://download.pytorch.org/whl/{}"
BASE_REQUIREMENTS = ["numpy", "pyyaml", "gymnasium", "stable-baselines3"]
MIN_CUDA_BLACKWELL = (12, 8)  # RTX 50xx (compute capability >= 12.0)


def _ver(text):
    try:
        return tuple(int(x) for x in str(text).split(".")[:2])
    except ValueError:
        return None


# -- planification ------------------------------------------------------------------------
def plan_torch(hw: dict, gpu_enabled=True) -> dict:
    nvidia = [g for g in hw.get("gpus") or [] if g.get("vendor") == "NVIDIA"]
    if not gpu_enabled:
        return {"variant": "cpu", "candidates": [], "reason": "GPU désactivé dans les paramètres."}
    if not nvidia:
        return {"variant": "cpu", "candidates": [], "reason": "Aucune carte NVIDIA détectée : PyTorch CPU."}
    driver_cuda = _ver(hw.get("cuda_driver"))
    if not driver_cuda:
        return {"variant": "cpu", "candidates": [],
                "reason": "Version CUDA du pilote inconnue : PyTorch CPU (mettre à jour le pilote NVIDIA)."}
    cc = _ver(nvidia[0].get("compute_cap"))
    minimum = MIN_CUDA_BLACKWELL if cc and cc >= (12, 0) else (11, 8)
    candidates = [name for name, v in TORCH_INDEXES if minimum <= v <= driver_cuda]
    if not candidates:
        return {"variant": "cpu", "candidates": [],
                "reason": f"{nvidia[0]['name']} demande CUDA >= {minimum[0]}.{minimum[1]} mais le pilote "
                          f"n'offre que CUDA {driver_cuda[0]}.{driver_cuda[1]} : mettre à jour le pilote NVIDIA. "
                          "PyTorch CPU en attendant."}
    return {"variant": candidates[0], "candidates": candidates,
            "reason": f"{nvidia[0]['name']} + pilote CUDA {driver_cuda[0]}.{driver_cuda[1]} : PyTorch {candidates[0]}."}


def plan_matlab(hw: dict, enabled=True, root="") -> dict:
    if not enabled:
        return {"enabled": False, "reason": "MATLAB désactivé dans les paramètres."}
    installs = list(hw.get("matlab") or [])
    if root:
        info = next((m for m in installs if Path(m["root"]) == Path(root)), None) or hardware.inspect_matlab(root)
        installs = [info] if info else []
    usable = [m for m in installs if m.get("complete") and m.get("engine_series")]
    if not usable:
        why = "Aucune installation MATLAB complète détectée."
        if any(not m.get("complete") for m in installs):
            why = "Installation(s) MATLAB incomplète(s) : réinstaller MATLAB (avec Simulink)."
        return {"enabled": False, "reason": why}
    m = usable[0]
    py_ok = not m.get("engine_python_versions") or BUNDLED_PYTHON in m["engine_python_versions"]
    plan = {"enabled": py_ok, "root": m["root"], "release": m.get("release"), "simulink": m.get("simulink"),
            "engine_spec": f"matlabengine=={m['engine_series']}.*", "python_ok": py_ok}
    if py_ok:
        plan["reason"] = f"MATLAB {m.get('release')} : {plan['engine_spec']}."
    else:
        plan["reason"] = (f"MATLAB {m.get('release')} ne supporte pas Python {BUNDLED_PYTHON} "
                          f"(versions : {', '.join(m['engine_python_versions'])}) : MATLAB Engine non installe.")
    if not m.get("simulink"):
        plan["reason"] += " Attention : Simulink est absent de cette installation."
    return plan


def make_plan(hw: dict, gpu_enabled=True, matlab_enabled=True, matlab_root="") -> dict:
    return {"python": BUNDLED_PYTHON, "torch": plan_torch(hw, gpu_enabled),
            "matlab": plan_matlab(hw, matlab_enabled, matlab_root)}


def plan_summary(plan: dict) -> str:
    return "\n".join([f"Python embarqué : {plan['python']}",
                      f"PyTorch : {plan['torch']['variant']} - {plan['torch']['reason']}",
                      f"MATLAB : {plan['matlab'].get('reason', '-')}"])


# -- runtime ------------------------------------------------------------------------------
def runtime_info_path(runtime_dir=None) -> Path:
    return Path(runtime_dir or RUNTIME_DIR) / "runtime.json"


def runtime_python(runtime_dir=None) -> Path | None:
    """Python du runtime installe, ou None."""
    try:
        data = json.loads(runtime_info_path(runtime_dir).read_text(encoding="utf-8"))
        p = Path(data["python"])
        return p if p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


# Le Python embarqué active `site` (necessaire a pip) et ajoute alors le site-packages UTILISATEUR :
# pip y verrait les paquets de l'utilisateur (et pourrait les desinstaller !). On l'ecarte de trois facons.
SITECUSTOMIZE = """import os, site, sys
site.ENABLE_USER_SITE = False
_user = os.path.normcase(site.getusersitepackages() or "?")
sys.path[:] = [p for p in sys.path if not os.path.normcase(p).startswith(_user)]
# Un Python embarqué (fichier ._pth) tourne en mode isolé et ignore PYTHONPATH : le code de
# l'application est donc ajoute ici, depuis RLSTUDIO_LIB.
_lib = os.environ.get("RLSTUDIO_LIB")
if _lib and _lib not in sys.path:
    sys.path.insert(0, _lib)
"""


def isolated_env(extra=None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP") and not k.startswith("PIP_")}
    env["PYTHONNOUSERSITE"] = "1"
    # Site utilisateur desactive : pip ne doit jamais basculer en « --user » (erreur « User site-packages are
    # disabled »), ni lire une configuration pip utilisateur/globale qui l'y forcerait.
    env["PIP_USER"] = "0"
    env["PIP_CONFIG_FILE"] = os.devnull
    env.update(extra or {})
    return env


def _stream(cmd, log, env=None, cwd=None) -> int:
    env = isolated_env() if env is None else env
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env, cwd=cwd, creationflags=hardware.NO_WINDOW)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    return proc.wait()


def create_runtime(runtime_dir: Path, log, vendor_dir=None) -> tuple[Path, str]:
    """Créé le Python du runtime : Python embarqué (installateur) ou venv (poste de developpement)."""
    vendor = Path(vendor_dir or VENDOR_DIR)
    embed = next(iter(sorted(vendor.glob("python-*-embed-amd64.zip"))), None)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if embed:
        target = runtime_dir / "python"
        if not (target / "python.exe").exists():
            log(f"Extraction de {embed.name}")
            with zipfile.ZipFile(embed) as z:
                z.extractall(target)
        for pth in target.glob("python*._pth"):  # autorise site-packages et pip
            text = pth.read_text(encoding="utf-8").replace("#import site", "import site")
            if "Lib\\site-packages" not in text:
                text = text.rstrip() + "\nLib\\site-packages\n"
            pth.write_text(text, encoding="utf-8")
        (target / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)  # sinon pip croit devoir installer en --user
        (target / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
        py = target / "python.exe"
        if _stream([py, "-m", "pip", "--version"], lambda _l: None) != 0:
            getpip = vendor / "get-pip.py"
            if not getpip.exists():
                raise RuntimeError(f"get-pip.py introuvable dans {vendor}")
            if _stream([py, getpip, "--no-warn-script-location"], log) != 0:
                raise RuntimeError("Installation de pip échouée.")
        return py, "embed"
    if FROZEN:
        raise RuntimeError(f"Python embarqué introuvable dans {vendor} : réinstaller l'application.")
    venv = runtime_dir / "venv"
    if not (venv / "Scripts" / "python.exe").exists() and not (venv / "bin" / "python").exists():
        if _stream([sys.executable, "-m", "venv", venv], log) != 0:
            raise RuntimeError("Création du venv échouée.")
    return (venv / "Scripts" / "python.exe" if (venv / "Scripts").exists() else venv / "bin" / "python"), "venv"


def provision(plan: dict, runtime_dir=None, log=print, skip=(), dry_run=False, vendor_dir=None) -> dict:
    """Installe (ou repare) le runtime selon le plan. Renvoie le contenu de runtime.json."""
    runtime_dir = Path(runtime_dir or RUNTIME_DIR)
    skip = set(skip)
    steps = []

    def pip(py, args):
        return [py, "-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check", *args]

    if dry_run:
        py = runtime_dir / "python" / "python.exe"
        kind = "embed"
    else:
        py, kind = create_runtime(runtime_dir, log, vendor_dir)

    # PyTorch d'abord : stable-baselines3 en depend et tirerait sinon la version CPU de PyPI, apres quoi
    # pip jugerait la variante CUDA « deja satisfaite ». `--upgrade` : 2.x.y+cu130 > 2.x.y (CPU).
    torch = plan.get("torch", {})
    if "torch" not in skip:
        for idx in torch.get("candidates", []):
            steps.append((f"PyTorch {idx}", pip(py, ["--upgrade", "torch", "--index-url", TORCH_URL.format(idx)])))
        steps.append(("PyTorch CPU", pip(py, ["--upgrade", "torch"])))  # derniere solution (PyPI)

    req_file = Path(vendor_dir or VENDOR_DIR) / "requirements-runtime.txt"
    base = ["-r", req_file] if req_file.exists() else BASE_REQUIREMENTS
    if "base" not in skip:
        steps.append(("Bibliothèques de base", pip(py, ["--upgrade", *base])))

    ml = plan.get("matlab", {})
    if "matlab" not in skip and ml.get("enabled") and ml.get("engine_spec"):
        steps.append(("MATLAB Engine", pip(py, [ml["engine_spec"]])))

    result = {"python": str(py), "kind": kind, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": plan,
              "steps": [], "errors": []}
    torch_done = "torch" in skip
    for title, cmd in steps:
        if title.startswith("PyTorch") and torch_done:
            continue
        log(f"== {title}")
        if dry_run:
            log("$ " + " ".join(str(c) for c in cmd))
            result["steps"].append({"step": title, "ok": None})
            continue
        ok = _stream(cmd, log) == 0
        result["steps"].append({"step": title, "ok": ok})
        if title.startswith("PyTorch"):
            torch_done = torch_done or ok
            if ok:
                result["torch_variant"] = title.split()[-1]
        elif not ok:
            result["errors"].append(f"{title} : échec")
    if not torch_done and "torch" not in skip and not dry_run:
        result["errors"].append("PyTorch : aucune variante n'a pu être installée")
    if not dry_run:
        runtime_info_path(runtime_dir).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
