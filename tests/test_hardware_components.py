import json
import subprocess
import sys

import pytest

from rlstudio import components, hardware


def hw(gpus=None, cuda=None, matlab=None):
    return {"gpus": gpus or [], "cuda_driver": cuda, "matlab": matlab or []}


def nvidia(name="NVIDIA GeForce RTX 5080", cc="12.0"):
    return {"vendor": "NVIDIA", "name": name, "driver": "617.14", "memory_gb": 16, "compute_cap": cc}


# -- nvidia-smi ---------------------------------------------------------------------------
def test_parse_nvidia_smi_new_and_old_formats():
    csv = "NVIDIA GeForce RTX 5080, 617.14, 16303, 12.0\n"
    gpus, cuda = hardware.parse_nvidia_smi(csv, "| NVIDIA-SMI 617.14  KMD Version: 617.14  CUDA UMD Version: 13.4 |")
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 5080" and gpus[0]["compute_cap"] == "12.0"
    assert gpus[0]["memory_gb"] == 15.9 and cuda == "13.4"
    _, old = hardware.parse_nvidia_smi("", "| NVIDIA-SMI 535.1  Driver Version: 535.1  CUDA Version: 12.2 |")
    assert old == "12.2"


# -- detection MATLAB ---------------------------------------------------------------------
def fake_matlab(tmp_path, release="R2026b", version="26.2.0.1", py=("3.12", "3.13"), complete=True, simulink=True):
    root = tmp_path / release
    (root / "bin").mkdir(parents=True)
    (root / "VersionInfo.xml").write_text(f"<x><version>{version}</version><release>{release}</release></x>")
    if complete:
        (root / "bin" / "matlab.exe").write_text("")
        eng = root / "extern" / "engines" / "python"
        eng.mkdir(parents=True)
        listing = ", ".join(f"'{v}'" for v in py)
        (eng / "setup.py").write_text(f"_supported_versions = [{listing}]\n")
    if simulink:
        (root / "toolbox" / "simulink").mkdir(parents=True)
    return root


def test_inspect_matlab(tmp_path):
    info = hardware.inspect_matlab(fake_matlab(tmp_path))
    assert info["release"] == "R2026b" and info["engine_series"] == "26.2"
    assert info["engine_python_versions"] == ["3.12", "3.13"] and info["complete"] and info["simulink"]
    partial = hardware.inspect_matlab(fake_matlab(tmp_path, "R2025a", "25.1.0.1", complete=False, simulink=False))
    assert not partial["complete"]
    assert hardware.inspect_matlab(tmp_path / "nexiste-pas") is None


# -- planificateur ------------------------------------------------------------------------
@pytest.mark.parametrize("gpus,cuda,expected", [
    ([nvidia()], "13.4", "cu130"),                 # RTX 5080, pilote recent
    ([nvidia()], "12.8", "cu128"),
    ([nvidia()], "12.6", "cpu"),                   # Blackwell : pilote trop ancien
    ([nvidia("RTX 3060", "8.6")], "12.6", "cu126"),
    ([nvidia("RTX 3060", "8.6")], "11.8", "cpu"),  # aucun index <= CUDA 11.8 propose
    ([nvidia()], None, "cpu"),                      # CUDA inconnu
    ([], None, "cpu"),
    ([{"vendor": "autre", "name": "Intel Arc"}], None, "cpu"),
])
def test_torch_variant(gpus, cuda, expected):
    assert components.plan_torch(hw(gpus, cuda))["variant"] == expected


def test_torch_candidates_fall_back_in_order():
    plan = components.plan_torch(hw([nvidia()], "13.4"))
    assert plan["candidates"] == ["cu130", "cu128"]  # cu126 trop ancien pour Blackwell


def test_gpu_disabled_forces_cpu():
    assert components.plan_torch(hw([nvidia()], "13.4"), gpu_enabled=False)["variant"] == "cpu"


def test_matlab_plan(tmp_path):
    ok = hardware.inspect_matlab(fake_matlab(tmp_path))
    plan = components.plan_matlab(hw(matlab=[ok]))
    assert plan["enabled"] and plan["engine_spec"] == "matlabengine==26.2.*"
    old = hardware.inspect_matlab(fake_matlab(tmp_path, "R2023a", "23.1.0.1", py=("3.8", "3.9", "3.10")))
    assert not components.plan_matlab(hw(matlab=[old]))["enabled"]  # Python embarque 3.12 non supporte
    partial = hardware.inspect_matlab(fake_matlab(tmp_path, "R2025a", "25.1.0.1", complete=False))
    assert "incomplèt" in components.plan_matlab(hw(matlab=[partial]))["reason"].lower()
    assert not components.plan_matlab(hw(matlab=[ok]), enabled=False)["enabled"]
    assert not components.plan_matlab(hw())["enabled"]


def test_matlab_root_override_picks_requested_install(tmp_path):
    new = hardware.inspect_matlab(fake_matlab(tmp_path))
    older = hardware.inspect_matlab(fake_matlab(tmp_path, "R2025b", "25.2.0.1", py=("3.12",)))
    plan = components.plan_matlab(hw(matlab=[new, older]), root=older["root"])
    assert plan["engine_spec"] == "matlabengine==25.2.*"


# -- provisioning -------------------------------------------------------------------------
def test_provision_dry_run_lists_pip_commands(tmp_path):
    plan = components.make_plan(hw([nvidia()], "13.4", [hardware.inspect_matlab(fake_matlab(tmp_path))]))
    lines = []
    result = components.provision(plan, tmp_path / "rt", lines.append, dry_run=True)
    text = "\n".join(lines)
    assert "https://download.pytorch.org/whl/cu130" in text and "matlabengine==26.2.*" in text
    assert not (tmp_path / "rt").exists()  # dry-run : rien n'est cree
    assert result["errors"] == []


def test_provision_skip_heavy_components(tmp_path):
    plan = components.make_plan(hw())
    lines = []
    components.provision(plan, tmp_path / "rt", lines.append, skip=("base", "torch", "matlab"), dry_run=True)
    assert not any("pip" in line for line in lines)


# -- worker en sous-processus -------------------------------------------------------------
@pytest.mark.runtime
def test_worker_train_emits_json_events(tmp_path):
    from rlstudio.paths import TEMPLATES_DIR
    from rlstudio.project import ProjectStore

    p = ProjectStore(root=tmp_path / "p", templates=TEMPLATES_DIR).create("t", "python_blank")
    p.cfg["algo"].update(total_timesteps=300, baseline_episodes=1)
    p.save()
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "train", "--project", str(p.path),
                        "--settings", str(tmp_path / "s.json")], capture_output=True, text=True, timeout=180)
    events = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    kinds = {e["type"] for e in events}
    assert r.returncode == 0, r.stderr
    assert {"log", "baseline", "episode", "done"} <= kinds


def test_worker_env_info():
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "env-info"], capture_output=True, text=True, timeout=120)
    info = json.loads(r.stdout.splitlines()[-1])
    assert info["type"] == "result" and info["python"] and "cuda_available" in info


def test_torch_is_installed_before_base_requirements(tmp_path):
    """stable-baselines3 tire PyTorch CPU : la variante CUDA doit etre posee avant, avec --upgrade."""
    plan = components.make_plan(hw([nvidia()], "13.4"))
    lines = []
    components.provision(plan, tmp_path / "rt", lines.append, dry_run=True)
    titles = [line for line in lines if line.startswith("== ")]
    assert titles.index("== PyTorch cu130") < titles.index("== Bibliothèques de base")
    cmd = next(line for line in lines if "whl/cu130" in line)
    assert "--upgrade" in cmd
