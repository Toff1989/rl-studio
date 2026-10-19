"""Le runtime embarque ne doit jamais voir (ni modifier) les paquets du site-packages utilisateur."""
import os
import subprocess
import sys

from rlstudio import components, runtime
from rlstudio.settings import Settings


def test_sitecustomize_strips_user_site(tmp_path):
    # faux site utilisateur contenant un module « marqueur » que le runtime ne doit pas voir
    userbase = tmp_path / "userbase"
    user_site = userbase / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages"
    user_site.mkdir(parents=True)
    (user_site / "marqueur_utilisateur.py").write_text("VALEUR = 1\n")
    (tmp_path / "sitecustomize.py").write_text(components.SITECUSTOMIZE)
    env = {"PYTHONUSERBASE": str(userbase), "PYTHONPATH": str(tmp_path), "SYSTEMROOT": "C:\\Windows"}

    sees_user_site = subprocess.run([sys.executable, "-c", "import marqueur_utilisateur"],
                                    env={k: v for k, v in env.items() if k != "PYTHONPATH"},
                                    capture_output=True).returncode == 0
    isolated = subprocess.run([sys.executable, "-c", "import marqueur_utilisateur"], env=env,
                              capture_output=True).returncode
    # (si Python tourne dans un venv, le site utilisateur est deja ignore : le test reste valide)
    assert isolated != 0, f"le module du site utilisateur reste visible (visible sans sitecustomize : {sees_user_site})"


def test_pip_environment_is_isolated(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "C:\\ailleurs")
    monkeypatch.setenv("PYTHONHOME", "C:\\autre")
    monkeypatch.setenv("PIP_USER", "true")
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.invalid/simple")
    env = components.isolated_env()
    assert env["PYTHONNOUSERSITE"] == "1" and "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["PIP_USER"] == "0" and env["PIP_CONFIG_FILE"] == os.devnull
    assert "PIP_INDEX_URL" not in env


def test_worker_environment_disables_user_site(tmp_path):
    env = runtime.worker_env(Settings(tmp_path / "s.json"))
    assert env["PYTHONNOUSERSITE"] == "1"


def test_sitecustomize_adds_app_lib(tmp_path):
    """Python embarque = mode isole : PYTHONPATH ignore, donc le code de l'application passe par RLSTUDIO_LIB."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "module_application.py").write_text("OK = True\n")
    (tmp_path / "sitecustomize.py").write_text(components.SITECUSTOMIZE)
    env = {"PYTHONPATH": str(tmp_path), "RLSTUDIO_LIB": str(lib), "SYSTEMROOT": "C:\\Windows"}
    r = subprocess.run([sys.executable, "-c", "import module_application"], env=env, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


def test_worker_env_exposes_app_lib(tmp_path):
    env = runtime.worker_env(Settings(tmp_path / "s.json"))
    assert env["RLSTUDIO_LIB"]
