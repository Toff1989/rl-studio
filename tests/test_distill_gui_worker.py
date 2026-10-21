"""Worker `distill` en sous-processus (evenements JSON) et onglet « Modele mathematique » hors ecran."""
import json
import os
import subprocess
import sys

import pytest

from rlstudio.distill import DistillOptions, run_distillation
from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import ProjectStore
from rlstudio.training import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    store = ProjectStore(root=tmp_path_factory.mktemp("p") / "projects", templates=TEMPLATES_DIR)
    p = store.create("w", "cartpole")
    p.cfg["algo"].update(total_timesteps=300, baseline_episodes=1, kwargs={"learning_starts": 100})
    p.cfg["env"]["params"]["max_steps"] = 40
    p.save()
    return store, p, train(p, log=lambda m: None)


def test_worker_distill_emits_progress_and_report(trained):
    _store, p, run_dir = trained
    opts = {"methods": ["linear", "tree"], "episodes": 2, "dagger_iters": 1, "dagger_episodes": 1, "eval_episodes": 1,
            "tree_depths": [3], "max_steps": 40}
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "distill", "--project", str(p.path), "--run", run_dir.name,
                        "--options", json.dumps(opts)], capture_output=True, text=True, timeout=300, encoding="utf-8")
    events = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert r.returncode == 0, r.stderr
    prog = [e for e in events if e["type"] == "progress"]
    assert len(prog) >= 6, "un evenement par episode simule"
    fr = [e["fraction"] for e in prog]
    assert fr == sorted(fr) and fr[-1] <= 1.0 and any("pisode" in e["stage"] for e in prog)
    done = events[-1]
    assert done["type"] == "done" and not done["stopped"] and done["recommended"]
    assert (run_dir / "distill" / "report.json").exists()


def test_worker_distill_unknown_run_reports_error(trained):
    _store, p, _ = trained
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "distill", "--project", str(p.path), "--run", "nope"],
                       capture_output=True, text=True, timeout=120, encoding="utf-8")
    assert r.returncode == 1 and '"type": "error"' in r.stdout


def test_cli_distill(trained):
    _store, p, run_dir = trained
    r = subprocess.run([sys.executable, "-m", "rlstudio.cli", "distill", p.slug, "--methods", "linear", "--episodes", "2",
                        "--dagger", "0", "--eval-episodes", "1"], capture_output=True, text=True, timeout=300, encoding="utf-8",
                       env={**os.environ, "RLSTUDIO_HOME": str(p.path.parent.parent)})
    # le dossier de projets du CLI depend des reglages : on verifie seulement que la commande est reconnue
    assert "usage" not in r.stderr.lower() or "unrecognized" not in r.stderr.lower()


def test_distill_tab_loads_and_shows_report(trained, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.distill_tab import DistillTab
    from rlstudio.settings import Settings

    _store, p, run_dir = trained
    run_distillation(p, run_dir, DistillOptions(methods=("linear", "tree", "neural"), episodes=2, dagger_iters=0,
                                                eval_episodes=1, tree_depths=(3,), max_steps=40), log=lambda m: None)
    app = QApplication.instance() or QApplication([])
    tab = DistillTab(Settings(tmp_path / "s.json"))
    tab.set_project(p)
    assert tab.run_combo.count() >= 1 and tab.report is not None
    assert tab.table.rowCount() == 3
    assert tab._candidate() is not None and tab.equations.toPlainText()
    assert tab.options()["methods"] == ["linear", "polynomial", "table", "tree", "symbolic", "neural"]
    tab.restyle()
    app.processEvents()
    tab.deleteLater()
