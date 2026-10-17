"""Evaluation en direct (worker + onglet) et graphiques temporels."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from rlstudio.distill import DistillOptions, run_distillation
from rlstudio.evaluation import policy_choices, run_evaluation
from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import ProjectStore
from rlstudio.training import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    store = ProjectStore(root=tmp_path_factory.mktemp("p") / "projects", templates=TEMPLATES_DIR)
    p = store.create("e", "cartpole")
    p.cfg["algo"].update(total_timesteps=300, baseline_episodes=0, kwargs={"learning_starts": 100})
    p.cfg["env"]["params"]["max_steps"] = 30
    p.save()
    metas = []
    run_dir = train(p, log=lambda m: None, on_meta=metas.append)
    assert metas and metas[0]["obs_names"] == ["x", "x_dot", "theta", "theta_dot"]
    return p, run_dir


def test_run_evaluation_streams_steps_and_episodes(trained):
    p, run_dir = trained
    steps, eps, metas = [], [], []
    s = run_evaluation(p, run_dir, "agent", 2, 1000, on_meta=metas.append, on_step=steps.append, on_episode=eps.append, log=lambda m: None)
    assert s["episodes"] == 2 and len(eps) == 2 and steps
    assert metas[0]["action_names"] == ["force"] and len(steps[0]["obs"]) == 4 and steps[0]["step"] == 1
    # arret demande : plus d'episode termine
    stop = {"n": 0}
    s = run_evaluation(p, run_dir, "agent", 5, 1, should_stop=lambda: stop.__setitem__("n", stop["n"] + 1) or stop["n"] > 10, log=lambda m: None)
    assert s["episodes"] < 5


def test_distilled_policy_can_be_replayed(trained):
    p, run_dir = trained
    run_distillation(p, run_dir, DistillOptions(methods=("linear",), episodes=2, dagger_iters=0, eval_episodes=1, max_steps=30), log=lambda m: None)
    ids = [i for i, _ in policy_choices(run_dir)]
    assert ids == ["agent", "distill:linear"]
    s = run_evaluation(p, run_dir, "distill:linear", 1, 5, log=lambda m: None)
    assert s["episodes"] == 1


def test_worker_evaluate_events(trained):
    p, run_dir = trained
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "evaluate", "--project", str(p.path), "--run", run_dir.name,
                        "--episodes", "2"], capture_output=True, text=True, timeout=300, encoding="utf-8")
    ev = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    kinds = [e["type"] for e in ev]
    assert r.returncode == 0 and kinds[0] == "meta" and "step" in kinds and kinds.count("episode") == 2 and kinds[-1] == "done"


def test_series_plot_and_eval_tab(trained, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.eval_tab import EvalTab
    from rlstudio.gui.live_plots import SeriesPlot
    from rlstudio.settings import Settings

    app = QApplication.instance() or QApplication([])
    pl = SeriesPlot("Observations")
    pl.set_names(["a", "b"])
    pl.append(0, [1.0, 10.0])
    pl.append(1, [2.0, 20.0])
    pl.set_normalized(True)
    assert len(pl.curves) == 2 and np.isclose(np.max(pl.curves[1].getData()[1]), 1.0)
    pl.append(2, [1, 2, 3])  # signal non declare : courbe supplementaire
    assert len(pl.curves) == 3
    p, _ = trained
    tab = EvalTab(Settings(tmp_path / "s.json"))
    tab.set_project(p)
    assert tab.run_combo.count() == 1 and tab.policy_combo.itemData(0) == "agent"
    tab._on_meta(["x", "v"], ["f"])
    tab._reset_views()
    tab._on_step({"episode": 1, "step": 1, "obs": np.array([0.0, 1.0]), "action": np.array([0.5]), "reward": 1.0, "total": 1.0})
    tab._on_episode({"n": 1, "reward": 3.0, "length": 5, "terminated": False})
    assert tab.k_mean.value.text() == "3.00"
    app.processEvents()


def test_series_plot_many_signals_distinct_colors_and_toggle():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.live_plots import SeriesPlot, palette

    app = QApplication.instance() or QApplication([])
    cols = palette(20)
    assert len(set(cols)) == 20  # une couleur distincte par signal
    pl = SeriesPlot("Observations")
    pl.set_names([f"s{i}" for i in range(14)])
    pl.append(0, list(range(14)))
    assert len(pl.curves) == 14 and len(pl.legend.chips) == 15  # 14 signaux + « tout masquer / afficher »
    pl.legend.chips[2].toggled.emit(False)  # masque la courbe s2
    assert not pl.curves[2].isVisible() and pl.curves[1].isVisible()
    pl.set_names([f"s{i}" for i in range(14)] + ["extra"])  # reconstruction : le masquage est conserve
    assert not pl.curves[2].isVisible()
    pl._set_all_visible(False)
    assert not any(c.isVisible() for c in pl.curves)
    pl._set_all_visible(True)
    assert all(c.isVisible() for c in pl.curves)
    app.processEvents()
