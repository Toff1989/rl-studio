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


@pytest.mark.runtime
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


@pytest.mark.runtime
def test_distilled_policy_can_be_replayed(trained):
    p, run_dir = trained
    run_distillation(p, run_dir, DistillOptions(methods=("linear",), episodes=2, dagger_iters=0, eval_episodes=1, max_steps=30), log=lambda m: None)
    ids = [i for i, _ in policy_choices(run_dir)]
    assert ids == ["agent", "distill:linear"]
    s = run_evaluation(p, run_dir, "distill:linear", 1, 5, log=lambda m: None)
    assert s["episodes"] == 1


@pytest.mark.runtime
def test_worker_evaluate_events(trained):
    p, run_dir = trained
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "evaluate", "--project", str(p.path), "--run", run_dir.name,
                        "--episodes", "2"], capture_output=True, text=True, timeout=300, encoding="utf-8")
    ev = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    kinds = [e["type"] for e in ev]
    assert r.returncode == 0 and kinds[0] == "meta" and "step" in kinds and kinds.count("episode") == 2 and kinds[-1] == "done"


@pytest.mark.runtime
@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
def test_series_plot_append_is_cheap_and_coalesced():
    """L'interface ne doit pas ralentir quand les points s'accumulent : ajout O(1), dessins regroupes."""
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.live_plots import SeriesPlot

    app = QApplication.instance() or QApplication([])
    pl = SeriesPlot("Observations")
    pl.set_names(["a", "b", "c", "d"])
    draws = []
    real = pl._redraw
    pl._redraw = lambda: (draws.append(1), real())[1]
    pl._timer.timeout.disconnect()
    pl._timer.timeout.connect(pl._redraw)
    t0 = time.perf_counter()
    for i in range(20000):
        pl.append(i, [i, 2.0 * i, -i, 0.5 * i])
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"20000 ajouts en {elapsed:.2f} s : l'ajout d'un point n'est plus O(1)"
    assert not draws                       # rien n'est dessine tant que la boucle d'evenements ne tourne pas
    assert len(pl.t) == 20000 and pl.t[-1] == 19999
    app.processEvents()
    import PySide6.QtTest as qt

    qt.QTest.qWait(120)
    assert 1 <= len(draws) <= 3            # un seul dessin pour tous les points recus
    assert np.isclose(pl.curves[1].getData()[1][-1], 2.0 * 19999)
    pl.reset()
    assert len(pl.t) == 0
    pl.append(0, [1.0, 2.0, 3.0, 4.0, 5.0])  # signal non declare apres une remise a zero
    assert len(pl.curves) == 5


@pytest.mark.gui
def test_decimate_keeps_peaks_and_bounds_size():
    from rlstudio.gui.live_plots import MAX_DRAWN, decimate

    t = np.arange(50001, dtype=float)
    y = np.sin(t / 300.0)[:, None]
    y[12345, 0] = 5.0                       # un pic isole ne doit pas disparaitre
    t2, y2 = decimate(t, y)
    assert len(t2) <= MAX_DRAWN + 2 and len(t2) == len(y2)
    assert y2.max() == 5.0 and t2[-1] == t[-1]
    small_t, small_y = decimate(t[:100], y[:100])
    assert small_t is t[:100] or np.array_equal(small_t, t[:100])
