"""Évaluation : un modèle identifié est rejoué à côté du système (courbes de comparaison, écarts), avec ou sans agent entraîné."""
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import pytest

pytest.importorskip("gymnasium")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rlstudio.evaluation import MODEL_SEP, _ModelOverlay, compare_errors, identified_choices, policy_choices, run_evaluation  # noqa: E402
from rlstudio.identify import IdentifyOptions, SteadyOptions  # noqa: E402
from rlstudio.identify.pipeline import run_identification  # noqa: E402
from rlstudio.identify.steady import run_steady  # noqa: E402
from rlstudio.paths import TEMPLATES_DIR  # noqa: E402
from rlstudio.project import ProjectStore  # noqa: E402


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Projet « Usine non linéaire » avec une identification en régime permanent et une dynamique."""
    store = ProjectStore(root=tmp_path_factory.mktemp("cmp") / "projects", templates=TEMPLATES_DIR)
    p = store.create("usine", "usine_identification")
    run_steady(p, SteadyOptions(methods=("linear", "sparse"), sparse_thresholds=(0.1,), samples=80, val_samples=20, seed=1), log=lambda m: None)
    time.sleep(1.1)  # le dossier d'une identification porte l'heure à la seconde : deux identifications ne doivent pas se confondre
    run_identification(p, IdentifyOptions(methods=("linear", "polynomial"), poly_degrees=(2,), episodes=12, val_episodes=4, hold_max=8, max_steps=60, seed=1),
                       log=lambda m: None)
    return p


def _pick(project, mode):
    for mid, label in identified_choices(project):
        if mode in label:
            return tuple(mid.split(MODEL_SEP, 1))
    raise AssertionError(f"aucun modèle {mode}")


def test_choices_without_a_trained_run():
    assert policy_choices(None) == [("steps", "Paliers d'actions aléatoires (sans agent)")]


def test_identified_models_are_listed_recommended_first(project):
    choices = identified_choices(project)
    assert any("régime permanent" in t for _, t in choices) and any("dynamique" in t for _, t in choices)
    assert sum("★" in t for _, t in choices) == 2  # un modèle recommandé par identification
    assert all(MODEL_SEP in mid for mid, _ in choices)


def test_dynamic_model_is_replayed_in_free_simulation(project):
    steps, eps, metas = [], [], []
    s = run_evaluation(project, None, "steps", 2, 7, on_meta=metas.append, on_step=steps.append, on_episode=eps.append,
                       log=lambda m: None, identified=_pick(project, "dynamique"))
    assert metas[0]["model_names"] == ["y1", "y2"] and metas[0]["obs_names"] == ["y1", "y2"]
    assert all(len(e["model"]) == 2 for e in steps)
    assert np.allclose(steps[0]["model"], steps[0]["obs"])  # simulation libre : départ = observation réelle
    assert len(eps) == 2 and set(eps[0]["model_error"]) == {"y1", "y2"}
    assert 0 <= s["model_nrmse"]["y1"] < 1.0 and s["model_rmse"]["y2"] >= 0


def test_steady_model_follows_the_current_actions(project):
    steps = []
    s = run_evaluation(project, None, "steps", 1, 3, on_step=steps.append, log=lambda m: None, identified=_pick(project, "régime permanent"))
    assert steps and all(np.all(np.isfinite(e["model"])) for e in steps)
    from rlstudio.identify.pipeline import load_candidate

    model = load_candidate(*_pick(project, "régime permanent"))
    assert np.allclose(steps[5]["model"], model.predict(steps[5]["action"])[0])  # y = f(u) appliqué à l'action courante
    assert np.isfinite(s["model_nrmse"]["y1"])


def test_without_identified_model_nothing_changes(project):
    steps = []
    s = run_evaluation(project, None, "steps", 1, 3, on_step=steps.append, log=lambda m: None)
    assert steps and all("model" not in e for e in steps) and "model_nrmse" not in s and s["episodes"] == 1


def test_steps_pilot_is_reproducible_and_within_bounds(project):
    runs = []
    for _ in range(2):
        acts = []
        run_evaluation(project, None, "steps", 1, 11, on_step=lambda e: acts.append(e["action"]), log=lambda m: None)
        runs.append(np.array(acts))
    assert np.allclose(runs[0], runs[1]) and runs[0].min() >= -2.0 - 1e-6 and len(np.unique(runs[0].round(6), axis=0)) > 3


def test_model_for_other_signals_is_refused():
    model = types.SimpleNamespace(static=True, obs_names=["zz"], act_names=["u1"])
    with pytest.raises(ValueError, match="absents de ce projet"):
        _ModelOverlay(model, ["y1", "y2"], ["u1", "u2"])
    model = types.SimpleNamespace(static=True, obs_names=["y1"], act_names=["force"])
    with pytest.raises(ValueError, match="actions absentes"):
        _ModelOverlay(model, ["y1", "y2"], ["u1", "u2"])


def test_diverging_dynamic_model_gives_nan_not_a_crash():
    class Blowup:
        static, obs_names, act_names = False, ["y"], ["u"]

        def step(self, x, u):
            return x * 1e9

    ov = _ModelOverlay(Blowup(), ["y"], ["u"])
    ov.reset(np.array([1.0]))
    for _ in range(3):
        ov.advance(np.array([0.0]))
    assert np.isnan(ov.values(np.array([1.0]), np.array([0.0]))).all()


def test_compare_errors():
    real = np.column_stack([np.linspace(0, 1, 50), np.full(50, 3.0)])
    model = real + np.array([0.1, 0.0])
    err = compare_errors(["a", "b"], real, model)
    assert err["a"][0] == pytest.approx(0.1) and err["a"][1] == pytest.approx(0.1 / real[:, 0].std())
    assert err["b"] == (0.0, 0.0)  # signal constant parfaitement suivi : écart nul, pas de division par zéro
    assert np.isnan(compare_errors(["a"], [[1.0]], [[np.nan]])["a"][0])


def test_worker_streams_the_model_curves(project):
    mid = _pick(project, "dynamique")
    r = subprocess.run([sys.executable, "-m", "rlstudio.worker", "evaluate", "--project", str(project.path), "--policy", "steps",
                        "--episodes", "2", "--ident-dir", mid[0], "--ident-model", mid[1]],
                       capture_output=True, text=True, timeout=300, encoding="utf-8")
    ev = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    kinds = [e["type"] for e in ev]
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    assert kinds[0] == "meta" and ev[0]["model_names"] == ["y1", "y2"] and kinds.count("episode") == 2 and kinds[-1] == "done"
    assert all(len(e["model"]) == 2 for e in ev if e["type"] == "step") and "model_error" in next(e for e in ev if e["type"] == "episode")
    assert "model_nrmse" in ev[-1]


# -- interface ------------------------------------------------------------------------------------------------
@pytest.fixture
def tab(project, tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.eval_tab import EvalTab
    from rlstudio.settings import Settings

    QApplication.instance() or QApplication([])
    t = EvalTab(Settings(tmp_path / "s.json"))
    t.set_project(project)
    return t


@pytest.mark.gui
def test_tab_offers_steps_and_identified_models_without_any_run(tab):
    assert [tab.policy_combo.itemData(i) for i in range(tab.policy_combo.count())] == ["steps"]
    assert tab.btn_start.isEnabled() and tab.model_combo.itemText(0) == "Aucun" and tab.model_combo.count() >= 5  # aucun + 2 identifications x 2-3 modèles
    assert tab.model_combo.toolTip() and tab.k_err.toolTip()


@pytest.mark.gui
def test_tab_draws_model_curves_dashed_in_the_color_of_the_signal(tab):
    from PySide6.QtCore import Qt

    tab._reset_views()
    tab.model_names = ["y1"]
    tab._on_meta(["y1", "y2"], ["u1", "u2"])
    assert tab.p_obs.names == ["y1", "y2", "y1 (modèle)"]
    pens = [c.opts["pen"] for c in tab.p_obs.curves]
    assert pens[2].style() == Qt.DashLine and pens[2].color() == pens[0].color() and pens[0].style() == Qt.SolidLine
    tab._on_step({"episode": 1, "step": 1, "obs": np.array([1.0, 2.0]), "action": np.array([0.0, 0.0]), "reward": 0.0, "total": 0.0,
                  "model": np.array([1.2])})
    assert tab.p_obs._width == 3 and tab.p_obs._y[0, 2] == pytest.approx(1.2)
    tab._on_episode({"n": 1, "reward": 1.0, "length": 5, "terminated": False,
                     "model_error": {"y1": {"rmse": 0.1, "nrmse": 0.25}}})
    assert tab.k_err.value.text() == "0.250" and "écart du modèle" in tab.logbox.toPlainText()


@pytest.mark.gui
def test_new_identification_becomes_available_without_reopening_the_project(tab, project):
    import shutil

    base = project.path / "identify"
    victim = sorted(base.iterdir())[0]
    hidden = base.parent / "_hidden"
    shutil.move(str(victim), hidden / victim.name) if hidden.mkdir(exist_ok=True) is None else None
    tab.refresh_models()
    n_before = tab.model_combo.count()
    shutil.move(str(hidden / victim.name), str(victim))
    tab.refresh_models()
    assert tab.model_combo.count() > n_before
