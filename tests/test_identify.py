"""Identification de dynamique : balayage, modeles, validation en simulation libre, code exporte (environnement jouet, sans MATLAB)."""
import types

import numpy as np
import pytest

from rlstudio.identify import IdentifyOptions, identify_env, models
from rlstudio.identify.collect import sweep
from rlstudio.identify.pipeline import load_candidate


class Pendulum:
    """Pendule amorti, frottement sec, couple sature : sin(theta), saturation et signe de la vitesse."""

    step_s, n_steps = 0.05, 60
    obs_names, action_names = ["theta", "omega"], ["torque"]
    action_space = types.SimpleNamespace(low=np.array([-3.0]), high=np.array([3.0]))
    observation_space = types.SimpleNamespace(shape=(2,))

    def reset(self, seed=None):
        r = np.random.default_rng(seed)
        self.x = np.array([r.uniform(-2.5, 2.5), r.uniform(-2, 2)])
        return self.x.copy(), {}

    def step(self, u):
        th, om = self.x
        tq = float(np.clip(u[0], -3, 3))
        for _ in range(5):
            dom = -9.81 * np.sin(th) - 0.3 * om - 0.4 * np.tanh(20 * om) + 2.0 * np.tanh(tq)
            om += 0.01 * dom
            th += 0.01 * om
        self.x = np.array([th, om])
        return self.x.copy(), 0.0, False, False, {}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    opts = IdentifyOptions(episodes=40, val_episodes=8, epochs=120, hidden=(24, 24), sparse_thresholds=(0.1, 0.03), poly_degrees=(3,), seed=1)
    out = tmp_path_factory.mktemp("id")
    rep = identify_env(Pendulum(), opts, name="pendule", out_dir=out, log=lambda m: None)
    return rep, out


def test_sweep_covers_actions_and_states():
    env = Pendulum()
    data = sweep(env, 10, 0, np.random.default_rng(0))
    U = np.concatenate([u for _, u in data.episodes])
    assert U.min() < -2.5 and U.max() > 2.5 and len(data) == 600
    assert np.all(np.abs(U) <= 3.0) and data.dt == 0.05 and data.obs_names == ["theta", "omega"]


def test_report_and_ranking(report):
    rep, out = report
    by = {c["id"]: c for c in rep["candidates"]}
    assert rep["recommended"] in by
    # la non-linearite est reelle : le lineaire est nettement moins fidele en simulation libre que le reseau
    assert by["neural"]["free_r2"] > by["linear"]["free_r2"] + 0.05
    assert by["neural"]["free_r2"] > 0.9
    assert (out / "report.md").exists() and (out / "neural" / "dynamics.m").exists()
    assert "1" in by["neural"]["nrmse_by_horizon"]


def test_sparse_identification_finds_sine(report):
    rep, _ = report
    sp = next(c for c in rep["candidates"] if c["id"] == "sparse-t0.1")
    txt = " ".join(sp["equations"])
    assert "sin(theta)" in txt and sp["complexity"] < 25


@pytest.mark.parametrize("cid", ["linear", "poly-d3", "sparse-t0.03", "neural"])
def test_exported_python_matches_model(report, cid):
    rep, out = report
    model = load_candidate(out, cid)
    ns = {}
    exec((out / cid / "dynamics.py").read_text(), ns)
    rng = np.random.default_rng(3)
    for _ in range(5):
        x, u = rng.uniform(-2, 2, 2), rng.uniform(-3, 3, 1)
        assert np.allclose(ns["dynamics"](x, u), model.step(x, u), atol=1e-8)
    m = (out / cid / "dynamics.m").read_text()
    assert m.startswith("function xn = dynamics(x, u)") and "%#codegen" in m and "inf" not in m.lower().replace("info", "")


def test_serialization_roundtrip(report):
    _, out = report
    a, b = load_candidate(out, "neural"), models.model_from_dict(load_candidate(out, "neural").to_dict())
    z = np.array([[0.3, -0.2, 1.0]])
    assert np.allclose(a.delta(z), b.delta(z))


def test_learning_curve_rounds():
    opts = IdentifyOptions(methods=("linear",), episodes=6, val_episodes=3, rounds=2, round_episodes=4)
    rep = identify_env(Pendulum(), opts, log=lambda m: None)
    assert [p["transitions"] for p in rep["learning_curve"]] == [360, 600]
