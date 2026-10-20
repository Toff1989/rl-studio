"""Modele entrees -> sorties en regime permanent (usine jouet non lineaire a dynamique lente, sans MATLAB)."""
import types

import numpy as np
import pytest

from rlstudio.identify import SteadyOptions, steady_env
from rlstudio.identify.collect import latin_hypercube, steady_sweep
from rlstudio.identify.pipeline import load_candidate


class Plant:
    """Deux entrees, deux sorties : saturation, terme quadratique, sin, valeur absolue ; premier ordre (la sortie met du temps a s'etablir)."""

    n_steps, obs_names, action_names = 120, ["y1", "y2"], ["u1", "u2"]
    action_space = types.SimpleNamespace(low=np.array([-2.0, 0.0]), high=np.array([2.0, 3.0]))
    observation_space = types.SimpleNamespace(shape=(2,))

    @staticmethod
    def static(u1, u2):
        return np.array([3 * np.tanh(u1) + 0.5 * u2 ** 2, np.sin(u1) * u2 + np.abs(u1)])

    def reset(self, seed=None):
        self.y = np.random.default_rng(seed).uniform(-1, 1, 2)
        self._init_vars = {"d": float(np.random.default_rng(seed).uniform(0, 1))}
        return self.y.copy(), {}

    def step(self, u):
        self.y = self.y + 0.3 * (self.static(*np.asarray(u, float)) - self.y)
        return self.y.copy(), 0.0, False, False, {}


class Oscillator(Plant):
    def step(self, u):
        self.k = getattr(self, "k", 0) + 1
        return np.array([np.sin(self.k), 0.0]), 0.0, False, False, {}


def test_latin_hypercube_covers_each_input():
    U = latin_hypercube(np.random.default_rng(0), 100, [0, -1], [1, 1], extreme=0.0)
    assert U.shape == (100, 2)
    for k in range(2):
        assert len(np.unique(np.floor((U[:, k] - [0, -1][k]) / ([1, 2][k]) * 100))) == 100  # une valeur par tranche


def test_sweep_waits_for_steady_state_and_drops_unsettled():
    data = steady_sweep(Plant(), np.array([[1.0, 2.0]]), 0)
    assert np.allclose(data.Y[0], Plant.static(1.0, 2.0), atol=5e-3) and data.steps[0] > 10
    osc = steady_sweep(Oscillator(), np.array([[0.0, 1.0]] * 3), 0, max_steps=30)
    assert len(osc) == 0 and osc.dropped == 3


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("st")
    opts = SteadyOptions(samples=150, val_samples=40, epochs=250, hidden=(24, 24), poly_degrees=(3,), sparse_thresholds=(0.1, 0.03), seed=2)
    return steady_env(Plant(), opts, name="usine", out_dir=out, log=lambda m: None), out


def test_ranking_and_files(report):
    rep, out = report
    by = {c["id"]: c for c in rep["candidates"]}
    assert rep["mode"] == "steady" and rep["dropped"] == 0
    assert by["linear"]["r2"] < 0.9 < by["neural"]["r2"]
    assert by["sparse-t0.03"]["r2"] > 0.999
    assert by["sparse-t0.03"]["complexity"] <= 6  # 4 termes : retrouve la formule
    assert rep["recommended"].startswith("sparse")
    assert (out / "report.md").exists() and (out / "neural" / "steady_state.m").exists()
    assert "sin(u1)*u2" in " ".join(by["sparse-t0.03"]["equations"])


@pytest.mark.parametrize("cid", ["linear", "poly-d3", "sparse-t0.03", "neural"])
def test_exported_code_matches_model(report, cid):
    _, out = report
    model = load_candidate(out, cid)
    assert model.static
    ns = {}
    exec((out / cid / "steady_state.py").read_text(), ns)
    rng = np.random.default_rng(1)
    for _ in range(5):
        u = rng.uniform([-2, 0], [2, 3])
        assert np.allclose(ns["steady_state"](u), model.predict(u[None])[0], atol=1e-8)
    m = (out / cid / "steady_state.m").read_text()
    assert m.startswith("function y = steady_state(u)") and "y = d;" in m


def test_scenario_variables_become_inputs():
    rep = steady_env(Plant(), SteadyOptions(methods=("linear",), samples=20, val_samples=5, scenario=True), log=lambda m: None)
    assert rep["in_names"] == ["u1", "u2", "d"]


def test_ranges_fixed_and_outputs():
    opts = SteadyOptions(methods=("linear",), samples=20, val_samples=5, ranges={"u1": [0.0, 1.0]}, fixed={"u2": 2.0}, outputs=["y2"])
    rep = steady_env(Plant(), opts, log=lambda m: None)
    assert rep["in_names"] == ["u1"] and rep["out_names"] == ["y2"]
    for bad in ({"fixed": {"nope": 1}}, {"ranges": {"u1": [1, 0]}}, {"fixed": {"u1": 0, "u2": 0}}, {"outputs": ["zz"]}):
        with pytest.raises(ValueError):
            steady_env(Plant(), SteadyOptions(methods=("linear",), samples=10, val_samples=3, **bad), log=lambda m: None)


def test_dynamic_ranges_and_fixed():
    from rlstudio.identify import IdentifyOptions, identify_env

    sys_path = __import__("sys").path
    sys_path.insert(0, "tests")
    from test_identify import Pendulum

    opts = IdentifyOptions(methods=("linear",), episodes=4, val_episodes=2, ranges={"torque": [-1, 1]})
    seen = []
    env = Pendulum()
    orig = env.step
    env.step = lambda u: (seen.append(float(u[0])), orig(u))[1]
    identify_env(env, opts, log=lambda m: None)
    assert max(map(abs, seen)) <= 1.0 + 1e-6
    seen.clear()
    identify_env(env, IdentifyOptions(methods=("linear",), episodes=3, val_episodes=1, fixed={"torque": 0.5}), log=lambda m: None)
    assert set(np.round(seen, 6)) == {0.5}
