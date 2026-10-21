import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from rlstudio.envs import expressions
from rlstudio.envs.loader import make_baseline, make_env
from rlstudio.envs.simulink import SimulinkEnv, validate_spec
from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import ProjectStore
from rlstudio.training import evaluate, train


@pytest.fixture
def store(tmp_path):
    return ProjectStore(root=tmp_path / "projects", templates=TEMPLATES_DIR)


# -- gestion de projets -------------------------------------------------------------------
def test_templates_are_listed(store):
    ids = {t[0] for t in store.list_templates()}
    assert {"python_blank", "simulink_blank", "cartpole", "cartpole_simulink", "pac_zone", "ventilation_co2", "ballon_ecs"} <= ids


def test_project_crud(store):
    p = store.create("Mon projet", "python_blank")
    assert p.name == "Mon projet" and p.slug == "mon-projet"
    assert store.create("Mon projet", "python_blank").slug == "mon-projet-2"

    p.cfg["algo"]["total_timesteps"] = 1234
    p.save()
    assert store.get("mon-projet").cfg["algo"]["total_timesteps"] == 1234

    store.rename(p, "Autre nom")
    assert store.get("mon-projet").name == "Autre nom"

    copy = store.duplicate(p)
    assert copy.slug != p.slug and (copy.path / "env.py").exists()

    trash = store.delete(p)
    assert trash.exists() and not p.path.exists()
    assert "mon-projet" not in [x.slug for x in store.list()]


# -- expressions --------------------------------------------------------------------------
def test_expressions():
    assert expressions.check_expression("-(y-1)**2", ["y"]) is None
    assert "inconnu" in expressions.check_expression("z + 1", ["y"])
    assert expressions.check_expression("1 +", ["y"]).startswith("Syntaxe")
    assert expressions.evaluate("-abs(y - 3)", {"y": 1.0}) == -2.0
    with pytest.raises(Exception):  # pas d'acces a __import__
        expressions.evaluate("__import__('os')", {})


# -- environnements Python ----------------------------------------------------------------
@pytest.mark.parametrize("template", ["python_blank", "cartpole"])
def test_python_templates_are_valid_envs(store, template):
    p = store.create("t", template)
    env = make_env(p)
    check_env(env, skip_render_check=True)
    assert make_baseline(p, env) is not None


def test_cartpole_physics_and_baseline(store):
    from rlstudio.envs.loader import load_module

    p = store.create("pendule", "cartpole")
    mod = load_module(p, "env.py")
    # sans force, un pendule incline continue de tomber (instabilite) et le chariot reste quasi immobile
    state = np.array([0.0, 0.0, 0.05, 0.0])
    for _ in range(20):
        state = mod.step_dynamics(state, 0.0)
    assert state[2] > 0.05
    # le regulateur de reference tient nettement plus longtemps qu'une action nulle
    env = mod.make_env()

    def episode_length(policy, seed):
        obs, _ = env.reset(seed=seed)
        n, done = 0, False
        while not done:
            obs, _, term, trunc, _ = env.step(policy(obs))
            n, done = n + 1, term or trunc
        return n

    baseline = mod.baseline_policy(env)
    zero = lambda o: np.zeros(1, dtype=np.float32)  # noqa: E731
    assert np.mean([episode_length(baseline, s) for s in range(5)]) > 3 * np.mean([episode_length(zero, s) for s in range(5)])


@pytest.mark.runtime
def test_short_training_run(store):
    p = store.create("t", "python_blank")
    p.cfg["algo"].update(total_timesteps=300, baseline_episodes=1)
    episodes = []
    run_dir = train(p, on_episode=lambda *a: episodes.append(a), log=lambda m: None)
    assert (run_dir / "model.zip").exists() and (run_dir / "baseline.json").exists()
    assert episodes


# -- Simulink generique (avec faux moteur MATLAB) -----------------------------------------
class FakeSession:
    """Imite rlstudio_step.m : un integrateur x' = u, observe via le signal 'x'."""

    def __init__(self):
        self.x, self.calls = 0.0, []

    def call(self, model_path, payload, t0, dt, reset):
        req = json.loads(payload)
        self.calls.append((req, t0, dt, reset))
        if reset:
            self.x = req["variables"].get("x0", 0.0)
        u = next((b["value"] for b in req["blocks"] if b["param"] == "Value"), 0.0)
        self.x += u * dt
        return json.dumps({s["name"]: [self.x, 2 * self.x] if s["name"] == "vec" else self.x
                           for s in req["signals"]})

    def close(self):
        pass


SPEC = {
    "model": "m.slx", "step_s": 1.0, "episode_s": 5.0,
    "actions": [{"name": "u", "block": "m/Const", "param": "Value", "low": -1, "high": 1}],
    "observations": [{"name": "x", "block": "m/Int", "port": 1}],
    "initial": [{"variable": "x0", "low": 1.0, "high": 2.0}],
    "reward": "-abs(x - 3) - 0.1*u**2",
    "terminate": "x > 100",
}


def test_simulink_env_with_fake_session(tmp_path):
    env = SimulinkEnv(tmp_path, SPEC, session=FakeSession())
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert 1.0 <= obs[0] <= 2.0
    x0 = float(obs[0])
    obs, r, term, trunc, _ = env.step(np.array([1.0]))
    assert abs(obs[0] - (x0 + 1.0)) < 1e-6
    assert abs(r - (-abs(obs[0] - 3) - 0.1)) < 1e-5
    for _ in range(4):
        _, _, term, trunc, _ = env.step(np.array([0.0]))
    assert trunc and not term


def test_simulink_vector_signal_size_mismatch(tmp_path):
    spec = {**SPEC, "observations": [{"name": "vec", "block": "m/V", "port": 1, "size": 3}], "reward": "-abs(u)", "terminate": ""}
    env = SimulinkEnv(tmp_path, spec, session=FakeSession())
    with pytest.raises(ValueError, match="size"):
        env.reset()


def test_validate_spec_reports_problems():
    bad = {**SPEC, "reward": "foo + 1", "actions": [{"name": "u", "low": 1, "high": 0}]}
    errs = validate_spec(bad)
    assert any("reward" in e for e in errs) and any("low" in e for e in errs)
    assert validate_spec(SPEC) == []
