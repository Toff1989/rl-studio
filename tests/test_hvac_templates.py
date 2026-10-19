"""Modeles CVC / batiment : environnements valides, physique coherente, reference meilleure que les politiques triviales."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
IDS = ["pac_zone", "ventilation_co2", "ballon_ecs"]


def load(tid):
    spec = importlib.util.spec_from_file_location(f"tpl_{tid}", TEMPLATES / tid / "env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def total_reward(mod, policy, seeds=(0, 1, 2)):
    env = mod.make_env()
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            total, done = total + r, term or trunc
        out.append(total)
    return float(np.mean(out))


@pytest.mark.parametrize("tid", IDS)
def test_template_is_valid_env_with_baseline(tid):
    mod = load(tid)
    env = mod.make_env()
    check_env(env, skip_render_check=True)
    assert len(env.obs_names) == env.observation_space.shape[0] and len(env.action_names) == env.action_space.shape[0]
    assert np.isfinite(total_reward(mod, mod.baseline_policy(env), seeds=(0,)))


@pytest.mark.parametrize("tid", IDS)
def test_template_yaml_is_complete(tid):
    import yaml

    cfg = yaml.safe_load((TEMPLATES / tid / "project.yaml").read_text(encoding="utf-8"))
    assert cfg["kind"] == "python" and cfg["env"]["entry"] == "env.py:make_env" and cfg["template_name"]


def test_pac_heats_and_cop_drops_in_the_cold():
    mod = load("pac_zone")
    q_cold, p_cold = mod.heat_pump(1.0, -10.0)
    q_mild, p_mild = mod.heat_pump(1.0, 10.0)
    assert q_cold / p_cold < q_mild / p_mild and q_cold < q_mild
    assert mod.heat_pump(0.1, 5.0) == (0.0, 0.0)  # sous la modulation minimale : arret
    ti, tm = 18.0, 18.0
    for _ in range(36):
        ti, tm = mod.step_thermal(ti, tm, 0.0, 6000.0, 0.0)
    assert ti > 18.5
    zero = lambda o: np.zeros(1, dtype=np.float32)  # noqa: E731
    env = mod.make_env()
    assert total_reward(mod, mod.baseline_policy(env)) > total_reward(mod, zero)


def test_co2_rises_with_occupants_and_ventilation_lowers_it():
    mod = load("ventilation_co2")
    c = 450.0
    for _ in range(60):
        c = mod.step_co2(c, 12, 0.0)
    assert c > 1000.0
    for _ in range(60):
        c = mod.step_co2(c, 0, 0.5)
    assert c < 600.0
    assert mod.power_kw(0.5, 25.0) == pytest.approx(8 * mod.power_kw(0.25, 25.0))  # ventilateur seul : loi cubique
    assert mod.power_kw(0.5, 0.0) > mod.power_kw(0.5, 25.0)                       # + chauffage de l'air neuf par temps froid
    env = mod.make_env()
    zero = lambda o: np.zeros(1, dtype=np.float32)  # noqa: E731
    full = lambda o: np.ones(1, dtype=np.float32)  # noqa: E731
    base = total_reward(mod, mod.baseline_policy(env))
    assert base > total_reward(mod, zero) and base > total_reward(mod, full)


def test_ballon_physics_and_baseline():
    mod = load("ballon_ecs")
    t, _ = mod.step_tank(50.0, 0.0, 0.0)
    assert t < 50.0                                  # pertes
    t2, _ = mod.step_tank(50.0, 2000.0, 0.0)
    assert t2 > t
    t3, _ = mod.step_tank(50.0, 0.0, 30.0)
    assert t3 < t                                    # un soutirage refroidit le ballon
    assert mod.step_tank(64.9, 2000.0, 0.0)[0] <= mod.T_SAFETY
    assert mod.price(23.0) < mod.price(12.0)
    env = mod.make_env()
    zero = lambda o: np.zeros(1, dtype=np.float32)  # noqa: E731
    always = lambda o: np.ones(1, dtype=np.float32)  # noqa: E731
    base = total_reward(mod, mod.baseline_policy(env))
    assert base > total_reward(mod, zero) and base > total_reward(mod, always)
