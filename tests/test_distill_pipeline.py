"""Extraction exacte du reseau SB3 et pipeline de distillation complet (cartpole, entrainements tres courts)."""
import numpy as np
import pytest

from rlstudio.distill import DistillOptions, run_distillation
from rlstudio.distill.export import distill_dir, load_candidate, load_report
from rlstudio.distill.policy import load_expert
from rlstudio.envs.loader import make_env
from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import ProjectStore
from rlstudio.training import train


@pytest.fixture
def store(tmp_path):
    return ProjectStore(root=tmp_path / "projects", templates=TEMPLATES_DIR)


def trained(store, algo, steps=400):
    p = store.create(f"d-{algo}", "cartpole")
    p.cfg["algo"].update(name=algo, total_timesteps=steps, baseline_episodes=1)
    if algo != "PPO":
        p.cfg["algo"]["kwargs"] = {"learning_starts": 100}
    p.cfg["env"]["params"]["max_steps"] = 60
    p.save()
    run_dir = train(p, log=lambda m: None)
    return p, run_dir


@pytest.mark.parametrize("algo", ["SAC", "PPO", "TD3"])
def test_neural_extraction_is_exact(store, algo):
    p, run_dir = trained(store, algo)
    env = make_env(p)
    try:
        expert = load_expert(p, run_dir, env)
        net = expert.neural()
        rng = np.random.default_rng(0)
        obs = rng.normal(size=(40, len(expert.obs_names))).astype(np.float32)
        ref = expert.act_batch(obs)
        got = net.predict_episodes([obs])
        assert np.allclose(ref, got, atol=1e-4), np.abs(ref - got).max()
        assert "tanh" in net.to_matlab("policy") or "max" in net.to_matlab("policy") or "tanh" in net.to_python("policy")
    finally:
        env.close()


def test_full_pipeline_and_report(store):
    p, run_dir = trained(store, "SAC")
    opts = DistillOptions(episodes=3, dagger_iters=1, dagger_episodes=2, eval_episodes=2, tree_depths=(3,),
                          symbolic_terms=(3,), max_steps=60)
    events = []
    report = run_distillation(p, run_dir, opts, log=lambda m: None, progress=lambda s, f: events.append(f))
    ids = {c["id"] for c in report["candidates"]}
    assert {"linear", "table", "tree-d3", "symbolic-k3", "neural"} <= ids
    assert report["recommended"] in ids
    assert events and events[-1] == 1.0 and events == sorted(events)
    # le reseau exact reproduit la politique apprise
    neural = next(c for c in report["candidates"] if c["id"] == "neural")
    assert neural["r2_expert"] > 0.999 and abs(neural["score"] - 1.0) < 1e-6
    # fichiers et rechargement
    for c in report["candidates"]:
        d = distill_dir(run_dir) / c["id"]
        assert (d / "policy.py").exists() and (d / "policy.m").exists() and (d / "equations.txt").exists()
    assert load_report(run_dir)["recommended"] == report["recommended"]
    sur = load_candidate(run_dir, "linear")
    assert sur.act(np.zeros(len(report["obs_names"]))).shape == (1,)
    assert (distill_dir(run_dir) / "report.md").read_text(encoding="utf-8").startswith("# Modèle")


def test_unknown_method_rejected(store):
    p, run_dir = trained(store, "SAC")
    with pytest.raises(ValueError):
        run_distillation(p, run_dir, DistillOptions(methods=("magie",)), log=lambda m: None)
