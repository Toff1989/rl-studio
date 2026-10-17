"""Entrainement et evaluation generiques d'un projet."""
import json
import time

import numpy as np
import yaml

from .envs.loader import make_baseline, make_env
from .project import Project

ALGOS = ("SAC", "PPO", "TD3")


def rollout(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total, steps, done = total + r, steps + 1, term or trunc
    return total, steps


def evaluate(env, policy, n_episodes=3, seed=1000):
    rets = [rollout(env, policy, seed + i)[0] for i in range(n_episodes)]
    return float(np.mean(rets)), float(np.std(rets))


def _callback(on_episode, on_state, should_stop, state_every):
    from stable_baselines3.common.callbacks import BaseCallback

    class Hooks(BaseCallback):
        episodes = 0

        def _on_step(self):
            info = self.locals["infos"][0]
            if on_state and self.num_timesteps % state_every == 0:
                on_state({
                    "timesteps": self.num_timesteps,
                    "obs": np.asarray(self.locals["new_obs"][0]).copy(),
                    "action": np.asarray(self.locals["actions"][0]).copy(),
                    "reward": float(self.locals["rewards"][0]),
                    "info": info,
                })
            if self.locals["dones"][0] and "episode" in info:
                self.episodes += 1
                if on_episode:
                    on_episode(self.episodes, float(info["episode"]["r"]), int(info["episode"]["l"]),
                               self.num_timesteps)
            return not (should_stop and should_stop())

    return Hooks()


def train(project: Project, on_episode=None, on_state=None, on_baseline=None, should_stop=None,
          log=print, state_every=3, on_meta=None):
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    algo = project.cfg["algo"]
    if algo["name"] not in ALGOS:
        raise ValueError(f"Algorithme inconnu : {algo['name']}")
    run_dir = project.new_run_dir()
    (run_dir / "project.yaml").write_text(yaml.safe_dump(project.cfg, allow_unicode=True), encoding="utf-8")
    log(f"[{project.name}] {algo['name']} | {algo['total_timesteps']} pas | {run_dir}")

    env = make_env(project)
    try:
        n_base = int(algo.get("baseline_episodes", 0))
        if n_base > 0:
            policy = make_baseline(project, env)
            label = "politique du projet"
            if policy is None:
                policy, label = (lambda obs: env.action_space.sample()), "politique aleatoire"
            mean, std = evaluate(env, policy, n_base)
            log(f"Reference ({label}) : {mean:.3f} +/- {std:.3f}")
            if on_baseline:
                on_baseline(mean, label)
            (run_dir / "baseline.json").write_text(json.dumps({"label": label, "mean": mean, "std": std}))

        model = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[algo["name"]](
            "MlpPolicy", Monitor(env), seed=int(algo["seed"]), device=algo["device"], verbose=0,
            **(algo.get("kwargs") or {}),
        )
        if on_meta:
            on_meta({"obs_names": list(getattr(env, "obs_names", None) or []),
                     "action_names": list(getattr(env, "action_names", None) or [])})
        t0 = time.time()
        model.learn(int(algo["total_timesteps"]), callback=_callback(on_episode, on_state, should_stop, state_every))
        model.save(run_dir / "model")
        log(f"Termine en {time.time() - t0:.0f} s. Modele : {run_dir / 'model.zip'}")
    finally:
        env.close()
    return run_dir
