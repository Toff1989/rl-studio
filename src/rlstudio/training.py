"""Entrainement et evaluation generiques d'un projet."""
import json
import time

import numpy as np
import yaml

from .envs.loader import make_baseline, make_env
from .project import Project

ALGOS = ("SAC", "PPO", "TD3")


def rollout(env, policy, seed=None, on_step=None):
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total, steps, done = total + r, steps + 1, term or trunc
        if on_step:
            on_step(steps)
    return total, steps


def evaluate(env, policy, n_episodes=3, seed=1000, on_episode=None, on_step=None):
    """Rejoue `n_episodes` episodes. `on_episode(i, n, retour, duree_s)` apres chacun, `on_step(i, n, pas)` a chaque pas."""
    rets = []
    for i in range(n_episodes):
        t = time.time()
        step_cb = (lambda k, i=i: on_step(i, n_episodes, k)) if on_step else None
        rets.append(rollout(env, policy, seed + i, on_step=step_cb)[0])
        if on_episode:
            on_episode(i, n_episodes, rets[-1], time.time() - t)
    return float(np.mean(rets)), float(np.std(rets))


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


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
          log=print, state_every=3, on_meta=None, on_progress=None, on_run=None):
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    algo = project.cfg["algo"]
    if algo["name"] not in ALGOS:
        raise ValueError(f"Algorithme inconnu : {algo['name']}")
    run_dir = project.new_run_dir()
    if on_run:
        on_run(run_dir)
    (run_dir / "project.yaml").write_text(yaml.safe_dump(project.cfg, allow_unicode=True), encoding="utf-8")
    log(f"[{project.name}] {algo['name']} | {algo['total_timesteps']} pas | {run_dir}")

    env = make_env(project)
    try:
        # le moteur MATLAB demarre au premier reset : on le fait ici pour le signaler (et le chronometrer)
        t = time.time()
        if project.kind == "simulink":
            log("Démarrage de MATLAB / Simulink (quelques dizaines de secondes)...")
        env.reset(seed=int(algo["seed"]))
        log(f"Environnement prêt en {fmt_duration(time.time() - t)}.")
        n_base = int(algo.get("baseline_episodes", 0))
        if n_base > 0:
            policy = make_baseline(project, env)
            label = "politique du projet"
            if policy is None:
                policy, label = (lambda obs: env.action_space.sample()), "politique aléatoire"
            log(f"Évaluation de référence : {n_base} episode(s) avec la {label} (avant l'apprentissage).")
            t_base = time.time()
            ep_steps = getattr(env, "n_steps", None)

            def base_episode(i, n, ret, duration):
                left = (time.time() - t_base) / (i + 1) * (n - i - 1)
                log(f"Reference {i + 1}/{n} | récompense {ret:.2f} | {fmt_duration(duration)}"
                    + (f" | reste ~{fmt_duration(left)}" if i + 1 < n else ""))
                if on_progress:
                    on_progress("baseline", (i + 1) / n)

            def base_step(i, n, k):
                if on_progress and ep_steps:
                    on_progress("baseline", (i + min(k / ep_steps, 1.0)) / n)

            mean, std = evaluate(env, policy, n_base, on_episode=base_episode, on_step=base_step)
            log(f"Référence ({label}) : {mean:.3f} +/- {std:.3f}")
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
        log(f"Apprentissage {algo['name']} : {int(algo['total_timesteps'])} pas sur {algo['device']}.")
        t0 = time.time()
        model.learn(int(algo["total_timesteps"]), callback=_callback(on_episode, on_state, should_stop, state_every))
        model.save(run_dir / "model")
        log(f"Terminé en {time.time() - t0:.0f} s. Modèle : {run_dir / 'model.zip'}")
    finally:
        env.close()
    return run_dir
