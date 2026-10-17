"""Collecte de donnees (politique experte, DAgger) et evaluation en boucle fermee."""
from __future__ import annotations

import numpy as np

from .surrogates import r_squared


class Stopped(Exception):
    """Interruption demandee par l'utilisateur."""


class Dataset:
    """Paires (observation -> action de l'expert), groupees par episode (necessaire pour l'integrale d'un PID)."""

    def __init__(self, obs_names, act_names, low, high):
        self.obs_names, self.act_names = list(obs_names), list(act_names)
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        self.episodes: list[tuple[np.ndarray, np.ndarray]] = []

    def add(self, obs, labels):
        self.episodes.append((np.asarray(obs, float), np.asarray(labels, float)))

    def copy(self):
        d = Dataset(self.obs_names, self.act_names, self.low, self.high)
        d.episodes = list(self.episodes)
        return d

    def obs_episodes(self):
        return [o for o, _ in self.episodes]

    def observations(self):
        return np.concatenate([o for o, _ in self.episodes])

    def actions(self):
        return np.concatenate([a for _, a in self.episodes])

    def __len__(self):
        return int(sum(len(o) for o, _ in self.episodes))


def rollout(env, behave, seed, reset=None, max_steps=None):
    """Un episode. `behave(obs) -> action`. Renvoie observations vues, actions executees, retour, longueur, arret anticipe."""
    obs, _ = env.reset(seed=int(seed))
    if reset:
        reset()
    seen, acts, total, terminated, steps = [], [], 0.0, False, 0
    done = False
    while not done:
        seen.append(np.asarray(obs, float).ravel())
        a = np.clip(np.asarray(behave(obs), float).reshape(-1), env.action_space.low, env.action_space.high)
        acts.append(a)
        obs, r, terminated, truncated, _ = env.step(a.astype(np.float32))
        total += float(r)
        steps += 1
        done = terminated or truncated or (max_steps is not None and steps >= max_steps)
    return {"obs": np.array(seen), "acts": np.array(acts), "return": total, "length": steps, "terminated": bool(terminated)}


def evaluate(env, act_fn, n_episodes, seed0, reset=None, expert=None, should_stop=None, max_steps=None, on_episode=None):
    """Retour moyen, episodes menes a terme (sans arret anticipe), et fidelite sur les etats effectivement visites."""
    runs = []
    for k in range(n_episodes):
        if should_stop and should_stop():
            raise Stopped()
        runs.append(rollout(env, act_fn, seed0 + k, reset=reset, max_steps=max_steps))
        if on_episode:
            on_episode(k + 1, n_episodes)
    rets = np.array([r["return"] for r in runs])
    out = {"return_mean": float(rets.mean()), "return_std": float(rets.std()),
           "length_mean": float(np.mean([r["length"] for r in runs])),
           "completed_rate": float(np.mean([not r["terminated"] for r in runs])),
           "obs_episodes": [r["obs"] for r in runs], "act_episodes": [r["acts"] for r in runs]}
    if expert is not None:
        labels = np.concatenate([expert.act_batch(r["obs"]) for r in runs])
        out["expert_labels"] = labels
        out["r2_visited"] = r_squared(labels, np.concatenate(out["act_episodes"]))
    return out


def collect_expert(env, expert, n_episodes, noise, seed0, rng, should_stop=None, max_steps=None, on_episode=None):
    """Rejoue l'expert avec un bruit d'action (pour visiter des etats voisins) ; etiquette chaque etat par l'expert."""
    data = Dataset(expert.obs_names, expert.act_names, expert.low, expert.high)
    half = (expert.high - expert.low) / 2.0

    def behave(o):
        return expert.act(o) + noise * half * rng.normal(size=len(half))

    for k in range(n_episodes):
        if should_stop and should_stop():
            raise Stopped()
        r = rollout(env, behave, seed0 + k, max_steps=max_steps)
        data.add(r["obs"], expert.act_batch(r["obs"]))
        if on_episode:
            on_episode(k + 1, n_episodes)
    return data


def dagger_round(env, expert, surrogate, data, n_episodes, beta, seed0, rng, should_stop=None, max_steps=None, on_episode=None):
    """Un tour de DAgger : on joue (en partie) avec le modele simple, l'expert etiquette les etats qu'il visite."""
    for k in range(n_episodes):
        if should_stop and should_stop():
            raise Stopped()

        def behave(o):
            return expert.act(o) if rng.random() < beta else surrogate.act(o)

        r = rollout(env, behave, seed0 + k, reset=surrogate.reset, max_steps=max_steps)
        data.add(r["obs"], expert.act_batch(r["obs"]))
        if on_episode:
            on_episode(k + 1, n_episodes)


def run_dagger(env, expert, fit, data, iterations, episodes, seed0, rng, should_stop=None, tick=None, max_steps=None, on_episode=None):
    """DAgger : ajuste, joue, reetiquette, agrege, rajuste. Renvoie (jeu de donnees agrege, dernier modele)."""
    data = data.copy()
    surrogate = fit(data)
    for it in range(1, iterations + 1):
        dagger_round(env, expert, surrogate, data, episodes, beta=0.5 ** it, seed0=seed0 + 1000 * it, rng=rng,
                     should_stop=should_stop, max_steps=max_steps,
                     on_episode=(lambda k, n, it=it: on_episode(it, k, n)) if on_episode else None)
        surrogate = fit(data)
        if tick:
            tick()
    return data, surrogate
