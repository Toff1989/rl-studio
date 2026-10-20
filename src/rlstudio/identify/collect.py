"""Balayage d'un environnement : sequences d'actions variees -> transitions (x, u, x+), groupees par episode."""
from __future__ import annotations

import numpy as np

from ..distill.collect import Stopped


class DynData:
    """Episodes (X (T+1, d), U (T, m)) : X[k+1] est l'observation obtenue en appliquant U[k] depuis X[k]."""

    def __init__(self, obs_names, act_names, low, high, dt):
        self.obs_names, self.act_names = list(obs_names), list(act_names)
        self.low, self.high, self.dt = np.asarray(low, float), np.asarray(high, float), float(dt)
        self.episodes: list[tuple[np.ndarray, np.ndarray]] = []

    def add(self, X, U):
        X, U = np.asarray(X, float), np.asarray(U, float)
        if len(U) and len(X) == len(U) + 1:
            self.episodes.append((X, U))

    def copy(self):
        d = DynData(self.obs_names, self.act_names, self.low, self.high, self.dt)
        d.episodes = list(self.episodes)
        return d

    def extend(self, other):
        self.episodes += other.episodes

    def Z(self):
        return np.concatenate([np.hstack([X[:-1], U]) for X, U in self.episodes])

    def delta(self):
        return np.concatenate([X[1:] - X[:-1] for X, _ in self.episodes])

    def states(self):
        return np.concatenate([X for X, _ in self.episodes])

    def __len__(self):
        return int(sum(len(U) for _, U in self.episodes))


def action_sequence(rng, low, high, n_steps, hold_max=10, extreme=0.2):
    """Paliers d'action aleatoires : duree variable, amplitude variable, une part aux bornes (couvre les saturations)."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    mid, half = (low + high) / 2.0, (high - low) / 2.0
    amp = rng.uniform(0.1, 1.0)  # episodes de faible excitation (proches d'un equilibre) et de forte excitation
    U = np.empty((n_steps, len(low)))
    k = 0
    while k < n_steps:
        h = int(rng.integers(1, max(1, hold_max) + 1))
        level = np.where(rng.random(len(low)) < extreme, rng.choice([-1.0, 1.0], len(low)), rng.uniform(-1, 1, len(low)) * amp)
        U[k: k + h] = mid + half * level
        k += h
    return np.clip(U, low, high)


def sweep(env, n_episodes, seed0, rng, hold_max=10, extreme=0.2, max_steps=None, should_stop=None, on_episode=None, action_fn=None):
    """Joue `n_episodes` sequences d'actions variees (chaque `reset` retire les conditions initiales de l'environnement).

    `action_fn(rng, low, high, n) -> U` remplace le generateur par defaut (par ex. politique apprise + bruit)."""
    low, high = np.asarray(env.action_space.low, float), np.asarray(env.action_space.high, float)
    names_o = getattr(env, "obs_names", None) or [f"x{i}" for i in range(int(np.prod(env.observation_space.shape)))]
    names_a = getattr(env, "action_names", None) or [f"u{i}" for i in range(len(low))]
    data = DynData(names_o, names_a, low, high, float(getattr(env, "step_s", None) or getattr(env, "dt", None) or 1.0))
    horizon = int(max_steps or getattr(env, "n_steps", None) or getattr(env, "max_steps", None) or 200)
    gen = action_fn or (lambda r, lo, hi, n: action_sequence(r, lo, hi, n, hold_max, extreme))
    for k in range(n_episodes):
        if should_stop and should_stop():
            raise Stopped()
        obs, _ = env.reset(seed=int(seed0 + k))
        U = gen(rng, low, high, horizon)
        X, used = [np.asarray(obs, float).ravel()], []
        for u in U:
            obs, _r, terminated, truncated, _ = env.step(np.asarray(u, np.float32))
            used.append(u)
            X.append(np.asarray(obs, float).ravel())
            if terminated or truncated:
                break
        data.add(X, used)
        if on_episode:
            on_episode(k + 1, n_episodes)
    return data
