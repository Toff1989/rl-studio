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

    static = False

    @property
    def n_in(self):
        return len(self.obs_names) + len(self.act_names)

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


def sweep(env, n_episodes, seed0, rng, hold_max=10, extreme=0.2, max_steps=None, should_stop=None, on_episode=None, action_fn=None, bounds=None):
    """Joue `n_episodes` sequences d'actions variees (chaque `reset` retire les conditions initiales de l'environnement).

    `action_fn(rng, low, high, n) -> U` remplace le generateur par defaut (par ex. politique apprise + bruit).
    `bounds=(low, high)` restreint les plages d'action balayees (une entree fixee : low == high)."""
    low, high = (np.asarray(b, float) for b in (bounds or (env.action_space.low, env.action_space.high)))
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


class SteadyData:
    """Points de regime permanent : entrees U (N, m) -> sorties Y (N, d)."""

    static = True

    def __init__(self, out_names, in_names, low, high):
        self.obs_names, self.act_names = list(out_names), list(in_names)  # memes attributs que DynData : les ajustements sont communs
        self.low, self.high, self.dt = np.asarray(low, float), np.asarray(high, float), 0.0
        self.U: list[np.ndarray] = []
        self.Y: list[np.ndarray] = []
        self.steps: list[int] = []
        self.dropped = 0

    @property
    def n_in(self):
        return len(self.act_names)

    def add(self, u, y, steps):
        self.U.append(np.asarray(u, float).ravel())
        self.Y.append(np.asarray(y, float).ravel())
        self.steps.append(int(steps))

    def Z(self):
        return np.array(self.U)

    def delta(self):
        return np.array(self.Y)

    def select_outputs(self, names):
        idx = [self.obs_names.index(n) for n in names]
        self.obs_names = [self.obs_names[i] for i in idx]
        self.Y = [y[idx] for y in self.Y]

    def copy(self):
        d = SteadyData(self.obs_names, self.act_names, self.low, self.high)
        d.U, d.Y, d.steps, d.dropped = list(self.U), list(self.Y), list(self.steps), self.dropped
        return d

    def extend(self, other):
        self.U += other.U
        self.Y += other.Y
        self.steps += other.steps
        self.dropped += other.dropped

    def __len__(self):
        return len(self.U)


def latin_hypercube(rng, n, low, high, extreme=0.15):
    """Plan d'entrees : hypercube latin (couverture uniforme de chaque entree) dont une part des coordonnees est prise aux bornes."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    m = len(low)
    cells = (np.array([rng.permutation(n) for _ in range(m)]).T + rng.random((n, m))) / n
    U = low + cells * (high - low)
    edge = rng.random((n, m)) < extreme
    return np.where(edge, np.where(rng.random((n, m)) < 0.5, low, high), U)


def steady_sweep(env, design, seed0, rtol=1e-3, atol=1e-6, window=3, max_steps=None, scenario=False, should_stop=None, on_point=None,
                 swept=None, base=None):
    """Pour chaque ligne de `design`, maintient l'action constante jusqu'a convergence des sorties (|dy| <= atol + rtol.|y| pendant
    `window` pas consecutifs) et releve la sortie. Les points qui ne convergent pas sont ecartes (compteur `dropped`).

    `scenario=True` : les variables tirees au `reset` (cle `initial` d'un projet Simulink : perturbations) deviennent des entrees.
    `swept` : indices des actions balayees (colonnes de `design`) ; `base` : vecteur d'action complet dont les autres composantes sont fixees."""
    env_low, env_high = np.asarray(env.action_space.low, float), np.asarray(env.action_space.high, float)
    swept = list(range(len(env_low))) if swept is None else list(swept)
    base = np.asarray(base if base is not None else (env_low + env_high) / 2.0, float)
    low, high = env_low[swept], env_high[swept]
    names_o = getattr(env, "obs_names", None) or [f"y{i}" for i in range(int(np.prod(env.observation_space.shape)))]
    names_a = list(getattr(env, "action_names", None) or [f"u{i}" for i in range(len(env_low))])
    names_a = [names_a[i] for i in swept]
    horizon = int(max_steps or getattr(env, "n_steps", None) or getattr(env, "max_steps", None) or 200)
    data, extra_names = None, []
    for k, u in enumerate(np.clip(design, low, high)):
        if should_stop and should_stop():
            raise Stopped()
        obs, _ = env.reset(seed=int(seed0 + k))
        u_full = base.copy()
        u_full[swept] = u
        extra = dict(getattr(env, "_init_vars", None) or {}) if scenario else {}
        if data is None:
            extra_names = sorted(extra)
            data = SteadyData(names_o, names_a + extra_names, np.r_[low, [-np.inf] * len(extra_names)], np.r_[high, [np.inf] * len(extra_names)])
        prev, calm, steps, settled = np.asarray(obs, float).ravel(), 0, 0, False
        while steps < horizon:
            obs, _r, terminated, truncated, _ = env.step(np.asarray(u_full, np.float32))
            y = np.asarray(obs, float).ravel()
            steps += 1
            calm = calm + 1 if np.all(np.abs(y - prev) <= atol + rtol * np.abs(y)) else 0
            prev = y
            if calm >= window:
                settled = True
                break
            if terminated or truncated:
                break
        if settled:
            data.add(np.r_[u, [extra[n] for n in extra_names]], prev, steps)
        else:
            data.dropped += 1
        if on_point:
            on_point(k + 1, len(design))
    if data is None:
        raise ValueError("Plan d'experience vide.")
    return data
