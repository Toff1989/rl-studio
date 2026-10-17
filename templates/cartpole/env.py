"""Pendule inverse sur chariot (CartPole), action continue.

Etat : position du chariot x, vitesse x', angle du pendule theta (0 = vertical), vitesse angulaire theta'.
Action : force normalisee dans [-1, 1] (multipliee par `force_mag`). Recompense : +1 par pas ou le pendule
reste debout, avec une petite penalite sur l'effort. L'episode s'arrete si le pendule tombe (|theta| > 12 deg)
ou si le chariot sort de la piste (|x| > 2,4 m), et est tronque apres `max_steps`.

Contrat d'un projet Python : `make_env(**params)` renvoie un gymnasium.Env ; `params` vient de `env.params`
dans project.yaml. Optionnel : `baseline_policy(env)` -> fonction obs -> action, tracee comme reference.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GRAVITY, MASS_CART, MASS_POLE, HALF_LENGTH, TAU = 9.8, 1.0, 0.1, 0.5, 0.02
THETA_LIMIT = 12 * math.pi / 180
X_LIMIT = 2.4


def step_dynamics(state, force):
    """Un pas d'Euler semi-implicite des equations du chariot-pendule."""
    x, x_dot, theta, theta_dot = state
    total_mass = MASS_CART + MASS_POLE
    pole_mass_length = MASS_POLE * HALF_LENGTH
    cos, sin = math.cos(theta), math.sin(theta)
    temp = (force + pole_mass_length * theta_dot ** 2 * sin) / total_mass
    theta_acc = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4.0 / 3.0 - MASS_POLE * cos ** 2 / total_mass))
    x_acc = temp - pole_mass_length * theta_acc * cos / total_mass
    x_dot += TAU * x_acc
    theta_dot += TAU * theta_acc
    return np.array([x + TAU * x_dot, x_dot, theta + TAU * theta_dot, theta_dot])


class CartPoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, force_mag=10.0, max_steps=500):
        self.force_mag, self.max_steps = float(force_mag), int(max_steps)
        high = np.array([X_LIMIT * 2, np.inf, THETA_LIMIT * 2, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.obs_names = ["x", "x_dot", "theta", "theta_dot"]
        self.action_names = ["force"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.05, 0.05, size=4)
        self.k = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.state = step_dynamics(self.state, u * self.force_mag)
        self.k += 1
        x, _, theta, _ = self.state
        fallen = abs(theta) > THETA_LIMIT or abs(x) > X_LIMIT
        reward = 0.0 if fallen else 1.0 - 0.01 * u * u
        return self.state.astype(np.float32), reward, bool(fallen), self.k >= self.max_steps, {}


def make_env(**params):
    return CartPoleEnv(**params)


def baseline_policy(env):
    """Regulateur proportionnel-derive sur l'angle et la position (reference de comparaison)."""
    def policy(obs):
        x, x_dot, theta, theta_dot = obs
        return np.array([np.clip(1.0 * theta + 0.35 * theta_dot + 0.02 * x + 0.05 * x_dot, -1.0, 1.0) * 10.0 / env.force_mag],
                        dtype=np.float32)
    return policy
