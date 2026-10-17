"""Environnement du projet. Exemple minimal : maintenir un integrateur pres d'une consigne.

Contrat : `make_env(**params)` renvoie un gymnasium.Env. `params` vient de `env.params`
dans project.yaml. Optionnel : `baseline_policy(env)` -> fonction obs -> action, utilisee
comme reference de comparaison pendant l'entrainement.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MyEnv(gym.Env):
    def __init__(self, setpoint=1.0, dt=0.1, horizon=200):
        self.setpoint, self.dt, self.horizon = setpoint, dt, horizon
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.k = float(self.np_random.uniform(-2, 2)), 0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(action[0], -1, 1))
        self.x += self.dt * u
        self.k += 1
        reward = -abs(self.x - self.setpoint) - 0.01 * u * u
        return self._obs(), reward, False, self.k >= self.horizon, {}

    def _obs(self):
        return np.array([self.x, self.setpoint - self.x], dtype=np.float32)


def make_env(**params):
    return MyEnv(**params)


def baseline_policy(env):
    return lambda obs: np.clip([2.0 * obs[1]], -1, 1).astype(np.float32)
