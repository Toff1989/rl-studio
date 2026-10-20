"""Usine non lineaire a 2 entrees / 2 sorties, pour essayer l'onglet « Identification » (regime permanent ou dynamique).

Regime permanent (formules exactes, a retrouver par l'identification) :

    y1 = 3 * tanh(u1) + 0.5 * u2**2
    y2 = sin(u1) * u2 + |u1|

Dynamique : premier ordre, y(k+1) = y(k) + 0.3 * (f(u) - y(k)) ; la sortie met donc quelques pas a s'etablir.
Les conditions initiales des sorties sont tirees a chaque `reset`. Entrees : u1 dans [-2, 2], u2 dans [0, 3].

Contrat d'un projet Python : `make_env(**params)` renvoie un gymnasium.Env. La recompense (suivi d'une consigne fixe sur y1)
n'a de sens que si l'on veut aussi entrainer un agent ; elle n'intervient pas dans l'identification.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

TAU_GAIN = 0.3


def steady(u1, u2):
    """Sorties en regime permanent pour des entrees constantes."""
    return np.array([3.0 * np.tanh(u1) + 0.5 * u2 ** 2, np.sin(u1) * u2 + abs(u1)])


class NonlinearPlant(gym.Env):
    metadata = {"render_modes": []}
    action_names = ["u1", "u2"]
    obs_names = ["y1", "y2"]

    def __init__(self, max_steps=120):
        self.max_steps = max_steps
        self.step_s = 1.0
        self.action_space = spaces.Box(np.array([-2.0, 0.0], np.float32), np.array([2.0, 3.0], np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.y = self.np_random.uniform(-1.0, 1.0, 2)
        self.k = 0
        return self.y.astype(np.float32), {}

    def step(self, action):
        u = np.clip(np.asarray(action, float), self.action_space.low, self.action_space.high)
        self.y = self.y + TAU_GAIN * (steady(*u) - self.y)
        self.k += 1
        reward = -float((self.y[0] - 2.0) ** 2)
        return self.y.astype(np.float32), reward, False, self.k >= self.max_steps, {}


def make_env(max_steps=120):
    return NonlinearPlant(max_steps)
