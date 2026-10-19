"""Ballon d'eau chaude sanitaire (300 L) chauffe par PAC, tarif heures pleines / creuses et production photovoltaique.

Ballon mono-noeud (temperature uniforme) : pertes vers l'ambiance, soutirages remplaces par de l'eau froide a 12 degC,
chauffe par une PAC dont le COP baisse avec la temperature du ballon. Pas de temps : 15 minutes ; episode = `max_steps`
pas (2 jours par defaut, depart a minuit). Action : puissance electrique dans [0, 1] (1 = 2 kW).
Tarif : heures creuses (22 h - 6 h) a 0,15 EUR/kWh, heures pleines a 0,27 EUR/kWh ; l'energie photovoltaique est
d'abord consommee sur place (surplus non valorise). Securite : le ballon est limite a 65 degC.
Recompense par pas : -10 * cout [EUR] - penalite si l'eau soutiree est trop froide (< 45 degC) ou ballon > 62 degC.

Contrat d'un projet Python : `make_env(**params)` renvoie un gymnasium.Env ; `params` vient de `env.params`
dans project.yaml. Optionnel : `baseline_policy(env)` -> fonction obs -> action, tracee comme reference.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DT = 900.0                  # pas de temps [s]
VOLUME_L = 300.0            # volume du ballon [L]
C_TANK = VOLUME_L * 4186.0  # capacite thermique [J/K]
UA_LOSS = 1.5               # pertes [W/K]
T_AMBIENT, T_COLD = 15.0, 12.0
P_EL_MAX = 2000.0           # puissance electrique maximale de la PAC [W]
T_COMFORT, T_MAX, T_SAFETY = 45.0, 62.0, 65.0
PRICE_PEAK, PRICE_OFFPEAK = 0.27, 0.15   # [EUR/kWh]


def price(hour):
    return PRICE_OFFPEAK if (hour >= 22.0 or hour < 6.0) else PRICE_PEAK


def cop(t_tank):
    return max(1.8, 4.2 - 0.045 * (t_tank - 20.0))


def draw_profile(hour):
    """Forme moyenne des soutirages [L/h] : douches du matin, midi, soiree."""
    return (45.0 * math.exp(-((hour - 7.5) / 1.0) ** 2) + 20.0 * math.exp(-((hour - 12.5) / 1.0) ** 2)
            + 55.0 * math.exp(-((hour - 20.0) / 1.5) ** 2) + 3.0)


def pv_power(hour, clouds, peak_w=1500.0):
    return peak_w * clouds * max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0)) ** 1.5


def step_tank(t, p_el, draw_l, dt=DT):
    """Un pas du ballon : (nouvelle temperature, energie thermique fournie [J])."""
    q = p_el * cop(t) * dt
    t_new = t + (q - UA_LOSS * (t - T_AMBIENT) * dt) / C_TANK
    t_new -= draw_l / VOLUME_L * (t_new - T_COLD)
    return min(t_new, T_SAFETY), q


class BallonEcsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=192, pv_peak_w=1500.0):
        self.max_steps, self.pv_peak_w = int(max_steps), float(pv_peak_w)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.obs_names = ["T_ballon", "prix", "pv", "soutirage", "h_sin", "h_cos", "T_prec_ecart"]
        self.action_names = ["puissance"]

    def _hour(self, k=None):
        return ((self.k if k is None else k) * DT / 3600.0) % 24.0

    def _clouds(self):
        return self.clouds[(self.k * int(DT)) // 86400 % len(self.clouds)]

    def _draw(self, k):
        return draw_profile(self._hour(k)) * DT / 3600.0 * self.demand * float(self.noise[k % len(self.noise)])

    def _obs(self):
        h = self._hour()
        return np.array([self.t, price(h), pv_power(h, self._clouds(), self.pv_peak_w) / 1000.0, self._draw(self.k) / (DT / 60.0),
                         math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0), self.t - self.t_prev],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = float(self.np_random.uniform(40.0, 58.0))
        self.t_prev = self.t
        self.demand = float(self.np_random.uniform(0.7, 1.4))              # taille du foyer
        self.noise = self.np_random.uniform(0.6, 1.4, size=self.max_steps + 1)
        self.clouds = self.np_random.uniform(0.2, 1.0, size=8)
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        h, p_el = self._hour(), u * P_EL_MAX
        draw = self._draw(self.k)
        pv = pv_power(h, self._clouds(), self.pv_peak_w)
        grid_kwh = max(0.0, p_el - pv) * DT / 3.6e6                         # seule l'energie du reseau est payee
        self.t_prev = self.t
        # temperature de l'eau au moment du soutirage (avant remplacement par de l'eau froide)
        t_draw = self.t
        self.t, _ = step_tank(self.t, p_el, draw)
        cold = max(0.0, T_COMFORT - t_draw) / 5.0
        reward = -10.0 * grid_kwh * price(h) - (draw / 10.0) * cold ** 2 - 0.05 * max(0.0, self.t - T_MAX)
        self.k += 1
        return self._obs(), float(reward), False, self.k >= self.max_steps, {"grid_kwh": grid_kwh}


def make_env(**params):
    return BallonEcsEnv(**params)


def baseline_policy(env):
    """Contacteur heures creuses : chauffe la nuit jusqu'a 58 degC ; en journee, uniquement si le ballon descend sous 42 degC."""
    def policy(obs):
        t, prix = float(obs[0]), float(obs[1])
        if prix <= PRICE_OFFPEAK + 1e-6:
            return np.array([1.0 if t < 58.0 else 0.0], dtype=np.float32)
        return np.array([1.0 if t < 42.0 else 0.0], dtype=np.float32)
    return policy
