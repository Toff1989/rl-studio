"""Ventilation a la demande d'une salle (qualite d'air, CO2), occupation variable.

Bilan de CO2 d'une salle de 180 m3 : generation par les occupants, apport d'air neuf (ventilateur) et infiltration.
Pas de temps : 1 minute ; episode = une journee de travail (de 7 h 30, `max_steps` minutes). L'occupation suit des
reunions tirees au hasard a chaque episode. Action : debit d'air neuf dans [0, 1] (1 = 0,5 m3/s).
Cout energetique : puissance du ventilateur (loi cubique du debit) + chauffage de l'air neuf (recuperateur a 60 %).
Recompense par pas : -(depassement de 1000 ppm / 200)^2 - `energy_weight` * energie [kWh].

Contrat d'un projet Python : `make_env(**params)` renvoie un gymnasium.Env ; `params` vient de `env.params`
dans project.yaml. Optionnel : `baseline_policy(env)` -> fonction obs -> action, tracee comme reference.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DT = 60.0                 # pas de temps [s]
VOLUME = 180.0            # volume de la salle [m3]
Q_MAX = 0.5               # debit d'air neuf maximal [m3/s]
Q_INFIL = 0.01            # infiltrations [m3/s]
CO2_OUT = 420.0           # CO2 exterieur [ppm]
CO2_PER_PERSON = 5.0e-6   # generation de CO2 par personne [m3/s]
CO2_LIMIT = 1000.0        # seuil de confort [ppm]
FAN_POWER = 600.0         # puissance du ventilateur a plein debit [W]
HEAT_RECOVERY = 0.6       # efficacite du recuperateur de chaleur
T_INDOOR = 21.0
RHO_CP = 1.2 * 1005.0     # capacite volumique de l'air [J/(m3.K)]
HOUR0 = 7.5


def step_co2(co2, occupants, flow_m3s, dt=DT):
    """Un pas d'Euler du bilan de CO2 [ppm]."""
    d = (occupants * CO2_PER_PERSON * 1e6 + (flow_m3s + Q_INFIL) * (CO2_OUT - co2)) / VOLUME
    return max(CO2_OUT, co2 + dt * d)


def power_kw(flow_m3s, t_ext):
    """Puissance electrique/thermique totale [kW] : ventilateur + rechauffage de l'air neuf."""
    fan = FAN_POWER * (flow_m3s / Q_MAX) ** 3
    heat = RHO_CP * flow_m3s * max(0.0, T_INDOOR - t_ext) * (1.0 - HEAT_RECOVERY)
    return (fan + heat) / 1000.0


def random_schedule(rng, n_steps):
    """Occupants par minute : 3 a 5 reunions de 4 a 16 personnes, 30 a 90 min, plus 1 a 2 personnes presentes en continu."""
    occ = np.full(n_steps, float(rng.integers(0, 3)))
    for _ in range(int(rng.integers(3, 6))):
        start = int(rng.integers(0, max(1, n_steps - 30)))
        length = int(rng.integers(30, 91))
        occ[start:start + length] += float(rng.integers(4, 17))
    return occ


class VentilationCo2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=600, energy_weight=5.0):
        self.max_steps, self.energy_weight = int(max_steps), float(energy_weight)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.obs_names = ["CO2", "occupants", "T_ext", "h_sin", "h_cos", "debit_precedent"]
        self.action_names = ["debit_air_neuf"]

    def _hour(self):
        return HOUR0 + self.k * DT / 3600.0

    def _obs(self):
        h = self._hour()
        occ = self.occupancy[min(self.k, self.max_steps - 1)]
        return np.array([self.co2, occ, self.t_ext, math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0),
                         self.last_u], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.occupancy = random_schedule(self.np_random, self.max_steps)
        self.t_ext = float(self.np_random.uniform(-5.0, 18.0))
        self.co2 = float(self.np_random.uniform(420.0, 600.0))
        self.k, self.last_u = 0, 0.0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        flow = u * Q_MAX
        occ = self.occupancy[self.k]
        self.co2 = step_co2(self.co2, occ, flow)
        energy = power_kw(flow, self.t_ext) * DT / 3600.0
        excess = max(0.0, self.co2 - CO2_LIMIT)
        reward = -(excess / 200.0) ** 2 - self.energy_weight * energy
        self.last_u, self.k = u, self.k + 1
        return self._obs(), float(reward), False, self.k >= self.max_steps, {"energy_kwh": energy}


def make_env(**params):
    return VentilationCo2Env(**params)


def baseline_policy(env):
    """Ventilation a la demande proportionnelle au CO2 (de 600 a 1000 ppm), reference de comparaison."""
    def policy(obs):
        return np.array([np.clip((obs[0] - 600.0) / 400.0, 0.05, 1.0)], dtype=np.float32)
    return policy
