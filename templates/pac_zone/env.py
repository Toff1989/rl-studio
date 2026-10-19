"""Pompe a chaleur air/eau sur une zone de batiment (chauffage), modele thermique RC a deux noeuds.

Etat physique : temperature de l'air interieur Ti et de la masse du batiment Tm (murs, dalle), meteo exterieure
(sinusoide journaliere + bruit lent), heure du jour. Pas de temps : 5 minutes.
Action : modulation du compresseur dans [0, 1] (1 = puissance nominale a 7 degC). En dessous de `min_modulation`,
le compresseur est a l'arret. Capacite et COP dependent de la temperature exterieure et de la charge partielle.
Recompense par pas : -(ecart de confort au-dela de +/- 0,5 K)^2 - `energy_weight` * energie electrique [kWh].
Consigne : 21 degC de 6 h a 22 h, 17 degC la nuit (reduit). Episode de `max_steps` pas (2 jours par defaut).

Contrat d'un projet Python : `make_env(**params)` renvoie un gymnasium.Env ; `params` vient de `env.params`
dans project.yaml. Optionnel : `baseline_policy(env)` -> fonction obs -> action, tracee comme reference.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DT = 300.0                      # pas de temps [s]
C_AIR, C_MASS = 3.0e6, 5.0e7    # capacites thermiques de l'air + mobilier et de la masse du batiment [J/K]
UA_IM, UA_IE, UA_ME = 400.0, 60.0, 90.0   # air<->masse, air<->exterieur (vitrages, renouvellement), masse<->exterieur [W/K]
P_NOM = 8000.0                  # puissance thermique nominale a 7 degC [W]
T_SUPPLY = 35.0                 # temperature d'eau de depart (plancher chauffant) [degC]
ETA_CARNOT = 0.45               # rendement par rapport au COP de Carnot
SET_DAY, SET_NIGHT = 21.0, 17.0


def setpoint(hour):
    return SET_DAY if 6.0 <= hour < 22.0 else SET_NIGHT


def heat_pump(modulation, t_ext, min_modulation=0.2):
    """(puissance thermique [W], puissance electrique [W]) pour une modulation donnee."""
    u = float(np.clip(modulation, 0.0, 1.0))
    if u < min_modulation:
        return 0.0, 0.0
    capacity = max(0.4, 1.0 + 0.03 * (t_ext - 7.0))                  # la capacite chute quand il fait froid
    cop_full = float(np.clip(ETA_CARNOT * (T_SUPPLY + 273.15) / max(T_SUPPLY - t_ext, 5.0), 1.5, 6.0))
    cop = cop_full * (1.15 - 0.30 * u)                               # meilleur rendement en charge partielle
    q = u * P_NOM * capacity
    return q, q / cop


def step_thermal(ti, tm, t_ext, q_heat, q_gain, dt=DT):
    """Un pas d'Euler du modele RC (Ti : air, Tm : masse)."""
    d_ti = (q_heat + q_gain - UA_IM * (ti - tm) - UA_IE * (ti - t_ext)) / C_AIR
    d_tm = (UA_IM * (ti - tm) - UA_ME * (tm - t_ext)) / C_MASS
    return ti + dt * d_ti, tm + dt * d_tm


class PacZoneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=576, energy_weight=3.0, min_modulation=0.2):
        self.max_steps, self.energy_weight, self.min_modulation = int(max_steps), float(energy_weight), float(min_modulation)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.obs_names = ["ecart_air", "ecart_masse", "T_ext", "consigne", "h_sin", "h_cos"]
        self.action_names = ["modulation"]

    def _hour(self):
        return (self.hour0 + self.k * DT / 3600.0) % 24.0

    def _t_ext(self):
        return self.t_mean + self.t_amp * math.cos(2 * math.pi * (self._hour() - 15.0) / 24.0) + self.drift

    def _obs(self):
        h, sp = self._hour(), setpoint(self._hour())
        return np.array([self.ti - sp, self.tm - sp, self._t_ext(), sp, math.sin(2 * math.pi * h / 24.0),
                         math.cos(2 * math.pi * h / 24.0)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t_mean = float(self.np_random.uniform(-5.0, 10.0))
        self.t_amp = float(self.np_random.uniform(2.0, 5.0))
        self.sun = float(self.np_random.uniform(0.2, 1.0))     # nebulosite du jour
        self.hour0 = float(self.np_random.uniform(0.0, 24.0))
        self.drift, self.k = 0.0, 0
        sp = setpoint(self.hour0)
        self.ti = sp + float(self.np_random.uniform(-1.5, 1.0))
        self.tm = self.ti + float(self.np_random.uniform(-1.0, 1.0))
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        h, t_ext = self._hour(), self._t_ext()
        q_heat, p_el = heat_pump(u, t_ext, self.min_modulation)
        gains = (250.0 if 6.0 <= h < 22.0 else 80.0) + 600.0 * self.sun * max(0.0, math.sin(math.pi * (h - 7.0) / 10.0))
        self.ti, self.tm = step_thermal(self.ti, self.tm, t_ext, q_heat, gains)
        self.drift = 0.98 * self.drift + float(self.np_random.normal(0.0, 0.15))
        self.k += 1
        dev = max(0.0, abs(self.ti - setpoint(self._hour())) - 0.5)
        energy = p_el * DT / 3.6e6
        reward = -(dev ** 2) - self.energy_weight * energy
        return self._obs(), float(reward), False, self.k >= self.max_steps, {"energy_kwh": energy}


def make_env(**params):
    return PacZoneEnv(**params)


def baseline_policy(env):
    """Thermostat proportionnel sur l'ecart de temperature de l'air (sans anticipation : reference de comparaison)."""
    def policy(obs):
        return np.array([np.clip(0.05 - 0.6 * obs[0], 0.0, 1.0)], dtype=np.float32)
    return policy
