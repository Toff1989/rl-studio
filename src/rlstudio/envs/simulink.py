"""Environnement Gymnasium generique pour un modele Simulink decrit dans project.yaml.

Spec attendue (cle `env` du projet) :

  model: model.slx            # chemin relatif au projet
  step_s: 60                  # duree simulee par pas de decision
  episode_s: 3600
  actions:                    # une entree = une composante de l'action
    - {name: g, block: "model/Gain", param: Gain, low: 0, high: 5}   # parametre d'un bloc
    - {name: u, variable: u_cmd, low: 0, high: 1}                    # ou variable de workspace
  observations:               # n'importe quel signal : sortie d'un bloc
    - {name: temp, block: "model/Plant", port: 1, size: 1, low: -inf, high: inf}
  initial:                    # variables de workspace tirees au hasard a chaque reset
    - {variable: x0, low: 0.2, high: 0.8}
  reward: "-(temp - 20)**2"   # variables : observations, actions, t, dt, step
  terminate: "temp > 100"     # optionnel

Les signaux observes sont journalises en memoire (le .slx n'est pas modifie).
L'etat du modele est conserve d'un pas a l'autre cote MATLAB (matlab/rlstudio_step.m).
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..paths import MATLAB_DIR, USER_DIR
from . import expressions
from .spec import validate_spec  # noqa: F401  (reexporte)


class MatlabSession:
    """Un moteur MATLAB par environnement (un process MATLAB par env pour paralleliser)."""

    def __init__(self):
        import matlab.engine

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(str(MATLAB_DIR), nargout=0)
        # Simulink cree un dossier `slprj` (cache de simulation, code genere) dans le dossier COURANT de MATLAB :
        # on y place le moteur dans un dossier de cache dedie, pour ne rien creer dans le depot ni chez l'utilisateur.
        cache = USER_DIR / "cache" / "simulink"
        cache.mkdir(parents=True, exist_ok=True)
        self.eng.cd(str(cache), nargout=0)

    def call(self, model_path, payload: str, t0: float, dt: float, reset: bool) -> str:
        return self.eng.rlstudio_step(str(model_path), payload, float(t0), float(dt), bool(reset), nargout=1)

    def inspect(self, model_path) -> list:
        """Blocs du modele avec leurs sorties et parametres numeriques modifiables."""
        return json.loads(self.eng.rlstudio_inspect(str(model_path), nargout=1))

    def analyze(self, model_path, init_path="") -> dict:
        """Entrees, sorties, constantes, solveur, variables et blocs a retirer d'un modele (conversion)."""
        return json.loads(self.eng.rlstudio_analyze(str(model_path), str(init_path or ""), nargout=1))

    def convert(self, model_path, out_path, spec: dict, init_path="") -> dict:
        """Enregistre une copie convertie du modele (voir matlab/rlstudio_convert.m) ; l'original n'est pas modifie."""
        return json.loads(self.eng.rlstudio_convert(str(model_path), str(out_path), json.dumps(spec), str(init_path or ""),
                                                    nargout=1))

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass


class SimulinkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, project_dir, spec: dict, session=None):
        errs = validate_spec(spec)
        if errs:
            raise ValueError("Spec Simulink invalide :\n- " + "\n- ".join(errs))
        self.sim_spec = spec
        self.model_path = (Path(project_dir) / spec["model"]).resolve()
        self.step_s = float(spec["step_s"])
        self.n_steps = int(round(float(spec["episode_s"]) / self.step_s))
        self.act_specs, self.obs_specs = spec["actions"], spec["observations"]
        self.action_names = [a["name"] for a in self.act_specs]
        self.obs_sizes = [int(o.get("size", 1)) for o in self.obs_specs]
        self.obs_names = [
            o["name"] if s == 1 else f"{o['name']}[{i}]"
            for o, s in zip(self.obs_specs, self.obs_sizes) for i in range(s)
        ]
        self.action_space = spaces.Box(
            np.array([float(a["low"]) for a in self.act_specs], dtype=np.float32),
            np.array([float(a["high"]) for a in self.act_specs], dtype=np.float32),
        )
        lo = np.concatenate([np.full(s, float(o.get("low", -np.inf))) for o, s in zip(self.obs_specs, self.obs_sizes)])
        hi = np.concatenate([np.full(s, float(o.get("high", np.inf))) for o, s in zip(self.obs_specs, self.obs_sizes)])
        self.observation_space = spaces.Box(lo.astype(np.float32), hi.astype(np.float32))
        self._session = session
        self._t = 0.0
        self._k = 0

    @property
    def session(self):
        if self._session is None:
            self._session = MatlabSession()
        return self._session

    # -- Gymnasium ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t, self._k = 0.0, 0
        self._init_vars = {
            i["variable"]: float(self.np_random.uniform(float(i["low"]), float(i["high"])))
            for i in self.sim_spec.get("initial", [])
        }
        obs = self._simulate({}, dt=0.0, reset=True)
        return obs, {"t": 0.0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), self.action_space.low, self.action_space.high)
        values = dict(zip(self.action_names, action.tolist()))
        obs = self._simulate(values, dt=self.step_s, reset=False)
        self._t += self.step_s
        self._k += 1
        ns = {**self._obs_dict(obs), **values, "t": self._t, "dt": self.step_s, "step": self._k}
        reward = float(np.sum(expressions.evaluate(self.sim_spec["reward"], ns)))
        terminated = bool(self.sim_spec.get("terminate") and expressions.evaluate(self.sim_spec["terminate"], ns))
        truncated = self._k >= self.n_steps
        return obs, reward, terminated, truncated, {"t": self._t}

    def close(self):
        if self._session is not None:
            self._session.close()

    # -- internes -----------------------------------------------------------------------
    def _obs_dict(self, obs):
        out, i = {}, 0
        for o, s in zip(self.obs_specs, self.obs_sizes):
            out[o["name"]] = float(obs[i]) if s == 1 else obs[i : i + s]
            i += s
        return out

    def _simulate(self, action_values: dict, dt: float, reset: bool):
        variables = dict(self._init_vars) if reset else {}
        blocks = []
        for a in self.act_specs:
            if a["name"] not in action_values:
                continue
            v = action_values[a["name"]]
            if a.get("variable"):
                variables[a["variable"]] = v
            else:
                blocks.append({"path": a["block"], "param": a["param"], "value": v})
        signals = [{"name": o["name"], "block": o["block"], "port": int(o.get("port", 1))} for o in self.obs_specs]
        payload = json.dumps({"variables": variables, "blocks": blocks, "signals": signals})
        reply = json.loads(self.session.call(self.model_path, payload, self._t, dt, reset))
        return self._parse_outputs(reply)

    def _parse_outputs(self, reply):
        parts = []
        for o, size in zip(self.obs_specs, self.obs_sizes):
            vec = np.atleast_1d(np.asarray(reply[o["name"]], dtype=float)).reshape(-1)
            if vec.size != size:
                raise ValueError(
                    f"Observation {o['name']!r} : le signal renvoie {vec.size} valeur(s), `size` vaut {size}."
                )
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)
