"""La politique apprise (« l'expert ») : chargement d'un run Stable-Baselines3 et extraction exacte de son reseau en NumPy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .surrogates import NeuralSurrogate


class Expert:
    """Politique SB3 deterministe (pour SAC : l'action moyenne, sans bruit d'exploration)."""

    def __init__(self, model, algo, obs_names, act_names, low, high):
        self.model, self.algo = model, algo
        self.obs_names, self.act_names = list(obs_names), list(act_names)
        self.low, self.high = np.asarray(low, float).ravel(), np.asarray(high, float).ravel()

    def act(self, obs) -> np.ndarray:
        a, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=True)
        return np.asarray(a, float).reshape(-1)

    def act_batch(self, obs) -> np.ndarray:
        a, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=True)
        return np.asarray(a, float).reshape(len(obs), -1)

    def neural(self, atol=1e-3) -> NeuralSurrogate:
        """Copie NumPy exacte du reseau ; verifiee numeriquement contre `model.predict`."""
        sur = extract_neural(self.model, self.algo, self.obs_names, self.act_names, self.low, self.high)
        rng = np.random.default_rng(0)
        probe = rng.normal(size=(128, len(self.obs_names)))
        err = float(np.max(np.abs(sur.predict(probe) - self.act_batch(probe))))
        tol = atol * max(1.0, float(np.max(self.high - self.low)))
        if err > tol:
            raise ValueError(f"l'extraction du réseau ne reproduit pas model.predict (écart max {err:.3g})")
        return sur


def extract_neural(model, algo, obs_names, act_names, low, high) -> NeuralSurrogate:
    import torch.nn as nn

    pol = model.policy
    squash = algo in ("SAC", "TD3")
    if algo == "SAC":
        modules = list(pol.actor.latent_pi) + [pol.actor.mu]
    elif algo == "TD3":
        modules = list(pol.actor.mu)
    elif algo == "PPO":
        modules = list(pol.mlp_extractor.policy_net) + [pol.action_net]
    else:
        raise ValueError(f"algorithme non géré pour l'extraction exacte : {algo}")

    layers = []
    for i, mod in enumerate(modules):
        if isinstance(mod, nn.Linear):
            layers.append([mod.weight.detach().cpu().numpy().astype(float), mod.bias.detach().cpu().numpy().astype(float), "identity"])
        elif isinstance(mod, nn.ReLU):
            layers[-1][2] = "relu"
        elif isinstance(mod, nn.Tanh):
            if algo == "TD3" and i == len(modules) - 1:
                continue  # le tanh final de TD3 est la mise a l'echelle de la sortie (squash)
            layers[-1][2] = "tanh"
        elif isinstance(mod, (nn.Flatten, nn.Identity, nn.Dropout)):
            continue
        else:
            raise ValueError(f"couche non gérée pour l'extraction exacte : {type(mod).__name__}")
    return NeuralSurrogate(obs_names, act_names, low, high, layers, squash)


def run_algo(run_dir: Path, project_algo: str | None = None) -> str:
    """Algorithme d'un run : lu dans la copie de project.yaml enregistree avec le run."""
    cfg = Path(run_dir) / "project.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8-sig")) or {}
        return (data.get("algo") or {}).get("name") or project_algo or "SAC"
    return project_algo or "SAC"


def load_expert(project, run_dir, env) -> Expert:
    """Charge `model.zip` d'un run ; l'environnement donne les noms et les bornes des actions."""
    from stable_baselines3 import PPO, SAC, TD3

    run_dir = Path(run_dir)
    zip_path = run_dir / "model.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Pas de modèle dans ce run : {zip_path}")
    algo = run_algo(run_dir, project.cfg["algo"]["name"])
    cls = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[algo]
    space = env.action_space
    if not hasattr(space, "low"):
        raise ValueError("L'extraction ne gère que les actions continues (Box).")
    obs_names = list(getattr(env, "obs_names", None) or [f"obs{i}" for i in range(int(np.prod(env.observation_space.shape)))])
    act_names = list(getattr(env, "action_names", None) or [f"action{i}" for i in range(int(np.prod(space.shape)))])
    model = cls.load(str(zip_path), device="cpu")
    return Expert(model, algo, obs_names, act_names, space.low, space.high)
