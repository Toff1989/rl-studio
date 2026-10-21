"""Evaluation d'un run entraîné (ou d'un modele extrait) : episodes rejoues pas à pas avec evenements en direct."""
import time
from pathlib import Path

import numpy as np


STEPS_POLICY = ("steps", "Paliers d'actions aléatoires (sans agent)")
MODEL_SEP = "|"


def policy_choices(run_dir=None):
    """[(id, libelle)] : l'agent entraine, chaque modele mathematique extrait de ce run, puis des paliers d'actions aleatoires
    (qui ne demandent ni entrainement ni extraction : de quoi comparer un modele identifie sur un projet neuf)."""
    from .distill.export import load_report

    out = []
    if run_dir is not None:
        out.append(("agent", "Agent entraîné"))
        rep = load_report(run_dir)
        for c in (rep or {}).get("candidates", []):
            out.append((f"distill:{c['id']}", f"Modèle extrait : {c['label']}"))
    return out + [STEPS_POLICY]


def identified_choices(project):
    """[(id, libelle)] : chaque modele identifie du projet (id = « dossier|modele », libelle lisible, le recommande en premier)."""
    import json

    base = Path(project.path) / "identify"
    out = []
    for d in sorted((d for d in base.iterdir() if (d / "report.json").exists()), reverse=True) if base.is_dir() else []:
        try:
            rep = json.loads((d / "report.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        mode = "régime permanent" if rep.get("mode") == "steady" else "dynamique"
        cands = sorted(rep.get("candidates", []), key=lambda c: c["id"] != rep.get("recommended"))
        for c in cands:
            star = " ★" if c["id"] == rep.get("recommended") else ""
            out.append((f"{d}{MODEL_SEP}{c['id']}", f"{d.name} · {mode} · {c['label']}{star}"))
    return out


class _ModelOverlay:
    """Un modele identifie rejoue a cote du systeme : sorties du modele pas a pas, a comparer aux observations reelles.

    Modele dynamique : simulation libre x(k+1) = x(k) + f(x(k), u(k)) depuis l'observation de depart, avec les actions du pilote.
    Modele statique : y = f(u) applique aux actions courantes (approximation quasi statique : il reagit sans delai,
    le vrai systeme met du temps a s'etablir)."""

    LIMIT = 1e12

    def __init__(self, model, obs_names, act_names, log=print):
        self.model, self.static = model, bool(model.static)
        missing = [n for n in model.obs_names if n not in obs_names]
        if missing:
            raise ValueError(f"le modèle identifié concerne des signaux absents de ce projet : {missing} (observations : {obs_names})")
        missing = [n for n in model.act_names if n not in act_names]
        if missing:
            raise ValueError(f"le modèle identifié concerne des actions absentes de ce projet : {missing} (actions : {act_names})")
        self.names = list(model.obs_names)
        self.obs_idx = [obs_names.index(n) for n in self.names]
        self.act_idx = [act_names.index(n) for n in model.act_names]
        unused = [n for n in act_names if n not in model.act_names]
        if unused:
            log(f"  entrée(s) fixée(s) lors de l'identification : {', '.join(unused)} ; le modèle les suppose à leur valeur d'origine.")
        self.reset(None)

    def reset(self, obs):
        self.x = None if obs is None else np.asarray(obs, float).ravel()[self.obs_idx]

    def values(self, obs, action):
        """Sorties du modele au pas courant (avant d'appliquer l'action) ; l'etat du modele avance ensuite via `advance`."""
        a = np.asarray(action, float).ravel()[self.act_idx]
        if self.static:
            return np.asarray(self.model.predict(a)[0], float)
        if self.x is None:
            self.x = np.asarray(obs, float).ravel()[self.obs_idx]
        return self.x.copy()

    def advance(self, action):
        if self.static or self.x is None:
            return
        a = np.asarray(action, float).ravel()[self.act_idx]
        nxt = self.model.step(self.x, a)
        self.x = nxt if np.all(np.isfinite(nxt)) and np.all(np.abs(nxt) < self.LIMIT) else np.full_like(nxt, np.nan)  # divergence


def compare_errors(names, real, model):
    """{signal: (rmse, rmse normalise)} sur un episode ; le normalise divise par l'ecart-type du signal reel (plancher : bruit numerique)."""
    out = {}
    real, model = np.asarray(real, float), np.asarray(model, float)
    for j, n in enumerate(names):
        ok = np.isfinite(model[:, j]) & np.isfinite(real[:, j])
        if not ok.any():
            out[n] = (float("nan"), float("nan"))
            continue
        r, m = real[ok, j], model[ok, j]
        rmse = float(np.sqrt(np.mean((m - r) ** 2)))
        scale = max(float(np.std(r)), 1e-3 * float(np.max(np.abs(r))), 1e-12)
        out[n] = (rmse, rmse / scale)
    return out


def run_evaluation(project, run_dir, policy_id="agent", episodes=5, seed=1000, delay_s=0.0, on_meta=None, on_step=None,
                   on_episode=None, should_stop=None, log=print, env=None, identified=None):
    """Rejoue des episodes. `identified=(dossier_du_rapport, id_du_modele)` : un modele identifie est rejoue a cote du systeme
    (sorties du modele dans l'evenement `step`, ecarts dans l'evenement `episode` et le resume)."""
    from .distill.export import load_candidate
    from .envs.loader import make_env
    from .identify.collect import action_sequence
    from .identify.pipeline import load_candidate as load_identified

    own = env is None
    env = env or make_env(project)
    try:
        obs_names = list(getattr(env, "obs_names", None) or [f"obs{i}" for i in range(int(np.prod(env.observation_space.shape)))])
        act_names = list(getattr(env, "action_names", None) or [f"action{i}" for i in range(int(np.prod(env.action_space.shape)))])
        low, high = np.asarray(env.action_space.low, float).ravel(), np.asarray(env.action_space.high, float).ravel()
        horizon = int(getattr(env, "n_steps", None) or getattr(env, "max_steps", None) or 200)
        script = None  # actions precalculees d'un episode (pilote « paliers »)
        reset = None
        if policy_id == "steps":
            rng = np.random.default_rng(int(seed))
            act = None
        else:
            from .distill.policy import load_expert

            expert = load_expert(project, run_dir, env)
            low, high = expert.low, expert.high
            if policy_id == "agent":
                act = expert.act
            elif policy_id.startswith("distill:"):
                sur = load_candidate(run_dir, policy_id.split(":", 1)[1])
                act, reset = sur.act, sur.reset
            else:
                raise ValueError(f"politique inconnue : {policy_id}")
        overlay = None
        if identified:
            overlay = _ModelOverlay(load_identified(*identified), obs_names, act_names, log)
        if on_meta:
            on_meta({"obs_names": obs_names, "action_names": act_names, "model_names": overlay.names if overlay else []})
        if overlay:
            how = ("régime permanent, y = f(u) appliqué aux actions courantes" if overlay.static
                   else "simulation libre depuis l'observation de départ")
            log(f"Modèle identifié rejoué à côté du système : {overlay.model.title()} ({how}).")
        returns, lengths, errors = [], [], []
        for n in range(1, int(episodes) + 1):
            if should_stop and should_stop():
                break
            obs, _ = env.reset(seed=int(seed) + n)
            if reset:
                reset()
            if policy_id == "steps":
                script = action_sequence(rng, low, high, horizon, hold_max=10, extreme=0.15)
            if overlay:
                overlay.reset(obs)
            real_rows, model_rows = [], []
            total, step, done, terminated = 0.0, 0, False, False
            while not done:
                if should_stop and should_stop():
                    break
                raw = script[min(step, len(script) - 1)] if script is not None else act(obs)
                a = np.clip(np.asarray(raw, float).reshape(-1), low, high)
                mvals = None
                if overlay:
                    mvals = overlay.values(obs, a)
                    real_rows.append(np.asarray(obs, float).ravel()[overlay.obs_idx])
                    model_rows.append(mvals)
                nxt, r, terminated, truncated, _ = env.step(a.astype(np.float32))
                if overlay:
                    overlay.advance(a)
                total += float(r)
                step += 1
                if on_step:
                    ev = {"episode": n, "step": step, "obs": np.asarray(obs, float).ravel(), "action": a,
                          "reward": float(r), "total": total}
                    if mvals is not None:
                        ev["model"] = mvals
                    on_step(ev)
                obs, done = nxt, bool(terminated or truncated)
                if delay_s > 0:
                    time.sleep(delay_s)
            else:
                returns.append(total)
                lengths.append(step)
                ep = {"n": n, "reward": total, "length": step, "terminated": bool(terminated)}
                if overlay and real_rows:
                    err = compare_errors(overlay.names, real_rows, model_rows)
                    errors.append(err)
                    ep["model_error"] = {k: {"rmse": v[0], "nrmse": v[1]} for k, v in err.items()}
                if on_episode:
                    on_episode(ep)
                continue
            break  # arret demande en cours d'episode
        summary = {"episodes": len(returns), "return_mean": float(np.mean(returns)) if returns else 0.0,
                   "return_std": float(np.std(returns)) if returns else 0.0,
                   "length_mean": float(np.mean(lengths)) if lengths else 0.0}
        if overlay and errors:
            summary["model_nrmse"] = {n: float(np.nanmean([e[n][1] for e in errors])) for n in overlay.names}
            summary["model_rmse"] = {n: float(np.nanmean([e[n][0] for e in errors])) for n in overlay.names}
            log("Écart entre le modèle identifié et le système (moyenne des épisodes ; RMSE, puis RMSE / écart-type du signal) :")
            for n in overlay.names:
                log(f"  {n} : {summary['model_rmse'][n]:.4g}  ({summary['model_nrmse'][n]:.3f})")
        log(f"Évaluation : {summary['episodes']} épisode(s), retour moyen {summary['return_mean']:.3f} +/- {summary['return_std']:.3f}")
        return summary
    finally:
        if own:
            env.close()
