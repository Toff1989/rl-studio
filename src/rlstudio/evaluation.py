"""Evaluation d'un run entraine (ou d'un modele extrait) : episodes rejoues pas a pas avec evenements en direct."""
import time

import numpy as np


def policy_choices(run_dir):
    """[(id, libelle)] : l'agent entraine, puis chaque modele mathematique extrait de ce run."""
    from .distill.export import load_report

    out = [("agent", "Agent entraine")]
    rep = load_report(run_dir)
    for c in (rep or {}).get("candidates", []):
        out.append((f"distill:{c['id']}", f"Modele extrait : {c['label']}"))
    return out


def run_evaluation(project, run_dir, policy_id="agent", episodes=5, seed=1000, delay_s=0.0, on_meta=None, on_step=None,
                   on_episode=None, should_stop=None, log=print, env=None):
    from .distill.export import load_candidate
    from .distill.policy import load_expert
    from .envs.loader import make_env

    own = env is None
    env = env or make_env(project)
    try:
        expert = load_expert(project, run_dir, env)
        if policy_id == "agent":
            act, reset = expert.act, None
        elif policy_id.startswith("distill:"):
            sur = load_candidate(run_dir, policy_id.split(":", 1)[1])
            act, reset = sur.act, sur.reset
        else:
            raise ValueError(f"politique inconnue : {policy_id}")
        if on_meta:
            on_meta({"obs_names": expert.obs_names, "action_names": expert.act_names})
        returns, lengths = [], []
        for n in range(1, int(episodes) + 1):
            if should_stop and should_stop():
                break
            obs, _ = env.reset(seed=int(seed) + n)
            if reset:
                reset()
            total, step, done, terminated = 0.0, 0, False, False
            while not done:
                if should_stop and should_stop():
                    break
                a = np.clip(np.asarray(act(obs), float).reshape(-1), expert.low, expert.high)
                nxt, r, terminated, truncated, _ = env.step(a.astype(np.float32))
                total += float(r)
                step += 1
                if on_step:
                    on_step({"episode": n, "step": step, "obs": np.asarray(obs, float).ravel(), "action": a,
                             "reward": float(r), "total": total})
                obs, done = nxt, bool(terminated or truncated)
                if delay_s > 0:
                    time.sleep(delay_s)
            else:
                returns.append(total)
                lengths.append(step)
                if on_episode:
                    on_episode({"n": n, "reward": total, "length": step, "terminated": bool(terminated)})
                continue
            break  # arret demande en cours d'episode
        summary = {"episodes": len(returns), "return_mean": float(np.mean(returns)) if returns else 0.0,
                   "return_std": float(np.std(returns)) if returns else 0.0,
                   "length_mean": float(np.mean(lengths)) if lengths else 0.0}
        log(f"Evaluation : {summary['episodes']} episode(s), retour moyen {summary['return_mean']:.3f} +/- {summary['return_std']:.3f}")
        return summary
    finally:
        if own:
            env.close()
