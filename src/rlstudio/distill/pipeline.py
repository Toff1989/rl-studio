"""Extraction d'un modele mathematique de la politique d'un run : collecte, DAgger, evaluation, rapport, exports."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..envs.loader import make_env
from . import collect, export
from .policy import load_expert
from .surrogates import (NeuralSurrogate, Surrogate, fit_linear, fit_polynomial, fit_symbolic, fit_table, fit_tree, r_squared)

METHODS = ("linear", "polynomial", "table", "tree", "symbolic", "neural")
METHOD_LABELS = {"linear": "Linéaire / PID", "polynomial": "Polynôme", "table": "Table de gains", "tree": "Arbre de décision",
                 "symbolic": "Équation symbolique", "neural": "Réseau exact"}


@dataclass
class DistillOptions:
    methods: tuple = METHODS
    episodes: int = 20            # episodes de la politique apprise, avec bruit d'action
    noise: float = 0.1            # ecart-type du bruit d'action, en fraction de la demi-etendue de l'action
    dagger_iters: int = 3         # tours de DAgger (0 : pas de DAgger)
    dagger_episodes: int = 10     # episodes par tour de DAgger
    eval_episodes: int = 10       # episodes d'evaluation en boucle fermee, par modele
    margin: float = 0.05          # un modele est « suffisant » s'il atteint (1 - margin) du score de la politique apprise
    tree_depths: tuple = (3, 5, 8)
    poly_degrees: tuple = (2, 3)
    symbolic_terms: tuple = (3, 6, 12)
    table_dims: int = 2
    integrators: tuple = ()       # observations à intégrer (action integrale d'un PID)
    integral_dt: float | None = None
    seed: int = 0
    max_steps: int | None = None

    @staticmethod
    def from_dict(d: dict | None) -> "DistillOptions":
        base = DistillOptions()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, tuple(v) if isinstance(getattr(base, k), tuple) and v is not None else v)
        return base


def _noop(*_a, **_k):
    return None


def _normalized(ret, rand, expert):
    denom = expert - rand
    if abs(denom) < 1e-9:
        return 1.0 if ret >= expert - 1e-9 else 0.0
    return float((ret - rand) / denom)


def run_distillation(project, run_dir, options: DistillOptions | None = None, log=print, progress=None,
                     should_stop=None, env=None) -> dict:
    opts = options or DistillOptions()
    unknown = [m for m in opts.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"méthodes inconnues : {unknown} (disponibles : {METHODS})")
    run_dir = Path(run_dir)
    progress = progress or _noop
    own_env = env is None
    env = env or make_env(project)
    try:
        return _run(project, run_dir, opts, env, log, progress, should_stop)
    finally:
        if own_env:
            env.close()


def _run(project, run_dir, opts, env, log, progress, should_stop):
    rng = np.random.default_rng(opts.seed)
    expert = load_expert(project, run_dir, env)
    n_obs, n_act = len(expert.obs_names), len(expert.act_names)
    log(f"Politique chargée : {expert.algo}, {n_obs} observations, {n_act} action(s).")

    families = [m for m in opts.methods]
    n_final = {"linear": 1, "polynomial": len(opts.poly_degrees), "table": 1, "tree": len(opts.tree_depths), "symbolic": len(opts.symbolic_terms), "neural": 1}
    n_rand = min(opts.eval_episodes, 5)
    # unite de travail = un episode simule (c'est ce qui prend du temps) ; les ajustements sont quasi instantanes
    total = n_rand + opts.eval_episodes + opts.episodes + sum(
        (0 if f == "neural" else opts.dagger_iters * opts.dagger_episodes) + n_final[f] * opts.eval_episodes for f in families)
    done = [0]
    total = max(total, 1)

    def bump(stage):
        done[0] += 1
        progress(stage, min(0.999, done[0] / total))

    def stage_cb(text):
        return lambda k, n: bump(f"{text} : épisode {k}/{n}")

    eval_seed0 = 10_000 + opts.seed
    t0 = time.time()
    # -- references : politique aleatoire et politique apprise ----------------------------------
    log("Évaluation de la politique aléatoire (référence basse) ...")
    random_res = collect.evaluate(env, lambda o: env.action_space.sample(), n_rand, 20_000 + opts.seed, should_stop=should_stop, max_steps=opts.max_steps,
                                  on_episode=stage_cb("Référence aléatoire"))
    log("Évaluation de la politique apprise ...")
    exp_res = collect.evaluate(env, expert.act, opts.eval_episodes, eval_seed0, expert=expert, should_stop=should_stop, max_steps=opts.max_steps,
                           on_episode=stage_cb("Évaluation de la politique apprise"))
    held_obs, held_labels = exp_res["obs_episodes"], exp_res["expert_labels"]
    rand_ret, exp_ret = random_res["return_mean"], exp_res["return_mean"]
    log(f"  retour : aléatoire {rand_ret:.3f}  |  politique apprise {exp_ret:.3f} +/- {exp_res['return_std']:.3f}")

    log(f"Collecte de {opts.episodes} épisodes de la politique apprise (bruit d'action {opts.noise:.0%}) ...")
    base = collect.collect_expert(env, expert, opts.episodes, opts.noise, seed0=0 + opts.seed, rng=rng, should_stop=should_stop, max_steps=opts.max_steps,
                                 on_episode=stage_cb("Collecte de données"))
    log(f"  {len(base)} paires (état -> action)")

    dt = opts.integral_dt or float(getattr(env, "step_s", 1.0) or 1.0)
    mid = lambda seq: seq[len(seq) // 2]  # noqa: E731

    def leaf(n):
        return max(5, n // 400)

    fit_for_dagger = {
        "linear": lambda d: fit_linear(d, integrators=opts.integrators, dt=dt),
        "polynomial": lambda d: fit_polynomial(d, degree=mid(opts.poly_degrees)),
        "table": lambda d: fit_table(d, k_dims=opts.table_dims),
        "tree": lambda d: fit_tree(d, max_depth=mid(opts.tree_depths), min_leaf=leaf(len(d))),
        "symbolic": lambda d: fit_symbolic(d, (mid(opts.symbolic_terms),))[mid(opts.symbolic_terms)],
    }
    finals = {
        "linear": lambda d: [("linear", fit_linear(d, integrators=opts.integrators, dt=dt))],
        "polynomial": lambda d: [(f"poly-d{k}", fit_polynomial(d, degree=k)) for k in opts.poly_degrees],
        "table": lambda d: [("table", fit_table(d, k_dims=opts.table_dims))],
        "tree": lambda d: [(f"tree-d{k}", fit_tree(d, max_depth=k, min_leaf=leaf(len(d)))) for k in opts.tree_depths],
        "symbolic": lambda d: [(f"symbolic-k{k}", s) for k, s in fit_symbolic(d, opts.symbolic_terms).items()],
    }

    rng_sample = np.random.default_rng(123)
    candidates = []

    def score(cand_id, family, sur: Surrogate):
        res = collect.evaluate(env, sur.act, opts.eval_episodes, eval_seed0, reset=sur.reset, expert=expert, should_stop=should_stop, max_steps=opts.max_steps,
                           on_episode=stage_cb(f"Évaluation en boucle fermée de {cand_id}"))
        pred = sur.predict_episodes(held_obs)
        r2_exp = r_squared(held_labels, pred)
        pairs = np.column_stack([held_labels[:, 0], pred[:, 0]])
        if len(pairs) > 300:
            pairs = pairs[rng_sample.choice(len(pairs), 300, replace=False)]
        eqs = sur.equations()
        entry = {"id": cand_id, "family": family, "label": sur.title(), "complexity": int(sur.complexity),
                 "r2_expert": float(r2_exp), "r2_visited": float(res["r2_visited"]), "return_mean": res["return_mean"],
                 "return_std": res["return_std"], "score": _normalized(res["return_mean"], rand_ret, exp_ret),
                 "completed_rate": res["completed_rate"], "length_mean": res["length_mean"],
                 "stateful": bool(sur.stateful), "scatter": pairs.round(5).tolist(),
                 "equations": eqs[:60] + (["..."] if len(eqs) > 60 else [])}
        entry["files"] = export.write_candidate(export.distill_dir(run_dir) / cand_id, sur)
        candidates.append(entry)
        log(f"  {cand_id:14s} complexite {entry['complexity']:6d}  R2 {r2_exp:6.3f}  retour {entry['return_mean']:9.3f}  score {entry['score']:6.3f}")

    for family in families:
        if should_stop and should_stop():
            raise collect.Stopped()
        log(f"== {METHOD_LABELS[family]}")
        if family == "neural":
            log("  extraction du réseau ...")
            score("neural", family, expert.neural())
            continue
        data = base
        if opts.dagger_iters > 0:
            log(f"  DAgger : {opts.dagger_iters} tour(s) de {opts.dagger_episodes} episode(s)")
            data, _ = collect.run_dagger(env, expert, fit_for_dagger[family], base, opts.dagger_iters, opts.dagger_episodes,
                                         seed0=500 + opts.seed, rng=rng, should_stop=should_stop,
                                         tick=lambda f=family: log(f"  tour de DAgger terminé, modèle {METHOD_LABELS[f]} réajusté"),
                                         max_steps=opts.max_steps,
                                         on_episode=lambda it, k, n, f=family: bump(f"DAgger {METHOD_LABELS[f]} : tour {it}/{opts.dagger_iters}, épisode {k}/{n}"))
        log("  ajustement final des modèles ...")
        for cand_id, sur in finals[family](data):
            score(cand_id, family, sur)

    # -- recommandation : le plus simple parmi ceux qui atteignent (1 - marge) du score de la politique apprise ------
    ok = [c for c in candidates if c["score"] >= 1.0 - opts.margin]
    recommended = (min(ok, key=lambda c: c["complexity"]) if ok else max(candidates, key=lambda c: c["score"]))["id"]

    sat = float(np.mean((np.abs(held_labels - expert.low) < 0.01 * (expert.high - expert.low)) | (np.abs(held_labels - expert.high) < 0.01 * (expert.high - expert.low))))
    notes = []
    if exp_res["completed_rate"] < 1.0:
        notes.append(f"La politique apprise interrompt elle-même {100 * (1 - exp_res['completed_rate']):.0f} % de ses épisodes : le score de référence est relatif à ce comportement.")
    if sat > 0.2:
        notes.append(f"{sat:.0%} des actions de la politique apprise sont à la borne : un modèle lisse peut mal reproduire ces paliers (préférer l'arbre).")
    if opts.dagger_iters == 0:
        notes.append("DAgger désactivé : les modèles sont ajustés sur les seuls états de la politique apprise et peuvent dériver en boucle fermée.")
    if any(c["stateful"] for c in candidates):
        notes.append("Un modèle à état (intégrale) doit être réinitialisé (reset / clear) au début de chaque épisode.")
    if not ok:
        notes.append(f"Aucun modèle n'atteint {(1 - opts.margin):.0%} du score de la politique apprise : augmenter les épisodes, les tours de DAgger ou la complexité.")
    notes.append("Hors des états visites pendant la collecte, les modèles simples extrapolent : valider en simulation avant tout usage.")
    if not ok or recommended != "neural":
        notes.append("La fidélité aux actions (R2) ne garantit pas la performance : se fier au score en boucle fermée.")

    report = {"version": __version__, "created": time.strftime("%Y-%m-%d %H:%M"), "project": project.name, "run": run_dir.name,
              "algo": expert.algo, "obs_names": expert.obs_names, "act_names": expert.act_names, "options": asdict(opts),
              "expert": {"return_mean": exp_ret, "return_std": exp_res["return_std"], "completed_rate": exp_res["completed_rate"], "saturation": sat},
              "random_return": rand_ret, "candidates": candidates, "recommended": recommended, "notes": notes,
              "duration_s": round(time.time() - t0, 1)}
    path = export.write_report(report, run_dir)
    log(f"Rapport : {path}")
    progress("Terminé", 1.0)
    return report
