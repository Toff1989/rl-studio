"""Modele entrees -> sorties en REGIME PERMANENT (carte statique y = f(u)), sans dynamique.

Plan d'expérience (hypercube latin sur les entrees) -> chaque point est simulé jusqu'a convergence -> sortie stabilisee relevee
-> ajustement (lineaire, polynome, equation parcimonieuse, reseau) -> validation sur des points jamais vus -> rapport et code exporte.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..distill.surrogates import r_squared
from ..envs.loader import make_env
from . import collect, models
from .pipeline import METHOD_LABELS, METHODS, _noop, identify_dir, write_candidate, write_report  # noqa: F401


@dataclass
class SteadyOptions:
    methods: tuple = METHODS
    samples: int = 200            # points de regime permanent pour l'ajustement
    val_samples: int = 50         # points neufs pour la validation
    extreme: float = 0.15         # part des coordonnees d'entree prises aux bornes
    rtol: float = 1e-3            # un point est stabilise quand |dy| <= atol + rtol.|y| pendant `window` pas
    atol: float = 1e-6
    window: int = 3
    max_settle_steps: int | None = None   # defaut : duree d'un episode de l'environnement
    scenario: bool = False        # les variables tirees au reset (perturbations, cle `initial`) deviennent des entrees du modele
    poly_degrees: tuple = (2, 3)
    sparse_thresholds: tuple = (0.3, 0.1, 0.03)
    hidden: tuple = (32, 32)
    epochs: int = 400
    margin: float = 0.02          # un modele est « suffisant » si son R2 sur points neufs atteint 1 - margin
    seed: int = 0
    ranges: dict = field(default_factory=dict)    # {entree: [min, max]} : plage balayee (defaut : bornes de l'action de l'environnement)
    fixed: dict = field(default_factory=dict)     # {entree: valeur} : entree maintenue constante (non balayee, exclue du modele)
    outputs: list | None = None                   # sorties a modeliser (defaut : toutes les observations)

    @staticmethod
    def from_dict(d):
        base = SteadyOptions()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, tuple(v) if isinstance(getattr(base, k), tuple) and v is not None else v)
        return base


def evaluate_static(model, val):
    U, Y = val.Z(), val.delta()
    P = model.predict(U)
    sd = np.maximum(Y.std(0), 1e-12)
    err = (P - Y) / sd
    per_out = [float(r_squared(Y[:, [k]], P[:, [k]])) for k in range(Y.shape[1])]
    pick = np.random.default_rng(0).choice(len(Y), min(len(Y), 200), replace=False)
    return {"r2": float(r_squared(Y, P)), "r2_by_output": dict(zip(val.obs_names, per_out)), "nrmse": float(np.sqrt((err ** 2).mean())),
            "max_abs_err": float(np.abs(err).max()),
            "scatter": {n: np.column_stack([Y[pick, k], P[pick, k]]).round(6).tolist() for k, n in enumerate(val.obs_names)}}


def run_steady(project, options: SteadyOptions | None = None, log=print, progress=None, should_stop=None, env=None) -> dict:
    opts = options or SteadyOptions()
    unknown = [m for m in opts.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"méthodes inconnues : {unknown} (disponibles : {METHODS})")
    own = env is None
    env = env or make_env(project)
    try:
        return steady_env(env, opts, name=getattr(project, "name", "projet"), out_dir=identify_dir(project) / time.strftime("%Y%m%d-%H%M%S"),
                          log=log, progress=progress, should_stop=should_stop)
    finally:
        if own:
            env.close()


def steady_env(env, opts, name="env", out_dir=None, log=print, progress=None, should_stop=None) -> dict:
    progress = progress or _noop
    rng = np.random.default_rng(opts.seed)
    t0 = time.time()
    env_low, env_high = np.asarray(env.action_space.low, float), np.asarray(env.action_space.high, float)
    names_a = list(getattr(env, "action_names", None) or [f"u{i}" for i in range(len(env_low))])
    unknown = [n for n in list(opts.fixed) + list(opts.ranges) if n not in names_a]
    if unknown:
        raise ValueError(f"entrées inconnues : {unknown} (actions disponibles : {names_a})")
    swept = [i for i, n in enumerate(names_a) if n not in opts.fixed]
    if not swept:
        raise ValueError("Toutes les entrées sont fixées : il n'y a rien à balayer.")
    low, high = env_low[swept].copy(), env_high[swept].copy()
    for j, i in enumerate(swept):
        if names_a[i] in opts.ranges:
            a, b = (float(v) for v in opts.ranges[names_a[i]])
            if not a < b:
                raise ValueError(f"plage de {names_a[i]} : min doit être < max ({a} >= {b}).")
            low[j], high[j] = max(a, env_low[i]), min(b, env_high[i])
            if not low[j] < high[j]:
                raise ValueError(f"plage de {names_a[i]} hors des bornes de l'action [{env_low[i]:g}, {env_high[i]:g}].")
            if (a, b) != (low[j], high[j]):
                log(f"  plage de {names_a[i]} ramenée aux bornes de l'action : [{low[j]:g}, {high[j]:g}]")
    base = (env_low + env_high) / 2.0
    for n, v in opts.fixed.items():
        i = names_a.index(n)
        base[i] = float(np.clip(float(v), env_low[i], env_high[i]))
    n_fits = sum({"polynomial": len(opts.poly_degrees), "sparse": len(opts.sparse_thresholds)}.get(m, 1) for m in opts.methods)
    total, done = opts.samples + opts.val_samples + n_fits, [0]

    def bump(stage):
        done[0] += 1
        progress(stage, min(0.999, done[0] / total))

    kw = dict(rtol=opts.rtol, atol=opts.atol, window=opts.window, max_steps=opts.max_settle_steps, scenario=opts.scenario, should_stop=should_stop, swept=swept, base=base)
    log(f"Régime permanent : {opts.samples} point(s) d'apprentissage + {opts.val_samples} de validation (hypercube latin sur {len(low)} entrée(s) : {', '.join(names_a[i] for i in swept)}) ...")
    train = collect.steady_sweep(env, collect.latin_hypercube(rng, opts.samples, low, high, opts.extreme), 100 + opts.seed,
                                 on_point=lambda k, n: bump(f"Points d'apprentissage : {k}/{n}"), **kw)
    val = collect.steady_sweep(env, rng.uniform(low, high, (opts.val_samples, len(low))), 900_000 + opts.seed,
                               on_point=lambda k, n: bump(f"Points de validation : {k}/{n}"), **kw)
    if opts.outputs:
        bad = [n for n in opts.outputs if n not in train.obs_names]
        if bad:
            raise ValueError(f"sorties inconnues : {bad} (observations disponibles : {train.obs_names})")
        train.select_outputs(opts.outputs)
        val.select_outputs(opts.outputs)
    dropped = train.dropped + val.dropped
    log(f"  {len(train)} + {len(val)} points stabilisés (en moyenne {np.mean(train.steps + val.steps):.1f} pas), {dropped} écarté(s) faute de convergence")
    if len(train) < 5 or len(val) < 2:
        raise ValueError("Trop peu de points stabilisés : augmenter la durée max (`max_settle_steps`) ou assouplir `rtol` / `atol`.")

    fits = []
    for m in opts.methods:
        if m == "linear":
            fits.append(("linear", m, lambda: models.fit_linear(train)))
        elif m == "polynomial":
            fits += [(f"poly-d{k}", m, lambda k=k: models.fit_polynomial(train, degree=k)) for k in opts.poly_degrees]
        elif m == "sparse":
            fits += [(f"sparse-t{t:g}", m, lambda t=t: models.fit_sparse(train, t)) for t in opts.sparse_thresholds]
        else:
            fits.append(("neural", m, lambda: models.fit_neural(train, hidden=opts.hidden, epochs=opts.epochs, seed=opts.seed,
                                                                should_stop=should_stop)))
    out_dir = Path(out_dir) if out_dir else None
    candidates = []
    for cid, family, build in fits:
        log(f"== {METHOD_LABELS[family]} ({cid})")
        model = build()
        res = evaluate_static(model, val)
        entry = {"id": cid, "family": family, "label": model.title(), "complexity": int(model.complexity), **res, "equations": model.equations()[:80]}
        if out_dir:
            entry["files"] = write_candidate(out_dir / cid, model)
        candidates.append(entry)
        bump(f"Ajustement : {cid}")
        log(f"  complexité {entry['complexity']:6d}  R2 {res['r2']:7.4f}  erreur max {res['max_abs_err']:.2f} écart-type")

    ok = [c for c in candidates if c["r2"] >= 1.0 - opts.margin]
    recommended = (min(ok, key=lambda c: c["complexity"]) if ok else max(candidates, key=lambda c: c["r2"]))["id"]
    notes = []
    if not ok:
        notes.append(f"Aucun modèle n'atteint R2 = {1 - opts.margin:.0%} : augmenter le nombre de points, ou la complexité (degré, réseau).")
    if dropped > 0.1 * (opts.samples + opts.val_samples):
        notes.append(f"{dropped} point(s) n'ont pas convergé (instabilité, oscillation entretenue ou durée trop courte) : le modèle ne les représente pas.")
    if max(train.steps) >= (opts.max_settle_steps or 10 ** 9) * 0.9:
        notes.append("Certains points convergent à la limite de durée : augmenter `max_settle_steps` pour éviter des sorties non stabilisées.")
    notes.append("Le modèle n'est valable que dans le domaine des entrées balayées (les polynômes et le réseau extrapolent mal au-dela).")
    notes.append("Si plusieurs régimes permanents existent pour une même entrée (hystérésis, bistabilité), la sortie dépend de l'histoire : "
                 "le modèle statique en donne une moyenne. Ajouter l'état en cause comme entrée.")
    report = {"mode": "steady", "version": __version__, "created": time.strftime("%Y-%m-%d %H:%M"), "project": name, "out_names": train.obs_names,
              "in_names": train.act_names, "options": asdict(opts), "n_train": len(train), "n_val": len(val), "dropped": dropped,
              "mean_settle_steps": float(np.mean(train.steps)), "candidates": candidates, "recommended": recommended, "notes": notes,
              "duration_s": round(time.time() - t0, 1)}
    if out_dir:
        report["dir"] = str(out_dir)
        write_report(report, out_dir)
        log(f"Rapport : {out_dir / 'report.json'}")
    progress("Terminé", 1.0)
    return report
