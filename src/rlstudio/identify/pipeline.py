"""Identification d'un modele reduit de la dynamique d'un environnement (typiquement un modele Simulink sans politique).

Balayage d'actions et de conditions initiales -> ajustement (lineaire, polynome, equation parcimonieuse, reseau)
-> validation en SIMULATION LIBRE sur des episodes jamais vus -> rapport et code exporte (Python, MATLAB).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..distill.collect import Stopped
from ..distill.surrogates import r_squared
from ..envs.loader import make_env
from . import collect, models

METHODS = ("linear", "polynomial", "sparse", "neural")
METHOD_LABELS = {"linear": "Lineaire", "polynomial": "Polynome", "sparse": "Equation parcimonieuse", "neural": "Reseau de neurones"}
HORIZONS = (1, 2, 5, 10, 20, 50, 100)


@dataclass
class IdentifyOptions:
    methods: tuple = METHODS
    episodes: int = 40            # episodes de balayage (apprentissage)
    val_episodes: int = 10        # episodes neufs, jamais utilises pour l'ajustement
    rounds: int = 0               # tours supplementaires de collecte (courbe d'apprentissage, donnees agregees)
    round_episodes: int = 20
    hold_max: int = 10            # duree maximale (en pas) d'un palier d'action
    extreme: float = 0.2          # part des paliers pris aux bornes de l'action
    poly_degrees: tuple = (2, 3)
    sparse_thresholds: tuple = (0.3, 0.1, 0.03)   # seuils de STLSQ, sur les coefficients normalises (plus grand = plus parcimonieux)
    hidden: tuple = (32, 32)
    epochs: int = 300
    margin: float = 0.05          # un modele est « suffisant » si sa fidelite en simulation libre atteint 1 - margin
    seed: int = 0
    max_steps: int | None = None
    ranges: dict = field(default_factory=dict)    # {action: [min, max]} : plage balayee
    fixed: dict = field(default_factory=dict)     # {action: valeur} : action maintenue constante

    @staticmethod
    def from_dict(d: dict | None) -> "IdentifyOptions":
        base = IdentifyOptions()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, tuple(v) if isinstance(getattr(base, k), tuple) and v is not None else v)
        return base


def identify_dir(project) -> Path:
    return Path(project.path) / "identify"


def _bounds(data):
    S = data.states()
    lo, hi = S.min(0), S.max(0)
    pad = 0.5 * np.maximum(hi - lo, 1e-9)
    return lo - pad, hi + pad


def evaluate(model, val, state_std, clip):
    """Fidelite sur des episodes neufs : un pas (sur l'increment) et simulation libre (sur l'etat), par horizon."""
    Z = np.concatenate([np.hstack([X[:-1], U]) for X, U in val.episodes])
    D = np.concatenate([X[1:] - X[:-1] for X, _ in val.episodes])
    r2_step = r_squared(D, model.delta(Z))
    sq, by_h, diverged = [], {h: [] for h in HORIZONS}, 0
    for X, U in val.episodes:
        with np.errstate(all="ignore"):
            P = model.rollout(X[0], U, clip=clip)
        bad = ~np.isfinite(P)
        if bad.any():
            diverged += 1
            P = np.where(bad, clip[1], P)
        e = ((P - X) / state_std) ** 2
        sq.append(e[1:])
        for h in HORIZONS:
            if h < len(X):
                by_h[h].append(e[h])
    e_all = np.concatenate(sq)
    return {"r2_one_step": float(r2_step), "free_r2": float(1.0 - e_all.mean()),
            "nrmse": float(np.sqrt(e_all.mean())),
            "nrmse_by_horizon": {str(h): float(np.sqrt(np.mean(np.concatenate(v)))) for h, v in by_h.items() if v},
            "diverged": diverged}


def _noop(*_a, **_k):
    return None


def run_identification(project, options: IdentifyOptions | None = None, log=print, progress=None, should_stop=None, env=None) -> dict:
    opts = options or IdentifyOptions()
    unknown = [m for m in opts.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"methodes inconnues : {unknown} (disponibles : {METHODS})")
    own = env is None
    env = env or make_env(project)
    try:
        return identify_env(env, opts, name=getattr(project, "name", "projet"), out_dir=identify_dir(project) / time.strftime("%Y%m%d-%H%M%S"),
                            log=log, progress=progress, should_stop=should_stop)
    finally:
        if own:
            env.close()


def identify_env(env, opts, name="env", out_dir=None, log=print, progress=None, should_stop=None, action_fn=None) -> dict:
    progress = progress or _noop
    rng = np.random.default_rng(opts.seed)
    t0 = time.time()
    n_fits = sum({"polynomial": len(opts.poly_degrees), "sparse": len(opts.sparse_thresholds)}.get(m, 1) for m in opts.methods)
    total = opts.episodes + opts.val_episodes + opts.rounds * opts.round_episodes + n_fits
    done = [0]

    def bump(stage, n=1):
        done[0] += n
        progress(stage, min(0.999, done[0] / max(total, 1)))

    env_low, env_high = np.asarray(env.action_space.low, float), np.asarray(env.action_space.high, float)
    names_a = list(getattr(env, "action_names", None) or [f"u{i}" for i in range(len(env_low))])
    unknown = [n for n in list(opts.fixed) + list(opts.ranges) if n not in names_a]
    if unknown:
        raise ValueError(f"actions inconnues : {unknown} (disponibles : {names_a})")
    low, high = env_low.copy(), env_high.copy()
    for i, n in enumerate(names_a):
        if n in opts.ranges:
            a, b = (float(v) for v in opts.ranges[n])
            low[i], high[i] = max(a, env_low[i]), min(b, env_high[i])
            if not low[i] < high[i]:
                raise ValueError(f"plage de {n} invalide ou hors des bornes de l'action [{env_low[i]:g}, {env_high[i]:g}].")
        if n in opts.fixed:
            low[i] = high[i] = float(np.clip(float(opts.fixed[n]), env_low[i], env_high[i]))
    sweep_kw = dict(hold_max=opts.hold_max, extreme=opts.extreme, max_steps=opts.max_steps, should_stop=should_stop, action_fn=action_fn, bounds=(low, high))
    log(f"Balayage : {opts.episodes} episode(s) d'apprentissage + {opts.val_episodes} de validation (actions variees, paliers de 1 a {opts.hold_max} pas) ...")
    train = collect.sweep(env, opts.episodes, 100 + opts.seed, rng, on_episode=lambda k, n: bump(f"Balayage : episode {k}/{n}"), **sweep_kw)
    val = collect.sweep(env, opts.val_episodes, 900_000 + opts.seed, rng, on_episode=lambda k, n: bump(f"Validation : episode {k}/{n}"), **sweep_kw)
    if not len(train) or not len(val):
        raise ValueError("Aucune transition collectee (episodes vides) : verifier l'environnement.")
    log(f"  {len(train)} transitions d'apprentissage, {len(val)} de validation ; pas de temps {train.dt:g} s")
    state_std = np.maximum(np.concatenate([val.states(), train.states()]).std(0), 1e-12)

    # -- courbe d'apprentissage : on mesure sur des episodes neufs, puis on les agrege ----------------------------
    curve = []
    for r in range(opts.rounds):
        fresh = collect.sweep(env, opts.round_episodes, 200_000 + 1000 * r + opts.seed, rng,
                              on_episode=lambda k, n, r=r: bump(f"Tour {r + 1} : episode {k}/{n}"), **sweep_kw)
        guide = models.fit_polynomial(train, degree=2)
        score = evaluate(guide, fresh, state_std, _bounds(train))["free_r2"]
        curve.append({"transitions": len(train), "free_r2_poly2": score})
        log(f"  tour {r + 1} : {len(train)} transitions -> fidelite du polynome (deg. 2) sur donnees neuves : {score:.3f}")
        train.extend(fresh)
    clip = _bounds(train)

    fits = []
    for m in opts.methods:
        if m == "linear":
            fits.append(("linear", m, lambda: models.fit_linear(train)))
        elif m == "polynomial":
            fits += [(f"poly-d{k}", m, lambda k=k: models.fit_polynomial(train, degree=k)) for k in opts.poly_degrees]
        elif m == "sparse":
            fits += [(f"sparse-t{t:g}", m, lambda t=t: models.fit_sparse(train, t)) for t in opts.sparse_thresholds]
        else:
            fits.append(("neural", m, lambda: models.fit_neural(train, hidden=opts.hidden, epochs=opts.epochs, seed=opts.seed, should_stop=should_stop)))
    out_dir = Path(out_dir) if out_dir else None
    candidates = []
    for cid, family, build in fits:
        if should_stop and should_stop():
            raise Stopped()
        log(f"== {METHOD_LABELS[family]} ({cid})")
        model = build()
        model.x0 = np.mean([X[0] for X, _ in train.episodes], axis=0)
        res = evaluate(model, val, state_std, clip)
        entry = {"id": cid, "family": family, "label": model.title(), "complexity": int(model.complexity), **res,
                 "equations": (model.equations()[:80])}
        if out_dir:
            entry["files"] = write_candidate(out_dir / cid, model)
        candidates.append(entry)
        bump(f"Ajustement : {cid}")
        log(f"  complexite {entry['complexity']:6d}  R2 un pas {res['r2_one_step']:7.4f}  fidelite en simulation libre {res['free_r2']:7.4f}"
            f"{'  (divergence sur ' + str(res['diverged']) + ' episode(s))' if res['diverged'] else ''}")

    ok = [c for c in candidates if c["free_r2"] >= 1.0 - opts.margin]
    recommended = (min(ok, key=lambda c: c["complexity"]) if ok else max(candidates, key=lambda c: c["free_r2"]))["id"]
    notes = _notes(candidates, ok, opts, train)
    report = {"version": __version__, "created": time.strftime("%Y-%m-%d %H:%M"), "project": name, "obs_names": train.obs_names,
              "act_names": train.act_names, "dt": train.dt, "options": asdict(opts), "n_train": len(train), "n_val": len(val),
              "learning_curve": curve, "candidates": candidates, "recommended": recommended, "notes": notes,
              "duration_s": round(time.time() - t0, 1)}
    if out_dir:
        report["dir"] = str(out_dir)
        write_report(report, out_dir)
        log(f"Rapport : {out_dir / 'report.json'}")
    progress("Termine", 1.0)
    return report


def _notes(candidates, ok, opts, train):
    notes = []
    best = max(candidates, key=lambda c: c["free_r2"])
    if not ok:
        notes.append(f"Aucun modele n'atteint {1 - opts.margin:.0%} de fidelite en simulation libre : augmenter les episodes ou les tours de collecte, "
                     f"ou elargir la famille de modeles.")
    if best["r2_one_step"] > 0.99 and best["free_r2"] < 0.8:
        notes.append("Bon a un pas mais mauvais en simulation libre : l'erreur s'accumule (modele instable) ou l'etat n'est pas completement observe "
                     "(etat cache dans le modele : integrateur, retard, thermique lente). Ajouter les signaux manquants aux observations.")
    if any(c["diverged"] for c in candidates):
        notes.append("Certains modeles divergent en simulation libre : leurs predictions sont ecretees ; ne pas les utiliser hors des plages balayees.")
    notes.append("Le modele n'est valable que dans les plages d'etat et d'action balayees : les etats atteignables par le systeme reel hors de ces plages "
                 "ne sont pas couverts (elargir les bornes d'action / conditions initiales de l'environnement).")
    notes.append("Les non-linearites tres fortes (saturations, frottements secs, commutations) sont lissees par les modeles polynomiaux ; "
                 "preferer le reseau, ou une bibliotheque de termes adaptee.")
    return notes


def write_candidate(dirpath: Path, model) -> dict:
    dirpath.mkdir(parents=True, exist_ok=True)
    stem = "steady_state" if model.static else "dynamics"
    (dirpath / f"{stem}.py").write_text(model.to_python(stem), encoding="utf-8")
    (dirpath / f"{stem}.m").write_text(model.to_matlab(stem), encoding="utf-8")
    (dirpath / "equations.txt").write_text(model.title() + "\n\n" + "\n".join(model.equations()) + "\n", encoding="utf-8")
    (dirpath / "model.json").write_text(json.dumps(model.to_dict()), encoding="utf-8")
    return {"python": f"{stem}.py", "matlab": f"{stem}.m", "equations": "equations.txt", "json": "model.json"}


def ensure_files(report_dir, cid) -> Path:
    """Dossier du modele `cid`, avec son code Python / MATLAB (regenere depuis model.json : un .rls n'embarque pas de code genere)."""
    d = Path(report_dir) / cid
    model = models.model_from_dict(json.loads((d / "model.json").read_text(encoding="utf-8")))
    stem = "steady_state" if model.static else "dynamics"
    if not (d / f"{stem}.py").exists() or not (d / f"{stem}.m").exists():
        write_candidate(d, model)
    return d


def load_candidate(report_dir, cid):
    return models.model_from_dict(json.loads((Path(report_dir) / cid / "model.json").read_text(encoding="utf-8")))


def report_markdown(r: dict) -> str:
    if r.get("mode") == "steady":
        return _steady_markdown(r)
    lines = [f"# Modele reduit identifie - {r['project']}", "",
             f"Genere le {r['created']} par RL Studio {r['version']} : {r['n_train']} transitions d'apprentissage, {r['n_val']} de validation, pas {r['dt']:g} s.", "",
             "Fidelite en simulation libre : 1 = trajectoire reproduite, 0 = aussi mauvais qu'une valeur constante (episodes jamais vus).", "",
             "| Modele | Complexite | R2 un pas | Fidelite sim. libre | NRMSE | |", "|---|---:|---:|---:|---:|---|"]
    for c in r["candidates"]:
        lines.append(f"| {c['label']} | {c['complexity']} | {c['r2_one_step']:.4f} | {c['free_r2']:.4f} | {c['nrmse']:.3f} | {'recommande' if c['id'] == r['recommended'] else ''} |")
    rec = next(c for c in r["candidates"] if c["id"] == r["recommended"])
    lines += ["", f"## Modele recommande : {rec['label']}", "", "```", *rec["equations"], "```", "",
              "Erreur normalisee (NRMSE) par horizon de prediction (en pas) : " + ", ".join(f"{h} : {v:.3f}" for h, v in rec["nrmse_by_horizon"].items())]
    if r["learning_curve"]:
        lines += ["", "## Courbe d'apprentissage (polynome degre 2, donnees neuves)", ""] + [f"- {p['transitions']} transitions : {p['free_r2_poly2']:.3f}" for p in r["learning_curve"]]
    return "\n".join(lines + ["", "## Points d'attention", ""] + [f"- {n}" for n in r["notes"]]) + "\n"


def write_report(report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")


def _steady_markdown(r: dict) -> str:
    lines = [f"# Modele entrees -> sorties en regime permanent - {r['project']}", "",
             f"Genere le {r['created']} par RL Studio {r['version']} : {r['n_train']} points d'apprentissage, {r['n_val']} de validation, "
             f"{r['dropped']} ecarte(s) ; stabilisation en {r['mean_settle_steps']:.1f} pas en moyenne.", "",
             f"Entrees : {', '.join(r['in_names'])} ; sorties : {', '.join(r['out_names'])}.", "",
             "R2 sur des points jamais vus : 1 = carte reproduite, 0 = aussi mauvais qu'une valeur moyenne.", "",
             "| Modele | Complexite | R2 | NRMSE | Erreur max | |", "|---|---:|---:|---:|---:|---|"]
    for c in r["candidates"]:
        lines.append(f"| {c['label']} | {c['complexity']} | {c['r2']:.4f} | {c['nrmse']:.3f} | {c['max_abs_err']:.2f} | {'recommande' if c['id'] == r['recommended'] else ''} |")
    rec = next(c for c in r["candidates"] if c["id"] == r["recommended"])
    lines += ["", f"## Modele recommande : {rec['label']}", "", "```", *rec["equations"], "```", "",
              "R2 par sortie : " + ", ".join(f"{k} : {v:.3f}" for k, v in rec["r2_by_output"].items())]
    return "\n".join(lines + ["", "## Points d'attention", ""] + [f"- {n}" for n in r["notes"]]) + "\n"
