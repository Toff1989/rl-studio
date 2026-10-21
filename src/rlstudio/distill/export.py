"""Ecriture des fichiers d'un modele extrait (Python, MATLAB, equations), rapport, et controleur Simulink."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..textutil import ascii_safe
from .surrogates import M_RESERVED, Surrogate, clean_names, surrogate_from_dict


def distill_dir(run_dir) -> Path:
    return Path(run_dir) / "distill"


def write_candidate(dirpath: Path, sur: Surrogate) -> dict:
    """policy.py, policy.m, equations.txt, surrogate.json dans `dirpath` ; renvoie les chemins relatifs."""
    dirpath.mkdir(parents=True, exist_ok=True)
    files = {"python": "policy.py", "matlab": "policy.m", "equations": "equations.txt", "json": "surrogate.json"}
    (dirpath / "policy.py").write_text(sur.to_python("policy"), encoding="utf-8")
    (dirpath / "policy.m").write_text(ascii_safe(sur.to_matlab("policy")), encoding="utf-8")
    (dirpath / "equations.txt").write_text(sur.title() + "\n\n" + "\n".join(sur.equations()) + "\n", encoding="utf-8")
    (dirpath / "surrogate.json").write_text(json.dumps(sur.to_dict()), encoding="utf-8")
    return files


def load_candidate(run_dir, cand_id) -> Surrogate:
    return surrogate_from_dict(json.loads((distill_dir(run_dir) / cand_id / "surrogate.json").read_text(encoding="utf-8")))


def load_report(run_dir) -> dict | None:
    f = distill_dir(run_dir) / "report.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8-sig"))
    except ValueError:
        return None


def report_markdown(report: dict) -> str:
    ex = report["expert"]
    lines = [f"# Modèle mathématique extrait - {report['project']}", "",
             f"Run `{report['run']}` ({report['algo']}), généré le {report['created']} par RL Studio {report['version']}.", "",
             f"- Retour moyen de la politique apprise : **{ex['return_mean']:.3f}** (écart-type {ex['return_std']:.3f}), "
             f"politique aléatoire : {report['random_return']:.3f}",
             f"- Score normalise : 0 = aléatoire, 1 = politique apprise ; tolérance : {report['options']['margin'] * 100:.0f} %",
             "", "| Modèle | Complexité | R2 (états de l'expert) | R2 (états visites) | Retour | Score | Menés à terme | |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in report["candidates"]:
        star = "recommande" if c["id"] == report["recommended"] else ""
        lines.append(f"| {c['label']} | {c['complexity']} | {c['r2_expert']:.3f} | {c['r2_visited']:.3f} | {c['return_mean']:.2f} | "
                     f"{c['score']:.3f} | {c['completed_rate'] * 100:.0f} % | {star} |")
    rec = next((c for c in report["candidates"] if c["id"] == report["recommended"]), None)
    if rec:
        lines += ["", f"## Modèle recommande : {rec['label']}", "", "```", *rec["equations"], "```", "",
                  f"Fichiers : `{rec['id']}/policy.py`, `{rec['id']}/policy.m`, `{rec['id']}/equations.txt`."]
    if report["notes"]:
        lines += ["", "## Points d'attention", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def write_report(report: dict, run_dir) -> Path:
    d = distill_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return d / "report.json"


def build_simulink_controller(sur: Surrogate, out_slx, sample_time=None, session=None, log=print) -> Path:
    """Genere un modele Simulink contenant le controleur : une entree par observation, une sortie par action.

    Necessite MATLAB (via le moteur Python). Un modele a etat (integrale) recoit un pas d'échantillonnage."""
    from ..envs.simulink import MatlabSession

    out = Path(out_slx).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ts = sample_time if sample_time else (getattr(sur, "dt", None) if sur.stateful else None)
    spec = {"obs_names": clean_names(sur.obs_names, M_RESERVED), "act_names": clean_names(sur.act_names, M_RESERVED),
            "code": ascii_safe(sur.to_matlab("fcn")), "sample_time": float(ts) if ts else -1.0}
    spec_file = out.with_suffix(".spec.json")
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    own = session is None
    session = session or MatlabSession()
    try:
        log(f"Génération du contrôleur Simulink : {out}")
        session.eng.rlstudio_build_controller(str(out), str(spec_file), nargout=0)
    finally:
        if own:
            session.close()
        spec_file.unlink(missing_ok=True)
    return out
