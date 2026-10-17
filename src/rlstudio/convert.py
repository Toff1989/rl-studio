"""Conversion d'un modele Simulink existant en projet RL Studio : specification, validation, project.yaml.

Logique pure (sans MATLAB ni interface) ; l'analyse et la conversion du modele sont faites par matlab/rlstudio_analyze.m
et matlab/rlstudio_convert.m, pilotes par le worker (commandes `analyze-model` et `convert-model`).

Specification (dict, produite par la fenetre de conversion) :

  project_name, model_path, init_path   projet cree, modele source, script/.mat d'initialisation (optionnel)
  actions       [{name, kind: "inport"|"param", path, param, default, low, high}]
                  inport : une entree racine remplacee par une constante pilotable ; param : un parametre de bloc existant
  observations  [{name, path, port}]    signaux (une sortie racine est remplacee par le bloc qui l'alimente)
  initial       [{variable, low, high}] variables du workspace tirees au hasard a chaque episode
  variables     [nom]                   variables du workspace de base a embarquer dans le modele converti
  remove        [chemin]                blocs a retirer (scopes, affichages...)
  step_s, episode_s, solver ("keep"|"fixed"), solver_step, reward, terminate
"""
import re
import time
from pathlib import Path

import yaml

from .envs import expressions
from .envs.spec import validate_spec

_RESERVED = set(expressions.SAFE_FUNCS) | expressions.BASE_NAMES | {"and", "or", "not", "if", "else", "in", "is", "True", "False", "None"}


def sanitize_name(name: str, fallback="signal") -> str:
    """Nom utilisable dans les expressions de recompense : lettres, chiffres, underscore, pas de chiffre en tete."""
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"{fallback}_{s}"
    if s in _RESERVED:
        s = f"{s}_"
    return s


def unique_names(names):
    """Rend une liste de noms unique (suffixe _2, _3...) en conservant l'ordre."""
    seen, out = {}, []
    for n in names:
        base = n
        k = seen.get(base, 0) + 1
        seen[base] = k
        out.append(base if k == 1 else f"{base}_{k}")
    return out


def validate_conversion(spec: dict) -> list[str]:
    """Liste de problemes de la specification (vide si OK) ; utilisee par la fenetre avant de lancer la conversion."""
    errs = []
    if not str(spec.get("project_name", "")).strip():
        errs.append("Donne un nom au projet.")
    if not spec.get("model_path"):
        errs.append("Aucun modele Simulink choisi.")
    acts, obs = spec.get("actions") or [], spec.get("observations") or []
    if not acts:
        errs.append("Il faut au moins une action (une entree ou un parametre pilote par l'agent).")
    if not obs:
        errs.append("Il faut au moins une observation.")
    for a in acts:
        if not a.get("name"):
            errs.append("Une action n'a pas de nom.")
        if a.get("kind") == "param" and not (a.get("path") and a.get("param")):
            errs.append(f"Action {a.get('name')!r} : bloc et parametre requis.")
        try:
            if not float(a.get("low", 0)) < float(a.get("high", 1)):
                errs.append(f"Action {a.get('name')!r} : le minimum doit etre inferieur au maximum.")
        except (TypeError, ValueError):
            errs.append(f"Action {a.get('name')!r} : minimum / maximum invalides.")
    names = [x.get("name") for x in acts + obs]
    if len(set(names)) != len(names):
        errs.append("Les noms des actions et observations doivent etre uniques.")
    for x in acts + obs:
        n = x.get("name") or ""
        if n and sanitize_name(n) != n:
            errs.append(f"Nom {n!r} invalide (lettres, chiffres et _ uniquement, sans mot reserve).")
    for i in spec.get("initial") or []:
        try:
            if float(i["low"]) > float(i["high"]):
                errs.append(f"Variable initiale {i.get('variable')!r} : minimum superieur au maximum.")
        except (KeyError, TypeError, ValueError):
            errs.append(f"Variable initiale {i.get('variable')!r} : bornes invalides.")
    try:
        if float(spec.get("step_s", 0)) <= 0 or float(spec.get("episode_s", 0)) <= 0:
            errs.append("Le pas de decision et la duree de l'episode doivent etre > 0.")
        elif float(spec["episode_s"]) < float(spec["step_s"]):
            errs.append("L'episode doit durer au moins un pas de decision.")
    except (TypeError, ValueError):
        errs.append("Pas de decision / duree d'episode invalides.")
    if spec.get("solver") == "fixed":
        try:
            if float(spec.get("solver_step", 0)) <= 0:
                errs.append("Le pas du solveur doit etre > 0.")
            elif float(spec["step_s"]) < float(spec["solver_step"]):
                errs.append("Le pas du solveur doit etre inferieur ou egal au pas de decision.")
        except (TypeError, ValueError):
            errs.append("Pas du solveur invalide.")
    if not spec.get("reward"):
        errs.append("Ecris une expression de recompense.")
    else:
        for key in ("reward", "terminate"):
            if spec.get(key):
                err = expressions.check_expression(spec[key], [n for n in names if n])
                if err:
                    errs.append(f"{'Recompense' if key == 'reward' else 'Arret anticipe'} : {err}")
    return errs


def matlab_spec(spec: dict) -> dict:
    """Partie de la specification lue par matlab/rlstudio_convert.m."""
    return {
        "inports": [{"path": a["path"], "name": a["name"], "default": a.get("default", 0.0)}
                    for a in spec.get("actions") or [] if a.get("kind") == "inport"],
        "remove": list(spec.get("remove") or []),
        "variables": list(spec.get("variables") or []),
        "observations": [{"name": o["name"], "path": o["path"], "port": int(o.get("port", 1))}
                         for o in spec.get("observations") or []],
        "solver": spec.get("solver", "keep"),
        "solver_step": float(spec.get("solver_step", 0.01)),
        "silence_loops": True,
    }


def build_env_cfg(spec: dict, result: dict, model_file: str) -> dict:
    """Cle `env` du project.yaml, a partir de la specification et de la reponse de rlstudio_convert."""
    converted = {a["name"]: a for a in result.get("actions") or []}
    actions = []
    for a in spec["actions"]:
        if a.get("kind") == "inport":
            block, param = converted[a["name"]]["block"], "Value"
        else:
            block, param = a["path"], a["param"]
        actions.append({"name": a["name"], "block": block, "param": param,
                        "low": float(a["low"]), "high": float(a["high"])})
    resolved = {o["name"]: o for o in result.get("observations") or []}
    observations = [{"name": o["name"], "block": resolved[o["name"]]["block"], "port": int(resolved[o["name"]]["port"]),
                     "size": 1} for o in spec["observations"]]
    env = {
        "model": model_file,
        "step_s": float(spec["step_s"]),
        "episode_s": float(spec["episode_s"]),
        "actions": actions,
        "observations": observations,
        "initial": [{"variable": i["variable"], "low": float(i["low"]), "high": float(i["high"])}
                    for i in spec.get("initial") or []],
        "reward": spec["reward"],
        "terminate": spec.get("terminate") or "",
    }
    errs = validate_spec(env)
    if errs:
        raise ValueError("Configuration Simulink invalide :\n- " + "\n- ".join(errs))
    return env


def write_project(project_dir, spec: dict, result: dict) -> dict:
    """Ecrit project.yaml dans `project_dir` (le modele converti y est deja) ; renvoie la config."""
    from .project import DEFAULT_ALGO

    model_file = f"{result['model']}.slx"
    cfg = {
        "name": spec["project_name"].strip(),
        "description": f"Modele Simulink converti ({Path(spec['model_path']).name}) : copie adaptee a RL Studio.",
        "kind": "simulink",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "env": build_env_cfg(spec, result, model_file),
        "algo": {**DEFAULT_ALGO, "name": "SAC", "total_timesteps": 20000, "baseline_episodes": 1},
    }
    (Path(project_dir) / "project.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg


def default_spec(analysis: dict) -> dict:
    """Proposition initiale a partir de l'analyse : toutes les entrees scalaires en actions, toutes les sorties en observations."""
    names = []
    actions = []
    for i in analysis.get("inports") or []:
        if i.get("convertible"):
            n = sanitize_name(i["name"], "u")
            actions.append({"name": n, "kind": "inport", "path": i["path"], "param": "Value", "default": 0.0,
                            "low": -1.0, "high": 1.0})
            names.append(n)
    observations = []
    for o in analysis.get("outports") or []:
        if o.get("observable"):
            observations.append({"name": sanitize_name(o["name"], "y"), "path": o["path"], "port": 1})
    for group in (actions, observations):
        for item, n in zip(group, unique_names([x["name"] for x in group])):
            item["name"] = n
    return {"actions": actions, "observations": observations}
