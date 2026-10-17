"""Validation de la configuration d'un projet Simulink (sans dependance lourde : utilisee par la GUI)."""
from . import expressions


def validate_spec(spec: dict) -> list[str]:
    """Liste de problemes detectes dans la spec (vide si OK) ; utilise par la GUI."""
    errs = []
    if not spec.get("model"):
        errs.append("Aucun modele Simulink indique.")
    if not spec.get("actions"):
        errs.append("Il faut au moins une action.")
    if not spec.get("observations"):
        errs.append("Il faut au moins une observation.")
    if float(spec.get("step_s", 0)) <= 0 or float(spec.get("episode_s", 0)) <= 0:
        errs.append("step_s et episode_s doivent etre > 0.")
    for a in spec.get("actions", []):
        if not a.get("name"):
            errs.append("Action sans nom.")
        if not a.get("variable") and not (a.get("block") and a.get("param")):
            errs.append(f"Action {a.get('name')!r} : indiquer `variable` ou `block`+`param`.")
        if not float(a.get("low", 0)) < float(a.get("high", 1)):
            errs.append(f"Action {a.get('name')!r} : low doit etre < high.")
    for o in spec.get("observations", []):
        if not o.get("name") or not o.get("block"):
            errs.append(f"Observation {o.get('name')!r} : nom et `block` requis.")
    names = [x["name"] for x in spec.get("actions", []) + spec.get("observations", []) if x.get("name")]
    if len(set(names)) != len(names):
        errs.append("Les noms d'actions/observations doivent etre uniques.")
    if not spec.get("reward"):
        errs.append("Expression de recompense manquante.")
    for key in ("reward", "terminate"):
        if spec.get(key):
            err = expressions.check_expression(spec[key], names)
            if err:
                errs.append(f"{key} : {err}")
    return errs
