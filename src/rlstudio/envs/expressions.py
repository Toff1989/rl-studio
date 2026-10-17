"""Evaluation restreinte d'expressions (recompense, condition d'arret) ecrites dans le YAML."""
import numpy as np

SAFE_FUNCS = {
    "abs": abs, "min": min, "max": max, "sum": np.sum, "mean": np.mean,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
    "tanh": np.tanh, "clip": np.clip, "square": np.square, "where": np.where,
    "pi": np.pi, "norm": np.linalg.norm,
}
BASE_NAMES = {"t", "dt", "step"}


def check_expression(expr: str, variables) -> str | None:
    """Renvoie un message d'erreur, ou None si l'expression est valide."""
    try:
        code = compile(expr, "<expression>", "eval")
    except SyntaxError as e:
        return f"Syntaxe invalide : {e.msg}"
    allowed = set(SAFE_FUNCS) | BASE_NAMES | set(variables)
    unknown = [n for n in code.co_names if n not in allowed]
    if unknown:
        return f"Nom(s) inconnu(s) : {', '.join(unknown)}. Disponibles : {', '.join(sorted(allowed))}"
    return None


def evaluate(expr: str, namespace: dict):
    return eval(compile(expr, "<expression>", "eval"), {"__builtins__": {}}, {**SAFE_FUNCS, **namespace})
