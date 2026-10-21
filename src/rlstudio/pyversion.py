"""Versions de Python prises en charge (source unique : doit rester alignée sur `requires-python` de pyproject.toml)."""
import sys

SUPPORTED = ((3, 10), (3, 12))  # bornes incluses
RECOMMENDED = (3, 12)  # version du Python embarqué dans l'installateur (voir components.BUNDLED_PYTHON)


def _fmt(v):
    return f"{v[0]}.{v[1]}"


def support_warning(version=None):
    """Message à afficher si le Python courant sort de la plage validée, sinon None (le programme continue : c'est un avertissement)."""
    v = tuple((version or sys.version_info)[:2])
    if SUPPORTED[0] <= v <= SUPPORTED[1]:
        return None
    return (f"Python {_fmt(v)} n'est pas validé pour RL Studio (versions prises en charge : {_fmt(SUPPORTED[0])} à "
            f"{_fmt(SUPPORTED[1])}, {_fmt(RECOMMENDED)} recommandée). Des composants (PySide6, PyTorch, MATLAB Engine) "
            "peuvent ne pas s'installer ou se comporter autrement.")
