"""RL Studio. Imports paresseux : `rlstudio.hardware` doit rester utilisable sans yaml (installateur)."""
__version__ = "0.3.5"

__all__ = ["Project", "ProjectStore"]


def __getattr__(name):
    if name in __all__:
        from . import project

        return getattr(project, name)
    raise AttributeError(name)
