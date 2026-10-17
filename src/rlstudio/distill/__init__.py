"""Extraction d'un modele mathematique de la politique apprise (distillation)."""
from .pipeline import METHOD_LABELS, METHODS, DistillOptions, run_distillation

__all__ = ["DistillOptions", "METHODS", "METHOD_LABELS", "run_distillation"]
