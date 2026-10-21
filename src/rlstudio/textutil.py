"""Utilitaires de texte."""
import unicodedata


def ascii_safe(text: str) -> str:
    """Retire les accents : le code genere (.m, bloc Simulink) reste lisible quel que soit l'encodage lu par MATLAB."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
