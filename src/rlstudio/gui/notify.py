"""Erreurs visibles dans l'interface : bandeau sur tous les onglets, fenêtre de détail, exceptions non gérées.

Une erreur écrite dans un journal n'est pas forcément vue (journal replié, autre onglet) : tout ce qui ressemble à une
erreur passe aussi par `ErrorBanner.report`, qui l'affiche en haut de la fenêtre jusqu'à ce que l'utilisateur la ferme.
"""
import re
import sys
import threading
import traceback

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout

from .widgets import button

SUMMARY_MAX = 230
HISTORY_MAX = 50

# début d'une ligne de journal qui signale une erreur (ERREUR, NomError: ..., NomException: ...) ; l'en-tête « Traceback » est exclu :
# c'est la dernière ligne du traceback, qui porte le message utile
_ERROR_LINE = re.compile(
    r"^\s*(?:ERREUR\b|ERROR\b|Erreur\b|Error\b|Exception\b|[\w.]*(?:Error|Exception)\b\s*[:(]|MATLAB:\w+|Impossible\b)")


def looks_like_error(line: str) -> bool:
    """La ligne de journal signale-t-elle une erreur ?"""
    return bool(_ERROR_LINE.match(str(line)))


def summarize(text: str) -> str:
    """Une ligne lisible pour un message d'erreur long : la dernière ligne utile d'un traceback, sinon la première."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return "Erreur inconnue."
    if any(ln.startswith("Traceback") for ln in lines):
        pick = next((ln for ln in reversed(lines) if not ln.startswith(("File ", "^", "~"))), lines[-1])
    else:
        pick = lines[0]
        if re.fullmatch(r"ERREUR\s*:?", pick) and len(lines) > 1:  # « ERREUR : » seul sur sa ligne : le détail suit
            pick = lines[1]
    pick = re.sub(r"^ERREUR\s*:\s*", "", pick)
    return pick if len(pick) <= SUMMARY_MAX else pick[:SUMMARY_MAX - 1] + "…"


class DetailDialog(QDialog):
    """Texte complet des erreurs reçues, avec copie dans le presse-papiers."""

    def __init__(self, parent, entries):
        super().__init__(parent)
        self.setWindowTitle("Détail des erreurs")
        self.resize(820, 460)
        lay = QVBoxLayout(self)
        self.box = QPlainTextEdit()
        self.box.setObjectName("mono")
        self.box.setReadOnly(True)
        self.box.setToolTip("Texte complet des erreurs, de la plus récente à la plus ancienne.")
        self.box.setPlainText("\n\n".join(f"[{e['time']}] {e['summary']}\n{e['detail']}".rstrip() for e in reversed(entries)))
        lay.addWidget(self.box, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Copier", None, self._copy, tip="Copie tout le texte dans le presse-papiers pour le transmettre au support."))
        row.addWidget(button("Fermer", "primary", self.accept, tip="Ferme cette fenêtre."))
        lay.addLayout(row)

    def _copy(self):
        QGuiApplication.clipboard().setText(self.box.toPlainText())


class ErrorBanner(QFrame):
    """Bandeau en haut de la fenêtre : dernière erreur, nombre d'erreurs, accès au détail."""

    incoming = Signal(str, str, str)  # (résumé, détail, niveau) : sûr depuis un autre thread (livraison dans le thread de l'interface)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("errorBanner")
        self.incoming.connect(self.report)
        self.setProperty("level", "error")
        self.history: list[dict] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 8, 20, 8)
        lay.setSpacing(12)
        self.icon = QLabel("⚠")
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text.setToolTip("Dernière erreur signalée. « Détails » affiche le texte complet.")
        self.btn_detail = button("Détails…", "small", self.show_details, tip="Affiche le texte complet des erreurs (utile pour comprendre ou signaler un problème).")
        self.btn_close = button("✕", "ghost", self.dismiss, tip="Masque ce bandeau (les erreurs restent consultables dans le journal).")
        self.btn_close.setFixedWidth(34)
        lay.addWidget(self.icon)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.btn_detail)
        lay.addWidget(self.btn_close)
        self.hide()

    def report(self, summary: str, detail: str = "", level: str = "error"):
        """Affiche une erreur (`level` : "error" ou "warning"). Une répétition identique n'ajoute rien."""
        import time

        summary = summarize(summary)
        if self.history and self.history[-1]["summary"] == summary and not self.isHidden():
            if len(detail) > len(self.history[-1]["detail"]):  # même erreur, avec cette fois le texte complet
                self.history[-1]["detail"] = detail
            return
        self.history = (self.history + [{"time": time.strftime("%H:%M:%S"), "summary": summary,
                                         "detail": detail or summary, "level": level}])[-HISTORY_MAX:]
        n = len(self.history)
        self.setProperty("level", level)
        self.style().unpolish(self)
        self.style().polish(self)
        self.text.setText(summary + (f"   (+ {n - 1} autre{'s' if n > 2 else ''})" if n > 1 else ""))
        self.show()

    def dismiss(self):
        self.history.clear()
        self.hide()

    def show_details(self):
        if self.history:
            DetailDialog(self.window(), self.history).exec()


def install_excepthook(banner: "ErrorBanner"):
    """Les exceptions Python non gérées (slot Qt, thread) deviennent une erreur affichée, au lieu de disparaître en console."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        previous(exc_type, exc, tb)
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            banner.incoming.emit(f"Erreur interne : {exc_type.__name__}: {exc}", text, "error")
        except Exception:
            pass

    sys.excepthook = hook

    def thread_hook(args):
        hook(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = thread_hook
