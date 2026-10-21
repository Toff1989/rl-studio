"""Themes de l'application : definition par couleurs de base, derivation des autres, QSS, graphiques.

Un theme = {bg, surface, text, accent, success, warning, danger, magenta, font}. Les nuances
(bordures, texte attenue, survol...) sont derivees automatiquement : un theme personnalise ne
demande donc que quelques couleurs.
"""
import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont, QPalette

BASE_KEYS = ["bg", "surface", "text", "accent", "success", "warning", "danger", "magenta"]
BASE_LABELS = {"bg": "Fond", "surface": "Cartes et champs", "text": "Texte", "accent": "Accent",
               "success": "Succès", "warning": "Avertissement", "danger": "Danger", "magenta": "Courbe secondaire"}

BUILTIN = {
    "Sombre": dict(bg="#13151a", surface="#1b1e26", text="#e7e9ef", accent="#4f7cf0", success="#3ecf8e",
                   warning="#f5a524", danger="#ef5b5b", magenta="#c678dd", font="Segoe UI"),
    "Clair": dict(bg="#f3f4f8", surface="#ffffff", text="#1b1f2a", accent="#3b6df0", success="#1f9d68",
                  warning="#d98200", danger="#d64545", magenta="#a24cc4", font="Segoe UI"),
    "Matrix": dict(bg="#000800", surface="#031203", text="#33ff66", accent="#00e640", success="#00ff41",
                   warning="#c8ff00", danger="#ff3b3b", magenta="#00ffa8", font="Consolas"),
}
DEFAULT_THEME = "Sombre"

C: dict = {}  # theme actif, derive ; modifie en place pour que `from .theme import C` reste valide


def _rgb(hex_color):
    c = QColor(hex_color)
    return c.red(), c.green(), c.blue()


def mix(a, b, t):
    """Melange a (1-t) + b (t)."""
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return "#{:02x}{:02x}{:02x}".format(round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t))


def luminance(hex_color):
    r, g, b = _rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def derive(base: dict) -> dict:
    t = {k: base.get(k, BUILTIN[DEFAULT_THEME][k]) for k in BASE_KEYS}
    t["font"] = base.get("font") or BUILTIN[DEFAULT_THEME]["font"]
    bg, surface, text, accent = t["bg"], t["surface"], t["text"], t["accent"]
    t.update(
        surface2=mix(surface, text, 0.07),
        alt_row=mix(surface, text, 0.03),
        border=mix(surface, text, 0.14),
        border2=mix(surface, text, 0.28),
        muted=mix(bg, text, 0.58),
        accent2=mix(accent, text, 0.30),
        on_accent="#000000" if luminance(accent) > 0.62 else "#ffffff",
        danger_text=mix(t["danger"], text, 0.35),
        accent_rgb="{}, {}, {}".format(*_rgb(accent)),
        success_rgb="{}, {}, {}".format(*_rgb(t["success"])),
        danger_rgb="{}, {}, {}".format(*_rgb(t["danger"])),
        text_rgb="{}, {}, {}".format(*_rgb(text)),
    )
    return t


QSS = """
QWidget { background: @bg; color: @text; font-family: "@font"; font-size: 13px; }
QLabel { background: transparent; }
QToolTip { background: @surface2; color: @text; border: 1px solid @border; padding: 4px 6px; }
QDialog, QMessageBox { background: @bg; }

#sidebar { background: @surface; border-right: 1px solid @border; }
#sidebar QLabel#appTitle { font-size: 18px; font-weight: 700; }
#topbar { background: @bg; }
QLabel#h1 { font-size: 22px; font-weight: 700; }
QLabel#muted { color: @muted; }
QLabel#sectionLabel { color: @muted; font-size: 11px; font-weight: 600; }
QLabel#badge { background: rgba(@accent_rgb, 0.16); color: @accent2; border-radius: 9px;
               padding: 2px 10px; font-size: 11px; font-weight: 600; }
QLabel#dirty { color: @warning; font-size: 12px; }
QLabel#kpiTitle { color: @muted; font-size: 11px; font-weight: 600; }
QLabel#kpiValue { font-size: 18px; font-weight: 700; }
QLabel#cardTitle { font-size: 14px; font-weight: 700; }
QLabel#hint { color: @muted; font-size: 12px; }
QLabel#statusbox { background: @surface2; border-radius: 6px; padding: 8px 10px; color: @muted; }
QLabel#statusbox[tone="ok"] { background: rgba(@success_rgb, 0.14); color: @success; }
QLabel#statusbox[tone="bad"] { background: rgba(@danger_rgb, 0.14); color: @danger_text; }

QFrame#card { background: @surface; border: 1px solid @border; border-radius: 10px; }
QFrame#card QLabel { background: transparent; }

QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: @surface2; border: 1px solid @border; border-radius: 6px;
    padding: 6px 9px; sélection-background-color: @accent; sélection-color: @on_accent; }
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: @accent; }
QPlainTextEdit#mono { font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0; border: none; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background: @surface2; border: 1px solid @border;
    sélection-background-color: @accent; sélection-color: @on_accent; outline: 0; }

QPushButton { background: @surface2; border: 1px solid @border; border-radius: 6px; padding: 7px 14px; }
QPushButton:hover { border-color: @accent2; }
QPushButton:pressed { background: @border; }
QPushButton:disabled { color: @border2; background: @surface; border-color: @border; }
QPushButton[variant="primary"] { background: @accent; border-color: @accent; color: @on_accent; font-weight: 600; }
QPushButton[variant="primary"]:hover { background: @accent2; border-color: @accent2; }
QPushButton[variant="primary"]:disabled { background: @surface2; border-color: @border; color: @border2; }
QPushButton[variant="danger"] { background: rgba(@danger_rgb, 0.14); border-color: @danger; color: @danger_text; }
QPushButton[variant="danger"]:hover { background: rgba(@danger_rgb, 0.26); }
QPushButton[variant="danger"]:disabled { background: @surface; border-color: @border; color: @border2; }
QPushButton[variant="ghost"] { background: transparent; border-color: transparent; color: @muted; }
QPushButton[variant="ghost"]:hover { background: @surface2; color: @text; }
QPushButton[variant="small"] { padding: 4px 10px; font-size: 12px; }
QPushButton::menu-indicator { width: 0; }

QTabWidget::pane { border: none; border-top: 1px solid @border; }
QTabWidget::tab-bar { left: 20px; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: @muted; padding: 10px 18px; margin-right: 2px;
               border: none; border-bottom: 2px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: @text; border-bottom: 2px solid @accent; }
QTabBar::tab:hover:!selected { color: @text; }

QListWidget { background: transparent; border: none; outline: 0; }
QListWidget::item { border-radius: 8px; margin: 2px 8px; padding: 2px; }
QListWidget::item:selected { background: rgba(@accent_rgb, 0.20); color: @text; }
QListWidget::item:hover:!selected { background: rgba(@text_rgb, 0.06); }
QListWidget#runs::item { padding: 8px 10px; margin: 2px 0; }

QTableWidget { background: @surface; alternate-background-color: @alt_row; border: 1px solid @border;
               border-radius: 6px; gridline-color: @border; sélection-background-color: rgba(@accent_rgb, 0.30);
               sélection-color: @text; }
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section { background: @surface2; color: @muted; border: none;
                       border-bottom: 1px solid @border; padding: 6px 8px; font-weight: 600; }
QTableCornerButton::section { background: @surface2; border: none; }
QTreeWidget { background: @surface; border: 1px solid @border; border-radius: 6px;
              sélection-background-color: rgba(@accent_rgb, 0.30); sélection-color: @text; outline: 0; }

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: @border2; border-radius: 5px; min-height: 32px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: @border2; border-radius: 5px; min-width: 32px; }
QPlainTextEdit QScrollBar:vertical { background: @surface2; }
QPlainTextEdit QScrollBar:horizontal { background: @surface2; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QProgressBar { background: @surface2; border: none; border-radius: 4px; max-height: 8px; min-height: 8px;
               text-align: center; color: transparent; }
QProgressBar::chunk { background: @accent; border-radius: 4px; }

QMenu { background: @surface2; border: 1px solid @border; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 7px 26px 7px 14px; border-radius: 4px; }
QMenu::item:selected { background: rgba(@accent_rgb, 0.28); }
QMenu::separator { height: 1px; background: @border; margin: 4px 8px; }

QCheckBox { background: transparent; spacing: 9px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid @border2; border-radius: 4px; background: @surface2; }
QCheckBox::indicator:checked { background: @accent; border-color: @accent; }
QSplitter::handle { background: transparent; }
#plot { border: 1px solid @border; border-radius: 10px; }
QPushButton#swatch { border: 1px solid @border2; border-radius: 6px; min-width: 64px; min-height: 22px; padding: 2px; }
"""


def stylesheet(t: dict) -> str:
    out = QSS
    for key in sorted(t, key=len, reverse=True):  # @accent2 / @accent_rgb avant @accent
        out = out.replace(f"@{key}", str(t[key]))
    return out


def apply(app, base: dict) -> dict:
    """Applique un theme (couleurs de base) a toute l'application et renvoie le theme derive."""
    t = derive(base)
    C.clear()
    C.update(t)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, key in [(QPalette.Window, "bg"), (QPalette.Base, "surface"), (QPalette.AlternateBase, "alt_row"),
                      (QPalette.Button, "surface2"), (QPalette.WindowText, "text"), (QPalette.Text, "text"),
                      (QPalette.ButtonText, "text"), (QPalette.ToolTipBase, "surface2"),
                      (QPalette.ToolTipText, "text"), (QPalette.Highlight, "accent"),
                      (QPalette.HighlightedText, "on_accent"), (QPalette.PlaceholderText, "muted")]:
        pal.setColor(role, QColor(t[key]))
    app.setPalette(pal)
    app.setFont(QFont(t["font"], 10))
    app.setStyleSheet(stylesheet(t))
    pg.setConfigOptions(background=t["surface"], foreground=t["muted"], antialias=True)
    return t
