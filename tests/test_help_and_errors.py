"""Aide intégrée (infobulles, guide) et erreurs visibles dans l'interface ; cohérence des textes et des versions."""
import ast
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rlstudio"


# -- textes : accents et slug (sans interface) ---------------------------------------------------------------
UNACCENTED = re.compile(r"\b(Parametres?|parametres?|Entrainements?|entrainements?|Modeles?|modeles?|Resultats?|resultats?|"
                        r"Recompenses?|recompenses?|Evaluations?|evaluations?|Duree|duree|Etats?|Periph\w+|Mathematiques?)\b")


def _visible_strings(path):
    """Chaînes de texte d'un module (hors docstrings, hors mots isolés : clés, noms de fichiers, identifiants)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {n.body[0].value.lineno for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.lineno not in docs and " " in n.value.strip():
            yield n.lineno, n.value


def test_user_visible_texts_are_accented():
    bad = [f"{p.relative_to(ROOT)}:{line}: {m.group(0)}" for p in sorted(SRC.rglob("*.py")) for line, s in _visible_strings(p)
           for m in UNACCENTED.finditer(s)]
    assert not bad, "texte sans accent :\n" + "\n".join(bad)


def test_slug_ignores_accents():
    from rlstudio.project import slugify

    assert slugify("Régulation d'une zone chauffée") == "regulation-d-une-zone-chauffee"
    assert slugify("   ") == "projet"


def test_generated_code_keeps_ascii_only():
    from rlstudio.textutil import ascii_safe

    assert ascii_safe("% Modèle réduit identifié - état") == "% Modele reduit identifie - etat"


def test_python_support_is_consistent_everywhere():
    from rlstudio import components, pyversion

    lo, hi = pyversion.SUPPORTED
    spec = re.search(r'requires-python = ">=(\d+)\.(\d+),<(\d+)\.(\d+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert spec and (int(spec[1]), int(spec[2])) == lo and (int(spec[3]), int(spec[4])) == (hi[0], hi[1] + 1)
    assert components.BUNDLED_PYTHON == f"{pyversion.RECOMMENDED[0]}.{pyversion.RECOMMENDED[1]}"
    assert lo <= pyversion.RECOMMENDED <= hi
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{lo[0]}.{lo[1]}" in readme and f"{hi[0]}.{hi[1]}" in readme


def test_python_support_warning():
    from rlstudio import pyversion

    assert pyversion.support_warning((3, 12)) is None and pyversion.support_warning((3, 10)) is None
    assert "3.13" in pyversion.support_warning((3, 13)) and "3.9" in pyversion.support_warning((3, 9))


def test_version_is_the_same_everywhere():
    from rlstudio import __version__

    assert f'version = "{__version__}"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f"badge/version-{__version__}-" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert f'AppVersion "{__version__}"' in (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
    assert f'$Version = "{__version__}"' in (ROOT / "packaging" / "build.ps1").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert (ROOT / "docs" / "releases" / f"{__version__}.md").is_file()


def test_readme_lists_every_template_and_command():
    from rlstudio.cli import main  # noqa: F401  (le module doit s'importer)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for d in sorted((ROOT / "templates").iterdir()):
        assert d.name in readme, f"modèle de projet absent du README : {d.name}"
    cli = (SRC / "cli.py").read_text(encoding="utf-8")
    for cmd in re.findall(r'sub\.add_parser\("(\w+)"', cli):
        assert cmd in readme, f"commande absente du README : {cmd}"


def test_guide_is_shipped_and_documents_each_tab():
    guide = (ROOT / "docs" / "guide.md").read_text(encoding="utf-8")
    for section in ("5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"):
        assert re.search(rf"^### {re.escape(section)} ", guide, re.M), section
    assert "guide.md" in (ROOT / "packaging" / "build.ps1").read_text(encoding="utf-8")


# -- interface ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    from rlstudio.project import ProjectStore
    from rlstudio.settings import Settings
    from rlstudio.gui import app as app_mod
    from rlstudio.gui.app import MainWindow

    from rlstudio.gui import settings_dialog

    monkeypatch.setattr(app_mod, "support_warning", lambda: None)  # le Python de test peut être hors plage : pas de bandeau au départ
    # la fenêtre Paramètres lance un processus qui importe PyTorch (plusieurs secondes de processeur) : inutile ici
    monkeypatch.setattr(settings_dialog.SettingsDialog, "_refresh_env_info", lambda self: None)

    store = ProjectStore(root=tmp_path / "projects")
    store.create("Py", "cartpole")
    store.create("Sl", "cartpole_simulink")
    w = MainWindow(store=store, settings=Settings(tmp_path / "settings.json"))
    yield w  # sans show() : l'affichage initial d'une fenêtre hors écran coûte plusieurs secondes et n'est pas nécessaire ici
    w.banner.dismiss()
    w._set_dirty(False)  # sinon la fermeture ouvre la boîte « Enregistrer les modifications ? »
    w.close()
    import shiboken6

    shiboken6.delete(w)  # une fenêtre qui survit ralentirait chaque changement de thème des tests suivants


def _missing_tooltips(root):
    from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit, QListWidget, QPlainTextEdit, QPushButton,
                                   QTableWidget, QTreeWidget, QWidget)

    kinds = (QPushButton, QComboBox, QAbstractSpinBox, QCheckBox, QPlainTextEdit, QListWidget, QTableWidget, QTreeWidget, QLineEdit)
    missing = []
    for c in root.findChildren(QWidget):
        if isinstance(c, QLineEdit) and isinstance(c.parent(), (QAbstractSpinBox, QComboBox)):
            continue  # champ interne d'un autre composant
        name = type(c).__name__
        if name == "PlotWidget" and type(c.parent()).__name__ == "SeriesPlot":
            continue  # l'infobulle est sur le SeriesPlot
        if (isinstance(c, kinds) or name == "PlotWidget") and not c.toolTip().strip():
            text = c.text() if hasattr(c, "text") and not isinstance(c, QPlainTextEdit) else ""
            missing.append(f"{name} {c.objectName()!r} {text!r} (dans {type(c.parent()).__name__})")
    return missing


@pytest.mark.gui
@pytest.mark.parametrize("which", [0, 1])
def test_every_control_of_the_main_window_has_a_tooltip(win, which):
    win._load_project(win.store.get(win.store.list()[which].slug))
    assert not _missing_tooltips(win), "\n".join(_missing_tooltips(win))
    for i in range(win.tabs.count()):
        assert win.tabs.tabToolTip(i).strip(), win.tabs.tabText(i)


@pytest.mark.gui
def test_every_control_of_the_secondary_windows_has_a_tooltip(win):
    from rlstudio.gui import theme
    from rlstudio.gui.convert_dialog import ConvertDialog
    from rlstudio.gui.explorer import SimulinkExplorer
    from rlstudio.gui.settings_dialog import SettingsDialog, ThemeEditor

    dlg = SettingsDialog(win, win.settings, win._apply_theme, win._all_themes, win.theme_name)
    blocks = [{"path": "m/g", "type": "Gain", "n_out": 1, "params": [{"name": "Gain", "value": 1}]}]
    for d in (dlg, ThemeEditor(dlg, theme.BUILTIN["Sombre"], "", set(), {}, lambda t: None),
              ConvertDialog(win, win.settings, win.store), SimulinkExplorer(win, blocks)):
        assert not _missing_tooltips(d), type(d).__name__ + "\n" + "\n".join(_missing_tooltips(d))


@pytest.mark.gui
def test_table_cells_and_headers_explain_themselves(win):
    win._load_project(win.store.get(win.store.list()[1].slug))
    for table in (win.t_actions, win.t_obs, win.t_init):
        assert table.rowCount() >= 0
        assert all(table.horizontalHeaderItem(j).toolTip() for j in range(table.columnCount()))
    win.t_actions.add_row({"name": "u"})
    assert all(win.t_actions.item(0, j).toolTip() for j in range(win.t_actions.columnCount()))


@pytest.mark.gui
def test_stylesheet_parses_for_every_builtin_theme(qapp):
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    from rlstudio.gui import theme

    seen = []
    previous = qInstallMessageHandler(lambda mode, ctx, msg: seen.append(msg))
    try:
        for name, base in theme.BUILTIN.items():
            theme.apply(qapp, base)
            qapp.processEvents()
    finally:
        qInstallMessageHandler(previous)
        theme.apply(qapp, theme.BUILTIN[theme.DEFAULT_THEME])
    assert not [m for m in seen if "stylesheet" in m.lower()], seen
    assert QtMsgType  # import utilisé


@pytest.mark.gui
def test_guide_opens_on_the_section_of_each_tab(win):
    from rlstudio.gui.guide import TAB_SECTIONS, GuideView

    view = GuideView()
    titles = [t for _, _, t in view._headings]
    assert view.toc.count() >= 15
    for tab, prefix in TAB_SECTIONS.items():
        assert any(t.startswith(prefix) for t in titles), (tab, prefix)
        assert view.goto_section(prefix)
    view.search.setText("DAgger")
    view.find_next()
    assert view.browser.textCursor().selectedText() == "DAgger"
    view.search.setText("zzzintrouvablezzz")
    view.find_next()
    assert view.status.text() == "Aucun résultat."
    assert {win.tabs.tabText(i) for i in range(win.tabs.count())} == set(TAB_SECTIONS)


@pytest.mark.gui
def test_guide_is_reachable_from_f1_the_sidebar_and_the_settings(win):
    from rlstudio.gui.settings_dialog import PAGES, SettingsDialog

    win.tabs.setCurrentIndex(win.tabs.indexOf(win.identify))
    win.open_guide()
    assert win._guide.isVisible() and win._guide.view.browser.textCursor().block().text().startswith("5.8")
    win._guide.close()
    win.btn_guide.click()
    assert win._guide.isVisible()
    win._guide.close()
    dlg = SettingsDialog(win, win.settings, win._apply_theme, win._all_themes, win.theme_name)
    assert "Guide d'utilisation" in PAGES and dlg.guide_view.browser.toPlainText().startswith("Guide d'utilisation")


@pytest.mark.gui
def test_guide_is_available_while_a_computation_runs(win):
    win.worker = object()  # un entraînement est en cours : les Paramètres sont verrouillés, pas le guide
    try:
        win.open_guide()
        assert win._guide.isVisible()
    finally:
        win.worker = None
        win._guide.close()


# -- erreurs visibles -----------------------------------------------------------------------------------------
def test_error_helpers():
    pytest.importorskip("PySide6")
    from rlstudio.gui.notify import looks_like_error, summarize

    assert summarize("ERREUR :\nTraceback (most recent call last):\n  File \"x.py\", line 3, in f\n    1/0\nZeroDivisionError: division by zero") \
        == "ZeroDivisionError: division by zero"
    assert summarize("ERREUR : le processus s'est arrêté (code 1).") == "le processus s'est arrêté (code 1)."
    assert summarize("ERREUR :\nModèle introuvable : a.slx") == "Modèle introuvable : a.slx"
    assert len(summarize("x" * 1000)) <= 230
    assert all(looks_like_error(s) for s in ("ERREUR : boom", "ValueError: boom", "Erreur de lancement : x", "MATLAB:undefined",
                                              "Impossible d'ouvrir le modèle"))
    assert not any(looks_like_error(s) for s in ("Episode 3 | 200 pas", "Traceback (most recent call last):", "Aucune erreur détectée",
                                                  "Terminé en 4 s."))


@pytest.mark.gui
def test_banner_shows_counts_dedupes_and_dismisses(win):
    b = win.banner
    assert b.isHidden()
    b.report("Première erreur", "détail 1")
    assert not b.isHidden() and "Première erreur" in b.text.text() and b.property("level") == "error"
    b.report("Première erreur", "un détail bien plus long que le premier")
    assert len(b.history) == 1 and b.history[0]["detail"].startswith("un détail bien plus long")
    b.report("Seconde erreur", level="warning")
    assert len(b.history) == 2 and "+ 1 autre" in b.text.text() and b.property("level") == "warning"
    b.dismiss()
    assert b.isHidden() and not b.history


@pytest.mark.gui
def test_unhandled_exceptions_are_shown(win):
    try:
        raise RuntimeError("panne interne")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    win.banner.incoming.emit("rien", "", "error")  # le signal est le chemin sûr depuis un autre thread
    assert any("RuntimeError: panne interne" in e["summary"] for e in win.banner.history)


@pytest.mark.gui
@pytest.mark.parametrize("tab_name", ["distill", "identify", "evaltab"])
def test_failure_and_logged_errors_reach_the_banner(win, tab_name):
    tab = getattr(win, tab_name)
    tab._t0 = None
    for attr in ("btn_stop", "btn_start", "run_combo", "policy_combo", "mode", "btn_read"):
        if hasattr(tab, attr):
            getattr(tab, attr)
    tab._on_log("Episode 3 | 200 pas")  # ligne ordinaire : rien
    assert win.banner.isHidden()
    tab._on_log("ValueError: observation hors bornes")
    assert not win.banner.isHidden() and "observation hors bornes" in win.banner.history[-1]["summary"]
    win.banner.dismiss()
    tab.runner = None
    tab._on_finished("ERREUR :\nTraceback (most recent call last):\n  File \"w.py\", line 1\nRuntimeError: calcul impossible")
    assert "calcul impossible" in win.banner.history[-1]["summary"] and "Traceback" in win.banner.history[-1]["detail"]


@pytest.mark.gui
def test_training_errors_reach_the_banner(win):
    import time

    win._t_run = time.monotonic()
    win._log("Episode 1 | 100 pas | récompense 3.00")
    assert win.banner.isHidden()
    win._log("RuntimeError: Simulink a refusé la simulation")
    assert not win.banner.isHidden() and "Simulink a refusé" in win.banner.history[-1]["summary"]
    win.banner.dismiss()
    win._on_finished("ERREUR :\nTraceback (most recent call last):\nKeyError: 'obs'")
    assert win.status.text() == "Erreur : KeyError: 'obs'"
    assert "Échec de l'entraînement" in win.banner.history[-1]["summary"]


@pytest.mark.gui
def test_settings_log_dialog_flags_errors(qapp, win):
    from rlstudio.gui.settings_dialog import LogDialog

    dlg = LogDialog(win, "Installation", ["--selftest"])
    dlg._on_line("Collecting numpy")
    dlg._on_line("ERROR: No matching distribution found for torch")
    dlg._finished(0)  # même avec un code 0, une erreur écrite dans la sortie est signalée
    assert "No matching distribution" in dlg.status.text() and dlg.status.property("tone") == "bad"


@pytest.mark.gui
def test_old_python_shows_a_warning_banner(qapp, tmp_path, monkeypatch):
    from rlstudio import pyversion
    from rlstudio.gui import app as app_mod
    from rlstudio.project import ProjectStore
    from rlstudio.settings import Settings

    monkeypatch.setattr(app_mod, "support_warning", lambda: pyversion.support_warning((3, 14)))
    w = app_mod.MainWindow(store=ProjectStore(root=tmp_path / "p"), settings=Settings(tmp_path / "s.json"))
    assert w.banner.history and w.banner.property("level") == "warning" and "3.14" in w.banner.history[-1]["summary"]
