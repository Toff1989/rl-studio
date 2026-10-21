"""Ouverture d'un .rls depuis la ligne de commande (double-clic dans l'Explorateur)."""
import pytest

from rlstudio import launcher


@pytest.mark.gui
def test_rls_argument_is_forwarded_to_the_gui(monkeypatch):
    opened = []
    monkeypatch.setattr("rlstudio.gui.app.main", lambda open_file=None: opened.append(open_file))
    assert launcher.main([r"C:\projets\pendule.RLS"]) == 0
    assert opened == [r"C:\projets\pendule.RLS"]


@pytest.mark.gui
def test_other_arguments_are_not_treated_as_projects(monkeypatch):
    opened = []
    monkeypatch.setattr("rlstudio.gui.app.main", lambda open_file=None: opened.append(open_file))
    launcher.main([r"C:\autre\fichier.txt"])
    launcher.main([])
    assert opened == [None, None]
