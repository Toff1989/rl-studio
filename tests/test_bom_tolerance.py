"""Fichiers edites sous Windows : un BOM UTF-8 (Bloc-notes, PowerShell 5.1) ne doit pas faire perdre les reglages."""
import codecs
import json

from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import Project, ProjectStore
from rlstudio.settings import Settings

BOM = codecs.BOM_UTF8


def test_settings_with_bom_are_read(tmp_path):
    f = tmp_path / "settings.json"
    f.write_bytes(BOM + json.dumps({"theme": "Matrix", "projects_dir": "D:/mes projets"}).encode("utf-8"))
    s = Settings(f)
    assert s.get("theme") == "Matrix" and str(s.projects_dir).replace("\\", "/") == "D:/mes projets"
    f.write_bytes(BOM + json.dumps({"theme": "Clair"}).encode("utf-8"))
    s.reload()
    assert s.get("theme") == "Clair"


def test_project_yaml_with_bom_is_read(tmp_path):
    store = ProjectStore(root=tmp_path / "p", templates=TEMPLATES_DIR)
    p = store.create("Pendule accentue : \u00e9t\u00e9", "cartpole")
    yaml_file = p.path / "project.yaml"
    yaml_file.write_bytes(BOM + yaml_file.read_bytes())
    assert Project(p.path).name == "Pendule accentue : \u00e9t\u00e9"


def test_corrupt_settings_still_fall_back_to_defaults(tmp_path):
    f = tmp_path / "settings.json"
    f.write_bytes(b"{pas du json")
    assert Settings(f).get("theme") == "Sombre"
