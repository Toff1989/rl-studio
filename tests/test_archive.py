"""Format .rls : aller-retour, filtrage a l'export, et refus des archives dangereuses a l'import."""
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from rlstudio import archive
from rlstudio.archive import ArchiveError, export_project, import_project, inspect_archive
from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(root=tmp_path / "a", templates=TEMPLATES_DIR)


@pytest.fixture
def other(tmp_path):
    return ProjectStore(root=tmp_path / "b", templates=TEMPLATES_DIR)


@pytest.fixture
def project(store):
    p = store.create("Pendule", "cartpole")
    (p.path / "donnees.csv").write_text("a,b\n1,2\n")
    (p.path / "__pycache__").mkdir()
    (p.path / "__pycache__" / "env.cpython-312.pyc").write_bytes(b"\0")
    (p.path / "modele.slxc").write_bytes(b"cache")
    (p.path / "slprj").mkdir()
    (p.path / "slprj" / "x.mat").write_bytes(b"\0")
    for run in ("20260101-100000", "20260102-100000"):
        d = p.runs_dir / run
        d.mkdir(parents=True)
        (d / "model.zip").write_bytes(b"modele-" + run.encode())
        (d / "baseline.json").write_text(json.dumps({"mean": 1.0}))
    return p


def names(rls):
    with zipfile.ZipFile(rls) as zf:
        return set(zf.namelist())


def make_zip(path, entries, manifest=True, project_yaml=True):
    """Archive fabriquee a la main (y compris dangereuse) pour tester l'import."""
    with zipfile.ZipFile(path, "w") as zf:
        if manifest:
            zf.writestr("rlstudio.json", json.dumps({"format": "rlstudio-project", "format_version": 1,
                                                    "name": "Piege", "kind": "python", "app_version": "0"}))
        if project_yaml:
            zf.writestr("project/project.yaml", "name: Piege\nkind: python\nenv: {entry: 'env.py:make_env'}\n")
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


# -- aller-retour ------------------------------------------------------------------------
def test_roundtrip_without_runs(project, other, tmp_path):
    out = export_project(project, tmp_path / "pendule")
    assert out.suffix == ".rls" and out.exists()
    imported = import_project(out, other)
    assert imported.name == "Pendule" and imported.kind == "python"
    assert (imported.path / "env.py").read_text() == (project.path / "env.py").read_text()
    assert (imported.path / "donnees.csv").exists()
    assert not imported.runs_dir.exists()


def test_export_excludes_caches_and_generated_files(project, tmp_path):
    n = names(export_project(project, tmp_path / "p.rls"))
    assert "rlstudio.json" in n and "project/project.yaml" in n and "project/env.py" in n
    assert not any("__pycache__" in x or x.endswith(".slxc") or "slprj" in x or "/runs/" in x for x in n)


@pytest.mark.parametrize("mode,expected", [("none", 0), ("last", 1), ("all", 2)])
def test_runs_selection(project, other, tmp_path, mode, expected):
    out = export_project(project, tmp_path / f"{mode}.rls", runs=mode)
    imported = import_project(out, other)
    assert len(imported.list_runs()) == expected
    if mode == "last":
        assert imported.list_runs()[0].name == "20260102-100000"  # le plus recent
        assert (imported.list_runs()[0] / "model.zip").read_bytes() == b"modele-20260102-100000"


def test_manifest_content(project, tmp_path):
    report = inspect_archive(export_project(project, tmp_path / "p.rls", runs="last"))
    m = report["manifest"]
    assert m["format"] == "rlstudio-project" and m["kind"] == "python" and m["runs"] == ["20260102-100000"]
    assert "env.py" in report["code_files"] and report["problems"] == []


def test_import_never_overwrites_existing_projects(project, store, tmp_path):
    out = export_project(project, tmp_path / "p.rls")
    first = import_project(out, store)
    second = import_project(out, store)
    assert len({project.slug, first.slug, second.slug}) == 3
    assert {project.name, first.name, second.name} == {"Pendule", "Pendule (2)", "Pendule (3)"}
    assert not [d for d in store.root.iterdir() if d.name.startswith(".import-")]  # pas de reste de staging


# -- archives dangereuses ------------------------------------------------------------------
@pytest.mark.parametrize("evil", ["project/../../evil.txt", "../evil.txt", "/abs/evil.txt", "C:/evil.txt",
                                   "project\\..\\..\\evil.txt", "project/a/../../../evil.txt"])
def test_path_traversal_is_rejected(tmp_path, other, evil):
    rls = make_zip(tmp_path / "x.rls", {evil: "pwn"})
    with pytest.raises(ArchiveError):
        import_project(rls, other)
    assert not (tmp_path / "evil.txt").exists() and not (other.root.parent / "evil.txt").exists()
    assert list(other.root.iterdir()) == []


def test_files_outside_project_folder_are_rejected(tmp_path, other):
    with pytest.raises(ArchiveError, match="hors du dossier"):
        import_project(make_zip(tmp_path / "x.rls", {"autre/fichier.txt": "x"}), other)


def test_symlink_is_rejected(tmp_path, other):
    rls = tmp_path / "x.rls"
    make_zip(rls, {})
    with zipfile.ZipFile(rls, "a") as zf:
        info = zipfile.ZipInfo("project/lien")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "C:/Windows/system.ini")
    with pytest.raises(ArchiveError, match="symbolique"):
        import_project(rls, other)


@pytest.mark.parametrize("name", ["tool.exe", "lib.dll", "run.bat", "x.ps1", "ext.pyd", "a.sh", "setup.msi"])
def test_native_executables_and_scripts_are_blocked(tmp_path, other, name):
    with pytest.raises(ArchiveError, match="executables refuses"):
        import_project(make_zip(tmp_path / "x.rls", {f"project/{name}": "MZ"}), other)


def test_code_files_are_reported_for_confirmation(tmp_path):
    rls = make_zip(tmp_path / "x.rls", {"project/env.py": "x=1", "project/m.slx": "x", "project/data.csv": "1"})
    assert inspect_archive(rls)["code_files"] == ["env.py", "m.slx"]


def test_entry_point_outside_project_is_rejected(tmp_path, other):
    rls = tmp_path / "x.rls"
    make_zip(rls, {}, project_yaml=False)
    with zipfile.ZipFile(rls, "a") as zf:
        zf.writestr("project/project.yaml", "name: X\nkind: python\nenv: {entry: '../../evil.py:make_env'}\n")
    with pytest.raises(ArchiveError, match="env.entry"):
        import_project(rls, other)


def test_loader_refuses_code_outside_project_folder(project):
    from rlstudio.envs.loader import load_module

    (project.path.parent / "evil.py").write_text("raise SystemExit('pwned')\n")
    with pytest.raises(ValueError, match="hors du dossier"):
        load_module(project, "../evil.py")


@pytest.mark.parametrize("make,msg", [
    (lambda p: p.write_bytes(b"pas un zip"), "pas un projet"),
    (lambda p: make_zip(p, {}, manifest=False), "Manifeste"),
    (lambda p: make_zip(p, {}, project_yaml=False), "project.yaml absent"),
])
def test_invalid_archives(tmp_path, other, make, msg):
    rls = tmp_path / "x.rls"
    make(rls)
    with pytest.raises(ArchiveError, match=msg):
        import_project(rls, other)


def test_newer_format_is_refused(tmp_path, other):
    rls = tmp_path / "x.rls"
    with zipfile.ZipFile(rls, "w") as zf:
        zf.writestr("rlstudio.json", json.dumps({"format": "rlstudio-project", "format_version": 99,
                                                "app_version": "9.9.9"}))
    with pytest.raises(ArchiveError, match="plus recente"):
        import_project(rls, other)


def test_size_limit_is_enforced(tmp_path, other, monkeypatch):
    monkeypatch.setattr(archive, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(ArchiveError, match="volumineuse"):
        import_project(make_zip(tmp_path / "x.rls", {"project/gros.bin": "x" * 500}), other)


def test_invalid_kind_is_rejected(tmp_path, other):
    rls = tmp_path / "x.rls"
    make_zip(rls, {}, project_yaml=False)
    with zipfile.ZipFile(rls, "a") as zf:
        zf.writestr("project/project.yaml", "name: X\nkind: autre\n")
    with pytest.raises(ArchiveError, match="kind"):
        import_project(rls, other)


# -- interface -----------------------------------------------------------------------------
def test_gui_import_with_and_without_confirmation(project, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.app import MainWindow
    from rlstudio.settings import Settings

    app = QApplication.instance() or QApplication([])
    out = export_project(project, tmp_path / "p.rls")
    win = MainWindow(ProjectStore(root=tmp_path / "c", templates=TEMPLATES_DIR), Settings(tmp_path / "s.json"))
    assert win.import_file(out, confirm=lambda report: False) is None  # refus de l'utilisateur
    assert win.store.list() == []
    asked = []
    imported = win.import_file(out, confirm=lambda report: asked.append(report) or True)
    assert imported is not None and imported.name == "Pendule"
    assert asked[0]["code_files"], "la confirmation doit recevoir la liste des fichiers de code"
    assert win.project.slug == imported.slug
    win.close()
    app.processEvents()
