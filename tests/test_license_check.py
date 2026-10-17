"""Le controle de conformite Qt/LGPL doit refuser un module GPL seulement et exiger les textes de licence."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("check_bundle", Path(__file__).parents[1] / "packaging" / "check_bundle.py")
check_bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_bundle)


def make_bundle(root: Path, extra_dlls=(), plugin_dirs=("platforms", "styles"), with_licenses=True):
    py = root / "_internal" / "PySide6"
    (py / "plugins").mkdir(parents=True)
    for module in ("Core", "Gui", "Widgets", *extra_dlls):
        (py / f"Qt6{module}.dll").write_text("")
    for d in plugin_dirs:
        (py / "plugins" / d).mkdir()
    if with_licenses:
        (root / "licenses").mkdir()
        for name in ("LICENSE", "licenses/LGPL-3.0.txt", "licenses/GPL-3.0.txt", "THIRD_PARTY_NOTICES.md",
                     "resources/templates/cartpole_simulink/LICENSE-modele.txt"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("x")
    return root


def test_clean_bundle_passes(tmp_path):
    assert check_bundle.check(make_bundle(tmp_path)) == []


def test_virtual_keyboard_is_rejected(tmp_path):
    problems = check_bundle.check(make_bundle(tmp_path, extra_dlls=("VirtualKeyboard",),
                                              plugin_dirs=("platforms", "platforminputcontexts")))
    assert any("VirtualKeyboard" in p for p in problems)
    assert any("platforminputcontexts" in p for p in problems)


def test_unknown_module_requires_review(tmp_path):
    assert any("Charts" in p for p in check_bundle.check(make_bundle(tmp_path, extra_dlls=("Charts",))))


def test_missing_third_party_model_license_is_reported(tmp_path):
    bundle = make_bundle(tmp_path)
    (bundle / "resources" / "templates" / "cartpole_simulink" / "LICENSE-modele.txt").unlink()
    assert any("LICENSE-modele.txt" in p for p in check_bundle.check(bundle))


def test_missing_license_texts_are_reported(tmp_path):
    problems = check_bundle.check(make_bundle(tmp_path, with_licenses=False))
    assert any("LGPL-3.0.txt" in p for p in problems)


def test_repository_ships_license_texts_and_notices():
    root = Path(__file__).parents[1]
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in (root / "licenses" / "LGPL-3.0.txt").read_text(encoding="utf-8")
    assert "LGPL" in (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
