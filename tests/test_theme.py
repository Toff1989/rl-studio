from rlstudio.gui import theme
from rlstudio.settings import Settings


def test_derive_fills_all_tokens_for_every_builtin_theme():
    for name, base in theme.BUILTIN.items():
        t = theme.derive(base)
        for key in ("surface2", "border", "muted", "accent2", "on_accent", "accent_rgb", "danger_text"):
            assert key in t, (name, key)
        assert "@" not in theme.stylesheet(t), name  # aucun jeton non remplace


def test_custom_theme_only_needs_a_few_colors():
    t = theme.derive({"bg": "#101010", "accent": "#ff00aa"})  # le reste vient des valeurs par defaut
    assert t["surface"] and t["font"] == "Segoe UI"
    assert t["accent_rgb"] == "255, 0, 170"


def test_text_on_accent_is_readable():
    assert theme.derive({"accent": "#ffff00"})["on_accent"] == "#000000"
    assert theme.derive({"accent": "#1a237e"})["on_accent"] == "#ffffff"


def test_settings_roundtrip_and_corrupt_file(tmp_path):
    f = tmp_path / "s" / "settings.json"
    s = Settings(f)
    s.set("theme", "Matrix")
    s.set("custom_themes", {"Perso": {"bg": "#000000"}})
    again = Settings(f)
    assert again.get("theme") == "Matrix" and again.get("custom_themes")["Perso"]["bg"] == "#000000"
    f.write_text("{pas du json", encoding="utf-8")
    assert Settings(f).get("theme", "Sombre") == "Sombre"  # fichier corrompu : valeurs par defaut
