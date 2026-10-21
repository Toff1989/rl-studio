"""Conversion d'un modele Simulink existant : specification, validation, project.yaml (sans MATLAB)."""
import pytest
import yaml

from rlstudio import convert
from rlstudio.envs.spec import validate_spec

ANALYSIS = {
    "inports": [{"path": "m/u", "name": "u", "port": 1, "dims": "-1", "convertible": True},
                {"path": "m/vec", "name": "vec", "port": 2, "dims": "[3]", "convertible": False}],
    "outports": [{"path": "m/y", "name": "y", "source": "m/Integrator", "source_port": 1, "observable": True},
                 {"path": "m/dead", "name": "dead", "source": "", "source_port": 1, "observable": False}],
}


def _spec(**over):
    spec = {
        "project_name": "Mon modele", "model_path": r"C:\modeles\m.slx", "init_path": "",
        "actions": [{"name": "u", "kind": "inport", "path": "m/u", "param": "Value", "default": 0.0, "low": -2.0, "high": 2.0},
                    {"name": "gain", "kind": "param", "path": "m/Gain", "param": "Gain", "low": 0.0, "high": 5.0}],
        "observations": [{"name": "y", "path": "m/y", "port": 1}],
        "initial": [{"variable": "x0", "low": 0.0, "high": 1.0}],
        "variables": ["a", "b"], "remove": ["m/Scope"],
        "step_s": 0.5, "episode_s": 20.0, "solver": "fixed", "solver_step": 0.01,
        "reward": "-(y - 3.0)**2 - 0.01*u**2", "terminate": "abs(y) > 100",
    }
    spec.update(over)
    return spec


def test_sanitize_and_unique_names():
    assert convert.sanitize_name("Angle theta (rad)") == "Angle_theta_rad"
    assert convert.sanitize_name("1er signal") == "signal_1er_signal"
    assert convert.sanitize_name("") == "signal"
    assert convert.sanitize_name("t") == "t_" and convert.sanitize_name("max") == "max_"  # mots reserves des expressions
    assert convert.unique_names(["y", "y", "z", "y"]) == ["y", "y_2", "z", "y_3"]


def test_default_spec_keeps_only_usable_signals():
    prop = convert.default_spec(ANALYSIS)
    assert [a["name"] for a in prop["actions"]] == ["u"] and prop["actions"][0]["kind"] == "inport"
    assert [o["name"] for o in prop["observations"]] == ["y"]  # sortie non reliee et entree vectorielle ecartees


def test_validate_conversion_ok_and_errors():
    assert convert.validate_conversion(_spec()) == []
    assert any("nom" in e for e in convert.validate_conversion(_spec(project_name=" ")))
    assert any("action" in e for e in convert.validate_conversion(_spec(actions=[])))
    bad_range = _spec(actions=[{"name": "u", "kind": "inport", "path": "m/u", "low": 1.0, "high": 1.0}])
    assert any("minimum" in e for e in convert.validate_conversion(bad_range))
    dup = _spec(observations=[{"name": "u", "path": "m/y", "port": 1}])
    assert any("uniques" in e for e in convert.validate_conversion(dup))
    assert any("invalide" in e for e in convert.validate_conversion(_spec(observations=[{"name": "a b", "path": "m/y"}])))
    assert any("Récompense" in e for e in convert.validate_conversion(_spec(reward="-(z - 1)**2")))  # nom inconnu
    assert any("pas du solveur" in e.lower() for e in convert.validate_conversion(_spec(solver_step=1.0)))
    assert any("pisode" in e.lower() for e in convert.validate_conversion(_spec(episode_s=0.1)))
    assert any("récompense" in e.lower() for e in convert.validate_conversion(_spec(reward="")))


def test_matlab_spec_only_converts_inports():
    ms = convert.matlab_spec(_spec())
    assert ms["inports"] == [{"path": "m/u", "name": "u", "default": 0.0}]  # le parametre de bloc reste en place
    assert ms["remove"] == ["m/Scope"] and ms["variables"] == ["a", "b"] and ms["solver"] == "fixed"
    assert ms["observations"] == [{"name": "y", "path": "m/y", "port": 1}]


def test_write_project_builds_a_valid_simulink_project(tmp_path):
    result = {"model": "m", "actions": [{"name": "u", "block": "m/Rl_u", "param": "Value"}],
              "observations": [{"name": "y", "block": "m/Integrator", "port": 1}], "warnings": []}
    cfg = convert.write_project(tmp_path, _spec(), result)
    saved = yaml.safe_load((tmp_path / "project.yaml").read_text(encoding="utf-8"))
    assert saved["kind"] == "simulink" and saved["name"] == "Mon modele" and saved["algo"]["name"] == "SAC"
    env = saved["env"]
    assert env["model"] == "m.slx"
    assert env["actions"][0] == {"name": "u", "block": "m/Rl_u", "param": "Value", "low": -2.0, "high": 2.0}
    assert env["actions"][1]["block"] == "m/Gain" and env["actions"][1]["param"] == "Gain"
    assert env["observations"][0] == {"name": "y", "block": "m/Integrator", "port": 1, "size": 1}  # sortie resolue en son bloc source
    assert env["initial"] == [{"variable": "x0", "low": 0.0, "high": 1.0}]
    assert validate_spec(env) == [] and cfg["env"] == env


def test_write_project_rejects_inconsistent_environment(tmp_path):
    result = {"model": "m", "actions": [{"name": "u", "block": "m/Rl_u", "param": "Value"}],
              "observations": [{"name": "y", "block": "m/Integrator", "port": 1}]}
    with pytest.raises(ValueError):
        convert.write_project(tmp_path, _spec(reward="-(zzz)**2"), result)


def test_convert_dialog_builds_a_valid_spec_from_an_analysis(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from rlstudio.gui.convert_dialog import ConvertDialog
    from rlstudio.project import ProjectStore
    from rlstudio.settings import Settings

    app = QApplication.instance() or QApplication([])
    analysis = {
        **ANALYSIS, "model": "m",
        "constants": [{"path": "m/perturbation", "name": "perturbation", "value": 0.1, "expression": "0.1", "is_variable": False},
                      {"path": "m/u2", "name": "u", "value": 2.0, "expression": "2", "is_variable": False}],
        "solver": {"type": "Variable-step", "name": "ode45", "step": "auto", "stop": "20"},
        "variables": [{"name": "a", "in_model": False, "in_base": True, "scalar": True, "value": 0.5},
                      {"name": "k", "in_model": True, "in_base": False, "scalar": True, "value": 2.0}],
        "removable": [{"path": "m/Scope", "type": "Scope", "reason": "affichage"}], "warnings": [],
    }
    dlg = ConvertDialog(None, Settings(tmp_path / "s.json"), ProjectStore(root=tmp_path / "projets"))
    dlg.analysis = analysis
    dlg._fill(analysis, convert.default_spec(analysis))
    assert dlg.t_actions.rowCount() == 3 and dlg.t_obs.rowCount() == 1
    names = [dlg.t_actions.item(r, 1).text() for r in range(3)]
    assert names == ["u", "perturbation", "u_2"]  # une constante au nom deja pris est renommee
    # seules les lignes cochees partent dans la specification : l'entree, pas les constantes
    assert [a["name"] for a in dlg._actions_spec()] == ["u"]
    dlg.t_actions.item(1, 0).setCheckState(Qt.Checked)
    assert [a["name"] for a in dlg._actions_spec()] == ["u", "perturbation"]
    assert dlg._actions_spec()[1]["kind"] == "param" and dlg._actions_spec()[1]["path"] == "m/perturbation"
    dlg.t_init.item(0, 0).setCheckState(Qt.Checked)  # variable a (workspace de base) : tiree au hasard et embarquee
    assert dlg._initial_spec()[0]["variable"] == "a" and dlg._embedded_variables() == ["a"]
    dlg.f_model.setText(r"C:\modeles\m.slx")
    dlg.f_reward.setText("-(y - 1.0)**2")
    spec = dlg._spec()
    assert spec["remove"] == ["m/Scope"] and spec["observations"][0]["path"] == "m/y"
    assert convert.validate_conversion(spec) == []
    dlg.t_actions.item(0, 5).setText("pas un nombre")
    with pytest.raises(ValueError):
        dlg._actions_spec()
    app.processEvents()
