"""Onglet « Identification » hors ecran : signaux du projet, options, chargement d'un rapport."""
import os
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rlstudio.identify import IdentifyOptions, SteadyOptions, identify_env, steady_env  # noqa: E402
from rlstudio.identify.pipeline import identify_dir  # noqa: E402
from rlstudio.paths import TEMPLATES_DIR  # noqa: E402
from rlstudio.project import ProjectStore  # noqa: E402
from rlstudio.settings import Settings  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_identify import Pendulum  # noqa: E402
from test_identify_steady import Plant  # noqa: E402


@pytest.fixture
def tab(tmp_path):
    from rlstudio.gui.identify_tab import IdentifyTab

    app = QApplication.instance() or QApplication([])
    store = ProjectStore(root=tmp_path / "projects", templates=TEMPLATES_DIR)
    project = store.create("sim", "cartpole_simulink")
    t = IdentifyTab(Settings(tmp_path / "s.json"))
    t.set_project(project)
    yield t, project, app
    t.deleteLater()


def test_signals_read_from_simulink_config(tab):
    t, project, _ = tab
    acts = project.cfg["env"]["actions"]
    assert t.in_table.rowCount() == len(acts) and t.out_list.count() >= 1
    o = t.options()
    assert o["mode"] == "steady" and o["ranges"] == {} and o["fixed"] == {} and "outputs" not in o
    assert o["methods"] == ["linear", "polynomial", "sparse", "neural"] and o["poly_degrees"] == [2, 3] and o["hidden"] == [32, 32]


def test_options_reflect_ui(tab):
    t, project, _ = tab
    name = project.cfg["env"]["actions"][0]["name"]
    t.in_table.item(0, 2).setText("0.5")           # plage restreinte
    t.s_samples.setValue(77)
    t.s_rtol.setValue(0.5)
    t.s_scenario.setChecked(True)
    t.f_poly.setText("2,4")
    t.method_boxes["neural"].setChecked(False)
    t.out_list.item(0).setCheckState(Qt.Unchecked)
    o = t.options()
    assert o["ranges"][name][0] == 0.5 and o["samples"] == 77 and abs(o["rtol"] - 0.005) < 1e-9 and o["scenario"]
    assert o["poly_degrees"] == [2, 4] and "neural" not in o["methods"] and t.out_list.item(0).text() not in o["outputs"]
    t.in_table.item(0, 0).setCheckState(Qt.Unchecked)
    t.in_table.item(0, 4).setText("0.25")
    if t.in_table.rowCount() == 1:  # tout fixe : rien a balayer
        with pytest.raises(ValueError):
            t.options()
        t.in_table.item(0, 0).setCheckState(Qt.Checked)
    else:
        assert t.options()["fixed"] == {name: 0.25}
    t.mode.setCurrentIndex(1)
    d = t.options()
    assert d["mode"] == "dynamic" and "episodes" in d and "samples" not in d and "outputs" not in d


@pytest.mark.parametrize("bad", ["poly", "hidden", "sparse", "min>max", "methods", "outputs"])
def test_invalid_options_raise(tab, bad):
    t, _, _ = tab
    if bad == "poly":
        t.f_poly.setText("a,b")
    elif bad == "hidden":
        t.f_hidden.setText("0")
    elif bad == "sparse":
        t.f_sparse.setText("-1")
    elif bad == "min>max":
        t.in_table.item(0, 2).setText("5")
        t.in_table.item(0, 3).setText("1")
    elif bad == "methods":
        for cb in t.method_boxes.values():
            cb.setChecked(False)
    else:
        for i in range(t.out_list.count()):
            t.out_list.item(i).setCheckState(Qt.Unchecked)
    with pytest.raises(ValueError):
        t.options()


def test_loads_steady_and_dynamic_reports(tab):
    t, project, app = tab
    base = identify_dir(project)
    steady_env(Plant(), SteadyOptions(methods=("linear", "sparse"), samples=60, val_samples=15, sparse_thresholds=(0.1,)), name="x",
               out_dir=base / "20260101-100000", log=lambda m: None)
    identify_env(Pendulum(), IdentifyOptions(methods=("linear", "polynomial"), episodes=8, val_episodes=3, poly_degrees=(2,)), name="x",
                 out_dir=base / "20260102-100000", log=lambda m: None)
    t.set_project(project)
    assert t.result_combo.count() == 2
    assert not t._steady() and t.table.rowCount() == 2 and "Fidélité" in t.table.horizontalHeaderItem(3).text()
    assert t._candidate() is not None and t.equations.toPlainText()
    t.restyle()
    app.processEvents()
    t.result_combo.setCurrentIndex(1)
    assert t._steady() and t.table.rowCount() == 2 and t.plot_out.count() == 2 and t.plot_out.isVisibleTo(t) is True
    t.table.selectRow(1)
    assert "y1 = " in t.equations.toPlainText() or "y2 = " in t.equations.toPlainText()
    app.processEvents()


def test_options_roundtrip_through_project_file(tab):
    t, project, _ = tab
    name = project.cfg["env"]["actions"][0]["name"]
    t.s_samples.setValue(123)
    t.in_table.item(0, 2).setText("0.2")
    t.f_hidden.setText("16, 8")
    t.method_boxes["linear"].setChecked(False)
    t.out_list.item(0).setCheckState(Qt.Unchecked)
    opts = t.options()
    t._save_options(opts)
    assert (identify_dir(project) / "options.json").is_file()
    t2 = type(t)(t.settings)
    t2.set_project(project)
    assert t2.options() == opts and t2.s_samples.value() == 123 and name in t2.options()["ranges"]
    t2.deleteLater()
