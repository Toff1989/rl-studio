"""Template `cartpole_simulink` : modele tiers sous licence MIT, adapte. Controles statiques (sans MATLAB) et execution reelle (opt-in)."""
import os
import re
import zipfile
from pathlib import Path

import pytest
import yaml

from rlstudio.envs.spec import validate_spec
from rlstudio.paths import TEMPLATES_DIR
from rlstudio.project import ProjectStore

TEMPLATE = TEMPLATES_DIR / "cartpole_simulink"
MODEL = TEMPLATE / "pendulum.slx"


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load((TEMPLATE / "project.yaml").read_text(encoding="utf-8"))["env"]


@pytest.fixture(scope="module")
def xml():
    with zipfile.ZipFile(MODEL) as z:
        return "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".xml"))


def test_template_config_is_valid_and_listed(spec, tmp_path):
    assert validate_spec(spec) == []
    store = ProjectStore(root=tmp_path / "p", templates=TEMPLATES_DIR)
    assert "cartpole_simulink" in {t[0] for t in store.list_templates()}
    created = store.create("Pendule", "cartpole_simulink")
    assert (created.path / "pendulum.slx").exists() and (created.path / "LICENSE-modele.txt").exists()


# -- licence et provenance du modele tiers -----------------------------------------------------
def test_original_mit_license_and_provenance_are_shipped():
    lic = (TEMPLATE / "LICENSE-modele.txt").read_text(encoding="utf-8")
    assert lic.startswith("MIT License") and "Copyright (c) 2021 Jitendra Singh" in lic
    assert "Permission is hereby granted" in lic and "THE SOFTWARE IS PROVIDED" in lic  # avis complet conserve
    src = (TEMPLATE / "SOURCE.md").read_text(encoding="utf-8")
    assert "github.com/jitendra825/Inverted-Pendulum-Simulink" in src and "b34c2f7579d93626e06abe5a337bf8e9f913a04b" in src
    assert "Jitendra Singh" in (TEMPLATES_DIR.parent / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")


# -- le modele ne doit contenir ni code ni dependance cachee ---------------------------------------
def test_model_contains_no_code_or_unavailable_blocks():
    with zipfile.ZipFile(MODEL) as z:
        names = z.namelist()
        assert not [n for n in names if n.lower().endswith((".mex", ".mexw64", ".dll", ".exe", ".p", ".bat", ".ps1"))]
        assert not [n for n in names if n.startswith("/") or ".." in n.replace("\\", "/").split("/")]
    diagram = diagram_xml()
    callbacks = re.findall(r'<P Name="(?:PreLoadFcn|PostLoadFcn|InitFcn|StartFcn|StopFcn|CloseFcn|PreSaveFcn|OpenFcn|'
                           r'LoadFcn|CopyFcn|ModelCloseFcn|MaskInitialization|MaskCallbacks)"[^>]*>([^<]+)</P>', diagram)
    assert not [c for c in callbacks if c.strip()], callbacks
    block_types = set(re.findall(r'BlockType="([^"]+)"', diagram))
    assert block_types, "aucun bloc trouve : le format du modele a change, adapter diagram_xml()"
    assert not block_types & {"S-Function", "MATLABFcn"}, block_types
    assert "Stateflow" not in diagram and "EmbeddedMATLAB" not in diagram and "vrlib" not in diagram and "VR Sink" not in diagram


def diagram_xml():
    """Description du diagramme (blocs, parametres, callbacks) : hors configSet, qui ne contient que des noms d'options."""
    with zipfile.ZipFile(MODEL) as z:
        parts = [n for n in z.namelist() if n.endswith(".xml") and (n.startswith("simulink/systems/") or n.endswith("blockdiagram.xml"))]
        return "\n".join(z.read(n).decode("utf-8", "replace") for n in parts)


def test_spec_matches_the_model(spec, xml):
    """Chaque bloc et variable cites dans project.yaml existe dans le modele."""
    for item in spec["actions"] + spec["observations"]:
        block_name = item["block"].split("/", 1)[1]
        assert f'Name="{block_name}"' in xml, f"bloc introuvable dans le modele : {item['block']}"
    for var in spec["initial"]:
        assert var["variable"] in xml, var["variable"]
    assert re.search(r'<P Name="Bias">-pi</P>', xml), "l'angle observe doit etre l'ecart a la verticale (Bias -pi)"
    assert re.search(r'<P Name="InitialCondition">pi \+ theta0</P>', xml)


# -- execution reelle dans MATLAB (opt-in : demarre MATLAB, une dizaine de secondes) --------------------
@pytest.mark.skipif(os.environ.get("RLSTUDIO_TEST_MATLAB") != "1", reason="definir RLSTUDIO_TEST_MATLAB=1 (demarre MATLAB)")
def test_template_runs_in_matlab(tmp_path):
    import numpy as np

    from rlstudio.envs.loader import make_env

    project = ProjectStore(root=tmp_path / "p", templates=TEMPLATES_DIR).create("Pendule", "cartpole_simulink")
    env = make_env(project)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4,) and abs(obs[1]) <= 0.151 and abs(obs[0]) <= 0.101  # theta, x : etat initial tire
        steps, terminated = 0, False
        while not terminated and steps < 100:  # sans commande, le pendule vertical (instable) tombe
            obs, _, terminated, _, _ = env.step(np.zeros(1, dtype=np.float32))
            steps += 1
        assert terminated and steps < 60 and abs(obs[1]) > 0.5
    finally:
        env.close()
