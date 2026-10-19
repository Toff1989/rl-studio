"""Surrogates sur donnees synthetiques : ajustement, export Python/MATLAB coherent, aller-retour JSON."""
import numpy as np
import pytest

from rlstudio.distill.collect import Dataset
from rlstudio.distill.surrogates import (fit_linear, fit_polynomial, fit_symbolic, fit_table, fit_tree, r_squared, surrogate_from_dict)


def make_data(fn, n_eps=6, length=120, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    d = Dataset([f"x{i}" for i in range(dim)], ["u"], [-2.0], [2.0])
    for _ in range(n_eps):
        obs = rng.normal(size=(length, dim))
        d.add(obs, np.clip(fn(obs), -2, 2).reshape(-1, 1))
    return d


def test_linear_recovers_gains():
    d = make_data(lambda o: 0.5 * o[:, 0] - 1.2 * o[:, 1] + 0.1)
    s = fit_linear(d)
    pred = s.predict_episodes(d.obs_episodes())
    assert r_squared(d.actions(), pred) > 0.98
    assert s.complexity >= 3


def test_linear_with_integrator_is_stateful_and_resets():
    d = make_data(lambda o: 0.5 * o[:, 0])
    s = fit_linear(d, integrators=("x1",), dt=0.1)
    assert s.stateful
    o = np.array([1.0, 1.0, 0.0])
    s.reset()
    a1 = s.act(o)
    s.act(o)
    s.reset()
    assert np.allclose(s.act(o), a1)


def test_tree_table_symbolic_fit_nonlinear():
    fn = lambda o: np.sin(o[:, 0]) + 0.5 * o[:, 1] ** 2 - 0.5  # noqa: E731
    d = make_data(fn, n_eps=10, length=200)
    tree = fit_tree(d, max_depth=8, min_leaf=3)
    assert r_squared(d.actions(), tree.predict_episodes(d.obs_episodes())) > 0.9
    table = fit_table(d, k_dims=2)
    assert r_squared(d.actions(), table.predict_episodes(d.obs_episodes())) > 0.7
    sym = fit_symbolic(d, (6, 12))
    assert r_squared(d.actions(), sym[12].predict_episodes(d.obs_episodes())) > 0.95
    assert sym[6].complexity <= sym[12].complexity


def test_polynomial_recovers_cross_terms_and_caps_degree():
    fn = lambda o: 0.3 * o[:, 0] * o[:, 1] - 0.4 * o[:, 2] ** 2 + 0.2  # noqa: E731
    d = make_data(fn, n_eps=8, length=150)
    lin = fit_linear(d)
    poly = fit_polynomial(d, degree=2)
    obs = d.obs_episodes()
    assert r_squared(d.actions(), poly.predict_episodes(obs)) > 0.99
    assert r_squared(d.actions(), poly.predict_episodes(obs)) > r_squared(d.actions(), lin.predict_episodes(obs)) + 0.2
    assert poly.E.shape[0] == 9  # 3 + 6 monomes de degre 1 et 2 en 3 variables
    # trop de monomes : le degre est abaisse plutot que d'exporter un modele enorme
    assert fit_polynomial(d, degree=6, max_terms=20).degree < 6


@pytest.mark.parametrize("kind", ["linear", "polynomial", "tree", "table", "symbolic"])
def test_json_roundtrip_and_python_export_match(kind):
    d = make_data(lambda o: np.tanh(o[:, 0]) - 0.3 * o[:, 2], n_eps=5)
    s = {"linear": lambda: fit_linear(d), "polynomial": lambda: fit_polynomial(d, degree=3), "tree": lambda: fit_tree(d, max_depth=4, min_leaf=3),
         "table": lambda: fit_table(d, k_dims=2), "symbolic": lambda: fit_symbolic(d, (6,))[6]}[kind]()
    s2 = surrogate_from_dict(s.to_dict())
    obs = d.observations()[:50]
    s.reset(); s2.reset()
    a = np.array([s.act(o) for o in obs]); b = np.array([s2.act(o) for o in obs])
    assert np.allclose(a, b)
    # le code Python genere reproduit le modele
    ns = {}
    exec(s.to_python("policy"), ns)
    fn = ns["policy"]
    if isinstance(fn, type):
        inst = fn()
        out = np.array([np.atleast_1d(inst(o)) if callable(inst) else None for o in obs])
    else:
        out = np.array([np.atleast_1d(fn(o)) for o in obs])
    assert np.allclose(out.reshape(a.shape), a, atol=1e-6)
    assert "function" in s.to_matlab("policy")


# -- code MATLAB / Simulink genere (opt-in : demarre MATLAB) ------------------------------------
import os  # noqa: E402


@pytest.mark.skipif(os.environ.get("RLSTUDIO_TEST_MATLAB") != "1", reason="definir RLSTUDIO_TEST_MATLAB=1 (demarre MATLAB)")
def test_generated_matlab_and_simulink_match_python(tmp_path):
    import matlab
    from rlstudio.distill.export import build_simulink_controller
    from rlstudio.envs.simulink import MatlabSession

    d = make_data(lambda o: np.tanh(o[:, 0]) - 0.3 * o[:, 2] + 0.2 * o[:, 1] ** 2, n_eps=6, length=150)
    models = {"m_linear": fit_linear(d), "m_poly": fit_polynomial(d, degree=3), "m_tree": fit_tree(d, max_depth=4, min_leaf=3), "m_table": fit_table(d, k_dims=2),
              "m_sym": fit_symbolic(d, (6,))[6]}
    obs = d.observations()[:12]
    session = MatlabSession()
    try:
        session.eng.addpath(str(tmp_path), nargout=0)
        for name, sur in models.items():
            (tmp_path / f"{name}.m").write_text(sur.to_matlab(name), encoding="utf-8")
            sur.reset()
            for o in obs:
                got = np.asarray(session.eng.feval(name, matlab.double(o.reshape(-1, 1).tolist()), nargout=1)).ravel()
                assert np.allclose(got, sur.act(o), atol=1e-6), (name, got, sur.act(o))
        # controleur Simulink : une entree par observation, sortie = meme valeur que Python
        slx = build_simulink_controller(models["m_linear"], tmp_path / "ctrl.slx", session=session)
        assert slx.exists()
        o = obs[0]
        session.eng.eval("mdl = 'ctrl'; load_system(mdl);", nargout=0)
        t = matlab.double([[0.0] + o.tolist(), [1.0] + o.tolist()])
        session.eng.workspace["ut"] = t
        session.eng.eval("so = sim(mdl, 'LoadExternalInput', 'on', 'ExternalInput', 'ut', 'StopTime', '1', 'SaveOutput', 'on', 'OutputSaveName', 'yout'); y = so.yout{1}.Values.Data(1, :);",
                         nargout=0)
        y = np.asarray(session.eng.workspace["y"]).ravel()
        assert np.allclose(y, models["m_linear"].act(o), atol=1e-5)
    finally:
        session.close()
