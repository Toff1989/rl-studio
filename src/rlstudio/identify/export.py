"""Export d'un modele identifie vers Simulink : bloc MATLAB Function (compatible generation de code).

- regime permanent : Inport (une par entree) -> Mux -> MATLAB Function `y = fcn(u)` -> Demux -> Outport (une par sortie) ;
- dynamique        : meme structure, la fonction garde l'etat (`persistent`) et avance d'un pas a chaque pas d'echantillonnage `dt` :
                     entrees = actions, sorties = etat (observations) ; etat initial = `x0` (moyenne des etats initiaux du balayage).

La construction du modele passe par matlab/rlstudio_build_controller.m (meme generateur que les controleurs extraits).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..distill.surrogates import M_RESERVED, clean_names, vec_m
from .models import DynModel


def block_code(model: DynModel, x0=None) -> str:
    """Texte du script de la MATLAB Function : `function y = fcn(u)`."""
    if model.static:
        return model.to_matlab("fcn")
    x0 = model.x0 if x0 is None else x0
    if x0 is None:
        raise ValueError("Etat initial inconnu : renseigner x0.")
    step = model.to_matlab("dynamics_step").replace("%#codegen\n", "")
    return "\n".join([
        "function x = fcn(u)",
        f"% {model._header()}",
        "% Sortie = etat courant ; l'etat avance d'un pas a chaque pas d'echantillonnage.",
        "%#codegen",
        "persistent xs;",
        "if isempty(xs)",
        f"    xs = {vec_m(x0)};",
        "end",
        "x = xs;",
        "xs = dynamics_step(xs, u(:));",
        "end",
        "",
        step,
    ])


def build_spec(model: DynModel, x0=None, sample_time=None) -> dict:
    """Specification JSON lue par rlstudio_build_controller.m (`obs_names` = entrees du bloc, `act_names` = ses sorties)."""
    if model.static:
        ins, outs = model.act_names, model.obs_names
        ts = sample_time if sample_time else -1.0
    else:
        ins, outs = model.act_names, model.obs_names
        ts = sample_time if sample_time else model.dt
        if not ts or ts <= 0:
            raise ValueError("Un modele dynamique exige un pas d'echantillonnage > 0.")
    return {"obs_names": clean_names(ins, M_RESERVED), "act_names": clean_names(outs, M_RESERVED), "code": block_code(model, x0),
            "sample_time": float(ts), "description": "Modele reduit identifie par RL Studio" +
            (" (regime permanent y = f(u))." if model.static else " (dynamique x(k+1) = x(k) + f(x, u)).")}


def build_simulink(model: DynModel, out_slx, x0=None, sample_time=None, session=None, log=print) -> Path:
    """Genere le .slx (necessite MATLAB via le moteur Python)."""
    from ..envs.simulink import MatlabSession

    out = Path(out_slx).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    spec_file = out.with_suffix(".spec.json")
    spec_file.write_text(json.dumps(build_spec(model, x0, sample_time)), encoding="utf-8")
    own = session is None
    session = session or MatlabSession()
    try:
        log(f"Generation du modele Simulink : {out}")
        session.eng.rlstudio_build_controller(str(out), str(spec_file), nargout=0)
    finally:
        if own:
            session.close()
        spec_file.unlink(missing_ok=True)
    return out
