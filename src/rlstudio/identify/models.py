"""Modeles de dynamique  x(k+1) = x(k) + f(x(k), u(k))  (NumPy pur) : lineaire, polynome, equation parcimonieuse, reseau.

Entrees z = [x, u] (observations puis actions), sortie : l'increment  delta = x(k+1) - x(k).
Chaque modele sait predire, se decrire (equations), se generer en Python / MATLAB et se serialiser en JSON.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..distill.surrogates import M_RESERVED, PY_RESERVED, clean_names, mat_m, mat_py, num, r_squared, vec_m, vec_py

RESERVED_PY = PY_RESERVED | {"x", "xn", "d", "h", "MU", "SD", "DSD"}
RESERVED_M = M_RESERVED | {"x", "xn", "d", "h", "MU", "SD", "DSD"}


class DynModel:
    """Base : un modele de l'increment  delta = f(z),  z = [x, u]."""

    kind = "dynamics"

    def __init__(self, obs_names, act_names, dt, mu, sd, label="", static=False):
        self.static = bool(static)  # vrai : carte entrees -> sorties en regime permanent, y = f(u) (pas d'etat, pas d'increment)
        self.obs_names, self.act_names = [str(n) for n in obs_names], [str(n) for n in act_names]
        self.dt = float(dt)
        self.mu, self.sd = np.asarray(mu, float).ravel(), np.maximum(np.asarray(sd, float).ravel(), 1e-12)
        self.label = label or self.kind
        self.x0 = None  # etat initial par defaut (modele dynamique) : moyenne des etats initiaux du balayage

    @property
    def n_x(self):
        return len(self.obs_names)

    @property
    def n_u(self):
        return len(self.act_names)

    @property
    def in_names(self):
        return self.act_names if self.static else self.obs_names + self.act_names

    def predict(self, U):
        """Mode statique : sorties en regime permanent pour des entrees U (N, m)."""
        return self.delta(np.atleast_2d(np.asarray(U, float)))

    # -- evaluation ---------------------------------------------------------------------------
    def delta(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, x, u) -> np.ndarray:
        z = np.concatenate([np.asarray(x, float).ravel(), np.asarray(u, float).ravel()])[None, :]
        return np.asarray(x, float).ravel() + self.delta(z)[0]

    def step_batch(self, X, U) -> np.ndarray:
        return np.asarray(X, float) + self.delta(np.hstack([X, U]))

    def rollout(self, x0, U, clip=None):
        """Simulation libre : l'etat predit est reinjecte ; `clip=(low, high)` evite les debordements numeriques."""
        x = np.asarray(x0, float).ravel()
        out = [x]
        for u in np.asarray(U, float):
            x = self.step(x, u)
            if clip is not None:
                x = np.clip(x, *clip)
            out.append(x)
        return np.array(out)

    # -- description --------------------------------------------------------------------------
    @property
    def complexity(self) -> int:
        raise NotImplementedError

    def title(self) -> str:
        return self.label

    def equations(self) -> list[str]:
        raise NotImplementedError

    # -- generation de code --------------------------------------------------------------------
    def code(self, lang, xs, us):
        """(constantes, corps) ; le corps remplit `d` (d[k] en Python, d(k+1) en MATLAB)."""
        raise NotImplementedError

    def _header(self):
        if self.static:
            return (f"{self.title()} - identifié par RL Studio. Régime permanent y = f(u). Entrées : {', '.join(self.act_names)} ; "
                    f"sorties : {', '.join(self.obs_names)}.")
        return (f"{self.title()} - identifié par RL Studio. x(k+1) = x(k) + f(x(k), u(k)), pas {num(self.dt)} s. "
                f"États : {', '.join(self.obs_names)} ; entrées : {', '.join(self.act_names)}.")

    def to_python(self, name="dynamics") -> str:
        names = clean_names(self.in_names, RESERVED_PY)
        xs, us = ([], names) if self.static else (names[: self.n_x], names[self.n_x:])
        consts, body = self.code("py", xs, us)
        if self.static:
            name = name if name != "dynamics" else "steady_state"
            lines = [f'"""{self._header()}"""', "import numpy as np", ""] + consts + ["", "", f"def {name}(u):"]
            lines += [f"    {n} = float(u[{i}])" for i, n in enumerate(us)] + [f"    d = np.zeros({self.n_x})"] + ["    " + ln for ln in body]
            return "\n".join(lines + ["    return d", ""])
        lines = [f'"""{self._header()}"""', "import numpy as np", "", f"DT = {num(self.dt)}"] + consts + ["", "", f"def {name}(x, u):"]
        lines += [f"    {n} = float(x[{i}])" for i, n in enumerate(xs)] + [f"    {n} = float(u[{i}])" for i, n in enumerate(us)]
        lines += [f"    d = np.zeros({self.n_x})"] + ["    " + ln for ln in body]
        lines += ["    return np.asarray(x, float) + d", ""]
        return "\n".join(lines)

    def to_matlab(self, name="dynamics") -> str:
        names = clean_names(self.in_names, RESERVED_M)
        xs, us = ([], names) if self.static else (names[: self.n_x], names[self.n_x:])
        consts, body = self.code("m", xs, us)
        if self.static:
            name = name if name != "dynamics" else "steady_state"
            lines = [f"function y = {name}(u)", f"% {self._header()}", "%#codegen"] + [f"{n} = u({i + 1});" for i, n in enumerate(us)]
            return "\n".join(lines + consts + [f"d = zeros({self.n_x}, 1);"] + body + ["y = d;", "end", ""])
        lines = [f"function xn = {name}(x, u)", f"% {self._header()}", "%#codegen"]
        lines += [f"{n} = x({i + 1});" for i, n in enumerate(xs)] + [f"{n} = u({i + 1});" for i, n in enumerate(us)]
        lines += consts + [f"d = zeros({self.n_x}, 1);"] + body + ["xn = x(:) + d;", "end", ""]
        return "\n".join(lines)

    # -- serialisation -------------------------------------------------------------------------
    def _base(self):
        return {"kind": self.kind, "label": self.label, "obs_names": self.obs_names, "act_names": self.act_names, "dt": self.dt,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "static": self.static,
                "x0": None if self.x0 is None else [float(v) for v in self.x0]}

    def to_dict(self) -> dict:
        raise NotImplementedError


# ================================================================================================
#  Modeles a termes : lineaire, polynome, equation parcimonieuse (meme classe, bibliotheques differentes)
# ================================================================================================
# Un terme : ("const",) | ("mono", ((i, p), ...)) | (fonction, i) avec fonction dans sin, cos, tanh, abs.
def _term_values(term, S):
    k = term[0]
    if k == "const":
        return np.ones(len(S))
    if k == "fx":  # fonction(s_i) * s_j
        return _term_values((term[1], term[2]), S) * S[:, term[3]]
    if k == "mono":
        out = np.ones(len(S))
        for i, p in term[1]:
            out = out * S[:, i] ** p
        return out
    f = {"sin": np.sin, "cos": np.cos, "tanh": np.tanh, "abs": np.abs}[k]
    return f(S[:, term[1]])


def _term_text(term, syms, lang):
    k = term[0]
    if k == "const":
        return "1"
    if k == "fx":
        return f"{_term_text((term[1], term[2]), syms, lang)}*{syms[term[3]]}"
    if k == "mono":
        pw = "**" if lang == "py" else "^"
        return "*".join(syms[i] if p == 1 else f"{syms[i]}{pw}{p}" for i, p in term[1])
    fn = {"sin": "np.sin", "cos": "np.cos", "tanh": "np.tanh", "abs": "np.abs"}[k] if lang == "py" else k
    return f"{fn}({syms[term[1]]})"


def _jsonable(term):
    return [term[0], [list(t) for t in term[1]]] if term[0] == "mono" else list(term)


def _from_json(t):
    if t[0] == "mono":
        return ("mono", tuple((int(i), int(p)) for i, p in t[1]))
    if t[0] == "fx":
        return ("fx", t[1], int(t[2]), int(t[3]))
    return (t[0],) if t[0] == "const" else (t[0], int(t[1]))


class TermModel(DynModel):
    """delta_k = sum_t C[k, t] * terme_t(s),  s = z brut (normalize=False) ou (z - mu) / sd (normalize=True)."""

    kind = "terms"

    def __init__(self, obs_names, act_names, dt, mu, sd, terms, C, normalize, label="", static=False):
        super().__init__(obs_names, act_names, dt, mu, sd, label, static)
        self.terms, self.C, self.normalize = list(terms), np.asarray(C, float), bool(normalize)

    def _s(self, Z):
        Z = np.asarray(Z, float)
        return (Z - self.mu) / self.sd if self.normalize else Z

    def delta(self, Z):
        S = self._s(Z)
        used = np.flatnonzero(np.abs(self.C).sum(0) > 0)
        Phi = np.column_stack([_term_values(self.terms[t], S) for t in used]) if len(used) else np.zeros((len(S), 0))
        return Phi @ self.C[:, used].T

    @property
    def complexity(self):
        return int(np.count_nonzero(self.C))

    def _syms(self, names, lang):
        return [f"z_{n}" for n in names] if self.normalize else list(names)

    def _expr(self, k, syms, lang, digits=None):
        parts = []
        for t in np.flatnonzero(self.C[k]):
            c = self.C[k, t]
            txt = _term_text(self.terms[t], syms, lang)
            cs = f"{c:.{digits}g}" if digits else num(c)
            parts.append(cs if txt == "1" else f"{cs}*{txt}")
        return " + ".join(parts).replace("+ -", "- ") if parts else "0"

    def equations(self):
        names = self.in_names
        syms = self._syms(names, "m")
        out = []
        if self.normalize:
            out.append("variables réduites : " + " ; ".join(f"z_{n} = ({n} - {self.mu[i]:.4g}) / {self.sd[i]:.4g}" for i, n in enumerate(names)))
        if self.static:
            return out + [f"{n} = {self._expr(k, syms, 'm', 5)}" for k, n in enumerate(self.obs_names)]
        out += [f"{n}(k+1) = {n}(k) + [{self._expr(k, syms, 'm', 5)}]" for k, n in enumerate(self.obs_names)]
        out.append(f"(pas de temps : {self.dt:g} s ; delta/dt ~ dérivée de l'état)")
        return out

    def code(self, lang, xs, us):
        py = lang == "py"
        names = xs + us
        syms = self._syms(names, lang)
        consts = [f"MU = {vec_py(self.mu) if py else vec_m(self.mu)}" + ("" if py else ";"),
                  f"SD = {vec_py(self.sd) if py else vec_m(self.sd)}" + ("" if py else ";")] if self.normalize else []
        body = []
        if self.normalize:
            body += [f"z_{n} = ({n} - MU[{i}]) / SD[{i}]" if py else f"z_{n} = ({n} - MU({i + 1})) / SD({i + 1});" for i, n in enumerate(names)]
        for k in range(self.n_x):
            body.append(f"d[{k}] = {self._expr(k, syms, lang)}" if py else f"d({k + 1}) = {self._expr(k, syms, lang)};")
        return consts, body

    def to_dict(self):
        d = self._base()
        d.update(terms=[_jsonable(t) for t in self.terms], C=self.C.tolist(), normalize=self.normalize)
        return d


def _linear_library(n):
    return [("const",)] + [("mono", ((i, 1),)) for i in range(n)]


def _poly_library(n, degree, max_terms=400):
    while degree > 1:
        from math import comb

        if comb(n + degree, degree) <= max_terms:
            break
        degree -= 1
    terms = [("const",)]
    for dg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n), dg):
            terms.append(("mono", tuple((i, combo.count(i)) for i in sorted(set(combo)))))
    return terms


def _sparse_library(n):
    terms = _poly_library(n, 3, max_terms=200)
    for i in range(n):
        terms += [("sin", i), ("cos", i), ("tanh", i), ("abs", i)]
    if n <= 8:  # produits fonction x variable (ex. sin(theta)*u) : non-linearites couplees frequentes en physique
        terms += [("fx", f, i, j) for f in ("sin", "cos", "tanh", "abs") for i in range(n) for j in range(n) if i != j]
    return terms


def _design(terms, S):
    Phi = np.column_stack([_term_values(t, S) for t in terms])
    scale = np.sqrt((Phi ** 2).mean(0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return Phi / scale, scale


def _solve(Phi, Y, active, ridge):
    A = Phi[:, active]
    coef = np.linalg.solve(A.T @ A + ridge * len(A) * np.eye(A.shape[1]), A.T @ Y)
    return coef


def _fit_terms(data, terms, normalize, ridge, threshold=0.0, label=""):
    static = bool(getattr(data, "static", False))
    Z, D = data.Z(), data.delta()
    mu, sd = Z.mean(0), Z.std(0)
    S = (Z - mu) / np.where(sd > 1e-12, sd, 1.0) if normalize else Z
    sd_eff = np.where(sd > 1e-12, sd, 1.0)
    dsd = np.maximum(D.std(0), 1e-12)
    Phi, scale = _design(terms, S)
    Yn = D / dsd
    C = np.zeros((D.shape[1], len(terms)))
    for k in range(D.shape[1]):
        active = np.arange(len(terms))
        coef = _solve(Phi, Yn[:, k], active, ridge)
        for _ in range(10 if threshold > 0 else 0):  # STLSQ : seuillage sequentiel
            big = np.abs(coef) >= threshold
            if not big.any():
                active = active[:0]
                break
            if big.all():
                break
            active = active[big]
            coef = _solve(Phi, Yn[:, k], active, ridge)
        c = np.zeros(len(terms))
        if len(active):
            c[active] = coef
        C[k] = c * dsd[k] / scale
    return TermModel(data.obs_names, data.act_names, data.dt, mu, sd_eff, terms, C, normalize, label, static)


def fit_linear(data, ridge=1e-9) -> TermModel:
    n = data.n_in
    return _fit_terms(data, _linear_library(n), False, ridge, label="Linéaire (A.x + B.u + c)")


def fit_polynomial(data, degree=3, ridge=1e-8) -> TermModel:
    n = data.n_in
    return _fit_terms(data, _poly_library(n, degree), True, ridge, label=f"Polynôme degré {degree}")


def fit_sparse(data, threshold, ridge=1e-8) -> TermModel:
    """SINDy : bibliotheque (monomes jusqu'au degre 3, sin, cos, tanh, abs) + regression a seuillage sequentiel."""
    n = data.n_in
    return _fit_terms(data, _sparse_library(n), False, ridge, threshold=threshold, label=f"Équation parcimonieuse (seuil {threshold:g})")


# ================================================================================================
#  Reseau de neurones (perceptron tanh, entraine par Adam avec bruit d'entree)
# ================================================================================================
class NeuralDynModel(DynModel):
    kind = "neural"

    def __init__(self, obs_names, act_names, dt, mu, sd, dsd, layers, label="", static=False):
        """layers : [(W (out,in), b (out,))] ; tanh partout sauf la derniere (lineaire). delta = dsd * sortie."""
        super().__init__(obs_names, act_names, dt, mu, sd, label or "Réseau de neurones", static)
        self.dsd = np.maximum(np.asarray(dsd, float).ravel(), 1e-12)
        self.layers = [(np.asarray(W, float), np.asarray(b, float)) for W, b in layers]

    def delta(self, Z):
        h = (np.asarray(Z, float) - self.mu) / self.sd
        for n, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if n < len(self.layers) - 1:
                h = np.tanh(h)
        return h * self.dsd

    @property
    def complexity(self):
        return int(sum(W.size + b.size for W, b in self.layers))

    def title(self):
        sizes = [self.layers[0][0].shape[1]] + [W.shape[0] for W, _ in self.layers]
        return f"{self.label} {'-'.join(map(str, sizes))}"

    def equations(self):
        return [f"réseau {self.title().split()[-1]} ({self.complexity} paramètres), activations tanh puis sortie linéaire",
                ("y = " if self.static else "x(k+1) = x(k) + ") + "DSD .* ( Wn.tanh(...tanh(W1.((z - MU)/SD) + b1)...) + bn )",
                "(formule explicite dans les fichiers exportés ; modèle opaque, à utiliser comme modèle réduit numérique)"]

    def code(self, lang, xs, us):
        py = lang == "py"
        mv, mm = (vec_py, mat_py) if py else (vec_m, mat_m)
        end = "" if py else ";"
        consts = [f"MU = {mv(self.mu)}{end}", f"SD = {mv(self.sd)}{end}", f"DSD = {mv(self.dsd)}{end}"]
        for n, (W, b) in enumerate(self.layers, 1):
            consts += [f"W{n} = {mm(W)}{end}", f"b{n} = {mv(b)}{end}"]
        z = ("np.array([" + ", ".join(xs + us) + "])") if py else ("[" + "; ".join(xs + us) + "]")
        body = [f"h = ({z} - MU) / SD" if py else f"h = ({z} - MU) ./ SD;"]
        for n in range(1, len(self.layers) + 1):
            lin = f"W{n} @ h + b{n}" if py else f"W{n} * h + b{n}"
            last = n == len(self.layers)
            body.append(f"h = {lin}" + end if last else (f"h = np.tanh({lin})" if py else f"h = tanh({lin});"))
        body.append("d[:] = DSD * h" if py else "d = DSD .* h;")
        return consts, body

    def to_dict(self):
        d = self._base()
        d.update(dsd=self.dsd.tolist(), layers=[[W.tolist(), b.tolist()] for W, b in self.layers])
        return d


def fit_neural(data, hidden=(32, 32), epochs=300, lr=3e-3, batch=256, noise=0.02, seed=0, weight_decay=1e-6, should_stop=None) -> NeuralDynModel:
    """Un pas, increment normalise ; le bruit gaussien sur les entrees rend le modele plus stable en simulation libre."""
    Z, D = data.Z(), data.delta()
    mu, sd = Z.mean(0), np.maximum(Z.std(0), 1e-12)
    dsd = np.maximum(D.std(0), 1e-12)
    X, Y = (Z - mu) / sd, D / dsd
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, Y.shape[1]]
    P = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        P += [rng.normal(0, np.sqrt(1.0 / i), (o, i)), np.zeros(o)]
    P[-1] = Y.mean(0).copy()  # sortie initiale = moyenne (utile en regime permanent, ou les sorties ne sont pas centrees)
    m, v = [np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P]
    n_layers, step, n = len(sizes) - 1, 0, len(X)
    for ep in range(epochs):
        if should_stop and should_stop():
            break
        perm = rng.permutation(n)
        rate = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.95 + 0.05)  # decroissance cosinus
        for s in range(0, n, batch):
            idx = perm[s: s + batch]
            xb = X[idx] + noise * rng.normal(size=(len(idx), X.shape[1]))
            acts = [xb]
            for k in range(n_layers):
                z = acts[-1] @ P[2 * k].T + P[2 * k + 1]
                acts.append(np.tanh(z) if k < n_layers - 1 else z)
            g = 2.0 * (acts[-1] - Y[idx]) / len(idx)
            grads = [None] * len(P)
            for k in reversed(range(n_layers)):
                grads[2 * k] = g.T @ acts[k] + weight_decay * P[2 * k]
                grads[2 * k + 1] = g.sum(0)
                if k:
                    g = (g @ P[2 * k]) * (1 - acts[k] ** 2)
            step += 1
            for j, (p, gr) in enumerate(zip(P, grads)):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr ** 2
                p -= rate * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return NeuralDynModel(data.obs_names, data.act_names, data.dt, mu, sd, dsd, [(P[2 * k], P[2 * k + 1]) for k in range(n_layers)],
                          label=f"Réseau de neurones ({'x'.join(map(str, hidden))})", static=bool(getattr(data, "static", False)))


def model_from_dict(d: dict) -> DynModel:
    model = _model_from_dict(d)
    model.x0 = d.get("x0")
    return model


def _model_from_dict(d: dict) -> DynModel:
    common = (d["obs_names"], d["act_names"], d["dt"], d["mu"], d["sd"])
    if d["kind"] == "terms":
        return TermModel(*common, [_from_json(t) for t in d["terms"]], d["C"], d["normalize"], label=d.get("label", ""), static=d.get("static", False))
    if d["kind"] == "neural":
        return NeuralDynModel(*common, d["dsd"], [(W, b) for W, b in d["layers"]], label=d.get("label", ""), static=d.get("static", False))
    raise ValueError(f"type de modèle inconnu : {d['kind']!r}")


__all__ = ["DynModel", "TermModel", "NeuralDynModel", "fit_linear", "fit_polynomial", "fit_sparse", "fit_neural", "model_from_dict", "r_squared"]
