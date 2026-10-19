"""Modeles simples (substituts) d'une politique : lineaire / PID, polynome, table, arbre, symbolique, reseau exact.

Tous sont en NumPy pur. Chacun sait : predire, decrire ses equations, se generer en code Python et MATLAB
(compatible generation de code), et se serialiser en JSON.

Convention : une politique est une fonction  observations (d,) -> actions (m,), bornee par [low, high].
"""
from __future__ import annotations

import itertools
import re

import warnings

import numpy as np

PY_RESERVED = {"u", "obs", "np", "LOW", "HIGH", "policy", "integ", "DT", "if", "else", "for", "while", "def", "class",
               "import", "lambda", "None", "True", "False", "return", "in", "is", "not", "and", "or",
               "PE", "PC", "PB", "PMU", "PSD", "pz", "pf"}
M_RESERVED = {"u", "obs", "LOW", "HIGH", "integ", "DT", "fcn", "policy", "if", "else", "elseif", "end", "for", "while",
              "function", "return", "pi", "i", "j", "eps", "inf", "nan", "sin", "cos", "tanh", "abs", "min", "max",
              "PE", "PC", "PB", "PMU", "PSD", "pz", "pf"}


def clean_names(names, reserved):
    """Identifiants valides (Python et MATLAB), uniques et sans collision avec les noms reserves."""
    out, used = [], set(reserved)
    for n in names:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(n)).strip("_") or "v"
        if s[0].isdigit():
            s = "v_" + s
        while s in used:
            s += "_"
        used.add(s)
        out.append(s)
    return out


def num(v) -> str:
    return f"{float(v):.10g}"


def vec_py(a) -> str:
    return "np.array([" + ", ".join(num(v) for v in np.asarray(a, float).ravel()) + "])"


def vec_m(a) -> str:
    return "[" + "; ".join(num(v) for v in np.asarray(a, float).ravel()) + "]"


def mat_py(a) -> str:
    return "np.array([" + ", ".join("[" + ", ".join(num(v) for v in row) + "]" for row in np.asarray(a, float)) + "])"


def mat_m(a) -> str:
    return "[" + "; ".join(" ".join(num(v) for v in row) for row in np.asarray(a, float)) + "]"


def r_squared(y_true, y_pred) -> float:
    """R2 moyen sur les sorties ; une sortie constante compte pour 1 si elle est reproduite, 0 sinon."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.ndim == 1:
        y_true, y_pred = y_true[:, None], y_pred[:, None]
    sse = ((y_true - y_pred) ** 2).sum(0)
    sst = ((y_true - y_true.mean(0)) ** 2).sum(0)
    scale = np.maximum(np.abs(y_true).mean(0), 1e-9)
    r2 = np.where(sst > 1e-12 * len(y_true) * scale ** 2, 1 - sse / np.maximum(sst, 1e-300),
                  np.where(sse / len(y_true) < 1e-6 * scale ** 2, 1.0, 0.0))
    return float(np.mean(r2))


class Surrogate:
    """Base : un modele simple de la politique."""

    kind = "surrogate"
    stateful = False  # vrai si la sortie depend de l'historique (integrateur du PID)

    def __init__(self, obs_names, act_names, low, high, label=""):
        self.obs_names, self.act_names = [str(n) for n in obs_names], [str(n) for n in act_names]
        self.low, self.high = np.asarray(low, float).ravel(), np.asarray(high, float).ravel()
        self.label = label or self.kind

    # -- evaluation -------------------------------------------------------------------------
    def reset(self):
        pass

    def raw(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return np.clip(self.raw(np.atleast_2d(np.asarray(X, float))), self.low, self.high)

    def act(self, obs) -> np.ndarray:
        return self.predict(np.asarray(obs, float).reshape(1, -1))[0].astype(np.float32)

    def predict_episodes(self, episodes) -> np.ndarray:
        """Predictions sur une liste d'episodes (tableaux d'observations), en respectant l'etat interne."""
        if not self.stateful:
            return self.predict(np.concatenate(episodes))
        outs = []
        for obs in episodes:
            self.reset()
            outs.append(np.array([self.act(o) for o in obs]))
        self.reset()
        return np.concatenate(outs)

    # -- description ------------------------------------------------------------------------
    @property
    def complexity(self) -> int:
        raise NotImplementedError

    def title(self) -> str:
        return self.label

    def equations(self) -> list[str]:
        raise NotImplementedError

    # -- generation de code -----------------------------------------------------------------
    def code(self, lang: str, v: list[str]) -> tuple[list[str], list[str]]:
        """(constantes, corps) : `v` = noms des observations ; le corps remplit u (u[k] en Python, u(k+1) en MATLAB)."""
        raise NotImplementedError

    def _header(self, name):
        return f"{self.title()} - extrait par RL Studio. Entrees : {', '.join(self.obs_names)} ; sorties : {', '.join(self.act_names)}."

    def to_python(self, name="policy") -> str:
        v = clean_names(self.obs_names, PY_RESERVED)
        consts, body = self.code("py", v)
        m = len(self.act_names)
        lines = [f'"""{self._header(name)}"""', "import numpy as np", "", f"LOW = {vec_py(self.low)}",
                 f"HIGH = {vec_py(self.high)}"] + consts + ["", "", f"def {name}(obs):"]
        lines += [f"    {n} = float(obs[{i}])" for i, n in enumerate(v)]
        lines += [f"    u = np.zeros({m})"] + ["    " + ln for ln in body] + ["    return np.clip(u, LOW, HIGH)", ""]
        return "\n".join(lines)

    def to_matlab(self, name="policy") -> str:
        v = clean_names(self.obs_names, M_RESERVED)
        consts, body = self.code("m", v)
        m = len(self.act_names)
        lines = [f"function u = {name}(obs)", f"% {self._header(name)}", "%#codegen"]
        lines += [f"{n} = obs({i + 1});" for i, n in enumerate(v)]
        lines += consts + [f"u = zeros({m}, 1);"] + body + [f"u = min(max(u, {vec_m(self.low)}), {vec_m(self.high)});", "end", ""]
        extra = self.matlab_local_functions()
        return "\n".join(lines + extra)

    def matlab_local_functions(self) -> list[str]:
        return []

    # -- serialisation ----------------------------------------------------------------------
    def _base_dict(self):
        return {"kind": self.kind, "label": self.label, "obs_names": self.obs_names, "act_names": self.act_names,
                "low": self.low.tolist(), "high": self.high.tolist()}

    def to_dict(self) -> dict:
        raise NotImplementedError


def _arr(a):
    return np.asarray(a, float).tolist()


# ================================================================================================
#  Lineaire / PID
# ================================================================================================
class LinearSurrogate(Surrogate):
    """u = b + W.obs [+ Wi.integrale(obs[integ_idx]) pour l'action integrale d'un PID]."""

    kind = "linear"

    def __init__(self, obs_names, act_names, low, high, W, b, integ_idx=(), Wi=None, dt=1.0, label=""):
        super().__init__(obs_names, act_names, low, high, label or ("PID (retour d'etat + integrale)" if len(integ_idx) else "Retour d'etat lineaire"))
        self.W, self.b = np.asarray(W, float), np.asarray(b, float).ravel()
        self.integ_idx = [int(i) for i in integ_idx]
        self.Wi = np.zeros((len(self.b), 0)) if Wi is None else np.asarray(Wi, float).reshape(len(self.b), -1)
        self.dt = float(dt)
        self.stateful = bool(self.integ_idx)
        self._I = np.zeros(len(self.integ_idx))

    def reset(self):
        self._I = np.zeros(len(self.integ_idx))

    def raw(self, X):
        return X @ self.W.T + self.b

    def act(self, obs):
        obs = np.asarray(obs, float).ravel()
        if not self.stateful:
            return super().act(obs)
        self._I = self._I + obs[self.integ_idx] * self.dt
        u = self.W @ obs + self.Wi @ self._I + self.b
        return np.clip(u, self.low, self.high).astype(np.float32)

    @property
    def complexity(self):
        return int(self.W.size + self.Wi.size + self.b.size)

    def equations(self):
        eqs = []
        for k, a in enumerate(self.act_names):
            terms = [f"{self.W[k, i]:+.4g}*{n}" for i, n in enumerate(self.obs_names) if abs(self.W[k, i]) > 1e-12]
            terms += [f"{self.Wi[k, j]:+.4g}*int({self.obs_names[i]})dt" for j, i in enumerate(self.integ_idx)]
            eqs.append(f"{a} = {self.b[k]:.4g} " + " ".join(terms) + f"   (borne a [{self.low[k]:.4g}, {self.high[k]:.4g}])")
        return eqs

    def code(self, lang, v):
        py = lang == "py"
        body, m = [], len(self.act_names)
        for k in range(m):
            expr = " + ".join([num(self.b[k])] + [f"{num(self.W[k, i])}*{n}" for i, n in enumerate(v) if abs(self.W[k, i]) > 1e-12]
                              + [f"{num(self.Wi[k, j])}*integ{'[' + str(j) + ']' if py else '(' + str(j + 1) + ')'}" for j in range(len(self.integ_idx))])
            body.append((f"u[{k}] = " if py else f"u({k + 1}) = ") + expr + ("" if py else ";"))
        return [], body

    def _stateful_python(self, name):
        v = clean_names(self.obs_names, PY_RESERVED)
        _, body = self.code("py", v)
        k = len(self.integ_idx)
        lines = [f'"""{self._header(name)}"""', "import numpy as np", "", f"LOW = {vec_py(self.low)}", f"HIGH = {vec_py(self.high)}",
                 f"DT = {num(self.dt)}   # pas de temps de l'integrale (secondes ou pas de decision)", "", "",
                 "class Policy:", f"    \"\"\"Politique a etat (integrale) : appeler reset() au debut de chaque episode.\"\"\"", "",
                 "    def __init__(self):", "        self.reset()", "", "    def reset(self):", f"        self.integ = np.zeros({k})", "",
                 "    def __call__(self, obs):"]
        lines += [f"        {n} = float(obs[{i}])" for i, n in enumerate(v)]
        upd = ", ".join(v[i] for i in self.integ_idx)
        lines += [f"        self.integ = self.integ + np.array([{upd}]) * DT", "        integ = self.integ", f"        u = np.zeros({len(self.act_names)})"]
        lines += ["        " + ln for ln in body] + ["        return np.clip(u, LOW, HIGH)", "", "", f"{name} = Policy()", ""]
        return "\n".join(lines)

    def to_python(self, name="policy"):
        return self._stateful_python(name) if self.stateful else super().to_python(name)

    def to_matlab(self, name="policy"):
        if not self.stateful:
            return super().to_matlab(name)
        v = clean_names(self.obs_names, M_RESERVED)
        _, body = self.code("m", v)
        k, m = len(self.integ_idx), len(self.act_names)
        upd = "; ".join(v[i] for i in self.integ_idx)
        lines = [f"function u = {name}(obs)", f"% {self._header(name)}", "% Politique a etat (integrale) : `clear " + name + "` la reinitialise.", "%#codegen",
                 "persistent integ;", f"DT = {num(self.dt)};", f"if isempty(integ), integ = zeros({k}, 1); end"]
        lines += [f"{n} = obs({i + 1});" for i, n in enumerate(v)]
        lines += [f"integ = integ + [{upd}] * DT;", f"u = zeros({m}, 1);"] + body
        lines += [f"u = min(max(u, {vec_m(self.low)}), {vec_m(self.high)});", "end", ""]
        return "\n".join(lines)

    def to_dict(self):
        d = self._base_dict()
        d.update(W=_arr(self.W), b=_arr(self.b), integ_idx=self.integ_idx, Wi=_arr(self.Wi), dt=self.dt)
        return d


def fit_linear(data, ridge=1e-8, integrators=(), dt=1.0, label="") -> LinearSurrogate:
    """Moindres carres (ridge leger) ; `integrators` : noms d'observations a integrer (action integrale)."""
    names = data.obs_names
    idx = [names.index(n) for n in integrators]
    feats = []
    for obs in data.obs_episodes():
        cols = [obs]
        if idx:
            cols.append(np.cumsum(obs[:, idx], axis=0) * dt)
        feats.append(np.hstack(cols))
    F, Y = np.concatenate(feats), data.actions()
    A = np.hstack([F, np.ones((len(F), 1))])
    reg = ridge * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(A.T @ A + reg * len(A), A.T @ Y)  # (d + k + 1, m)
    d = len(names)
    return LinearSurrogate(names, data.act_names, data.low, data.high, W=coef[:d].T, b=coef[-1], integ_idx=idx,
                           Wi=coef[d:-1].T if idx else None, dt=dt, label=label)


# ================================================================================================
#  Polynomial : u = b + somme des monomes (produits des observations normalisees) jusqu'au degre D
# ================================================================================================
def _exponents(d, degree):
    """Exposants de tous les monomes de degre 1..degree (hors constante) en d variables, ordre croissant de degre."""
    rows = []
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            e = np.zeros(d, int)
            for i in combo:
                e[i] += 1
            rows.append(e)
    return np.array(rows, int).reshape(-1, d)


def _monomials(Z, E):
    """Matrice (n, nombre de monomes) des produits prod_i Z[:, i] ** E[t, i]."""
    n = len(Z)
    F = np.ones((n, len(E)))
    top = int(E.max()) if E.size else 0
    for i in range(E.shape[1]):
        powers = Z[:, i, None] ** np.arange(top + 1)  # (n, top + 1)
        F *= powers[:, E[:, i]]
    return F


class PolynomialSurrogate(Surrogate):
    """u = b + C . monomes(z), z = (obs - mu) / sd : polynome multivarie (termes croises inclus) de degre `degree`."""

    kind = "polynomial"

    def __init__(self, obs_names, act_names, low, high, degree, mu, sd, E, C, b, label=""):
        super().__init__(obs_names, act_names, low, high, label or f"Polynome de degre {int(degree)}")
        self.degree = int(degree)
        self.mu, self.sd = np.asarray(mu, float).ravel(), np.asarray(sd, float).ravel()
        self.E = np.asarray(E, int).reshape(-1, len(self.mu))
        self.C = np.asarray(C, float).reshape(len(np.atleast_1d(b)), -1)
        self.b = np.asarray(b, float).ravel()

    @property
    def complexity(self):
        return int(self.C.size + self.b.size)

    def raw(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return _monomials(Z, self.E) @ self.C.T + self.b

    def _term(self, t, names, lang):
        py = lang == "py"
        parts = []
        for i, e in enumerate(self.E[t]):
            if e == 1:
                parts.append(names[i])
            elif e > 1:
                parts.append(f"{names[i]}**{e}" if py else f"{names[i]}^{e}")
        return "*".join(parts)

    def equations(self):
        zn = [f"z_{n}" for n in self.obs_names]
        head = [f"z_{n} = ({n} - {m:.4g}) / {s:.4g}" for n, m, s in zip(self.obs_names, self.mu, self.sd)]
        eqs = []
        for k, a in enumerate(self.act_names):
            order = [t for t in np.argsort(-np.abs(self.C[k])) if abs(self.C[k, t]) > 1e-12][:40]
            terms = [f"{self.C[k, t]:+.4g}*{self._term(t, zn, 'py')}" for t in sorted(order)]
            more = int(np.count_nonzero(np.abs(self.C[k]) > 1e-12)) - len(order)
            eqs.append(f"{a} = {self.b[k]:.4g} " + " ".join(terms) + (f" ... (+{more} termes)" if more > 0 else "")
                       + f"   (borne a [{self.low[k]:.4g}, {self.high[k]:.4g}])")
        return head + eqs

    def title(self):
        return f"{self.label} ({self.E.shape[0]} monomes)"

    def code(self, lang, v):
        py = lang == "py"
        if py:
            consts = [f"PMU = {vec_py(self.mu)}", f"PSD = {vec_py(self.sd)}", f"PE = {mat_py(self.E)}", f"PC = {mat_py(self.C)}", f"PB = {vec_py(self.b)}"]
            body = [f"pz = (np.array([{', '.join(v)}]) - PMU) / PSD", "pf = np.prod(pz ** PE, axis=1)", "u[:] = PC @ pf + PB"]
        else:
            consts = [f"PMU = {vec_m(self.mu)};", f"PSD = {vec_m(self.sd)};", f"PE = {mat_m(self.E)};", f"PC = {mat_m(self.C)};", f"PB = {vec_m(self.b)};"]
            body = [f"pz = ([{'; '.join(v)}] - PMU) ./ PSD;", f"pf = prod(repmat(pz', {self.E.shape[0]}, 1) .^ PE, 2);", "u = PC * pf + PB;"]
        return consts, body

    def to_dict(self):
        d = self._base_dict()
        d.update(degree=self.degree, mu=_arr(self.mu), sd=_arr(self.sd), E=self.E.tolist(), C=_arr(self.C), b=_arr(self.b))
        return d


def fit_polynomial(data, degree=2, ridge=1e-6, max_terms=400, label="") -> PolynomialSurrogate:
    """Moindres carres regularises sur les monomes des observations normalisees. Si le nombre de monomes depasse
    `max_terms`, le degre est abaisse (jamais sous 1) pour garder un modele exportable."""
    X, Y = data.observations(), data.actions()
    d = X.shape[1]
    degree = max(1, int(degree))
    while degree > 1 and len(_exponents(d, degree)) > max_terms:
        degree -= 1
    mu = X.mean(0)
    sd = np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    E = _exponents(d, degree)
    F = _monomials((X - mu) / sd, E)
    fm, ym = F.mean(0), Y.mean(0)
    Fc, Yc = F - fm, Y - ym
    A = Fc.T @ Fc / len(F) + ridge * np.eye(F.shape[1])
    coef = np.linalg.solve(A, Fc.T @ Yc / len(F))  # (nombre de monomes, m)
    b = ym - fm @ coef
    return PolynomialSurrogate(data.obs_names, data.act_names, data.low, data.high, degree, mu, sd, E, coef.T, b, label=label)


# ================================================================================================
#  Arbre de decision (CART, sorties multiples, feuilles constantes)
# ================================================================================================
class TreeSurrogate(Surrogate):
    kind = "tree"

    def __init__(self, obs_names, act_names, low, high, feat, thr, left, right, value, label=""):
        super().__init__(obs_names, act_names, low, high, label or "Arbre de decision")
        self.feat, self.thr = np.asarray(feat, int), np.asarray(thr, float)
        self.left, self.right = np.asarray(left, int), np.asarray(right, int)
        self.value = np.asarray(value, float).reshape(len(self.feat), -1)

    @property
    def n_leaves(self):
        return int((self.feat < 0).sum())

    @property
    def depth(self):
        def d(n):
            return 0 if self.feat[n] < 0 else 1 + max(d(self.left[n]), d(self.right[n]))
        return d(0)

    @property
    def complexity(self):
        return self.n_leaves * len(self.act_names)

    def title(self):
        return f"{self.label} (profondeur {self.depth}, {self.n_leaves} feuilles)"

    def raw(self, X):
        idx = np.zeros(len(X), int)
        rows = np.arange(len(X))
        for _ in range(self.depth + 1):
            f = self.feat[idx]
            active = f >= 0
            if not active.any():
                break
            go_left = X[rows, np.where(active, f, 0)] <= self.thr[idx]
            idx = np.where(active, np.where(go_left, self.left[idx], self.right[idx]), idx)
        return self.value[idx]

    def equations(self):
        lines = []

        def walk(n, ind):
            if self.feat[n] < 0:
                lines.append("  " * ind + "alors " + ", ".join(f"{a} = {x:.4g}" for a, x in zip(self.act_names, self.value[n])))
                return
            name = self.obs_names[self.feat[n]]
            lines.append("  " * ind + f"si {name} <= {self.thr[n]:.4g} :")
            walk(self.left[n], ind + 1)
            lines.append("  " * ind + "sinon :")
            walk(self.right[n], ind + 1)
        walk(0, 0)
        return lines

    def code(self, lang, v):
        py, body = lang == "py", []

        def walk(n, ind):
            pad = ("    " * ind) if py else ("  " * ind)
            if self.feat[n] < 0:
                for k, x in enumerate(self.value[n]):
                    body.append(pad + (f"u[{k}] = {num(x)}" if py else f"u({k + 1}) = {num(x)};"))
                return
            cond = f"{v[self.feat[n]]} <= {num(self.thr[n])}"
            body.append(pad + (f"if {cond}:" if py else f"if {cond}"))
            walk(self.left[n], ind + 1)
            body.append(pad + ("else:" if py else "else"))
            walk(self.right[n], ind + 1)
            if not py:
                body.append(pad + "end")
        walk(0, 0)
        return [], body

    def to_dict(self):
        d = self._base_dict()
        d.update(feat=self.feat.tolist(), thr=self.thr.tolist(), left=self.left.tolist(), right=self.right.tolist(), value=_arr(self.value))
        return d


def _best_split(X, Y, min_leaf):
    n = len(X)
    total = Y.sum(0)
    best = None
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable")
        xs, cs = X[order, f], np.cumsum(Y[order], axis=0)
        nl = np.arange(1, n)[:, None]
        sl = cs[:-1]
        score = (sl ** 2 / nl + (total - sl) ** 2 / (n - nl)).sum(1)
        valid = (xs[:-1] != xs[1:]) & (nl[:, 0] >= min_leaf) & (n - nl[:, 0] >= min_leaf)
        if not valid.any():
            continue
        score = np.where(valid, score, -np.inf)
        i = int(np.argmax(score))
        if best is None or score[i] > best[0]:
            best = (float(score[i]), f, float((xs[i] + xs[i + 1]) / 2))
    if best is None:
        return None
    gain = best[0] - float((total ** 2).sum() / n)
    return (best[1], best[2], gain)


def fit_tree(data, max_depth=5, min_leaf=20, label="") -> TreeSurrogate:
    X, Y = data.observations(), data.actions()
    feat, thr, left, right, value = [], [], [], [], []
    scale = float(((Y - Y.mean(0)) ** 2).sum()) or 1.0

    def build(ids, depth):
        node = len(feat)
        feat.append(-1), thr.append(0.0), left.append(-1), right.append(-1), value.append(Y[ids].mean(0))
        if depth >= max_depth or len(ids) < 2 * min_leaf:
            return node
        split = _best_split(X[ids], Y[ids], min_leaf)
        if split is None or split[2] < 1e-7 * scale:
            return node
        f, t, _ = split
        mask = X[ids, f] <= t
        feat[node], thr[node] = f, t
        left[node] = build(ids[mask], depth + 1)
        right[node] = build(ids[~mask], depth + 1)
        return node

    build(np.arange(len(X)), 0)
    return TreeSurrogate(data.obs_names, data.act_names, data.low, data.high, feat, thr, left, right, value, label=label)


# ================================================================================================
#  Table (1 ou 2 entrees, interpolation lineaire)
# ================================================================================================
class TableSurrogate(Surrogate):
    kind = "table"

    def __init__(self, obs_names, act_names, low, high, dims, grids, table, label=""):
        super().__init__(obs_names, act_names, low, high, label or "Table de gains")
        self.dims = [int(i) for i in dims]
        self.grids = [np.asarray(g, float) for g in grids]
        m = len(self.act_names)
        self.table = np.asarray(table, float).reshape(*[len(g) for g in self.grids], m)

    @property
    def complexity(self):
        return int(self.table.size)

    def title(self):
        return f"{self.label} sur {' x '.join(self.obs_names[i] for i in self.dims)}"

    def raw(self, X):
        if len(self.dims) == 1:
            g, x = self.grids[0], X[:, self.dims[0]]
            return np.stack([np.interp(x, g, self.table[:, k]) for k in range(self.table.shape[-1])], axis=1)
        gx, gy = self.grids
        x = np.clip(X[:, self.dims[0]], gx[0], gx[-1])
        y = np.clip(X[:, self.dims[1]], gy[0], gy[-1])
        i = np.minimum(np.searchsorted(gx, x, side="right") - 1, len(gx) - 2)
        j = np.minimum(np.searchsorted(gy, y, side="right") - 1, len(gy) - 2)
        fx = ((x - gx[i]) / (gx[i + 1] - gx[i]))[:, None]
        fy = ((y - gy[j]) / (gy[j + 1] - gy[j]))[:, None]
        t = self.table
        return (t[i, j] * (1 - fx) * (1 - fy) + t[i + 1, j] * fx * (1 - fy) + t[i, j + 1] * (1 - fx) * fy + t[i + 1, j + 1] * fx * fy)

    def equations(self):
        names = [self.obs_names[i] for i in self.dims]
        out = [f"actions = interpolation lineaire d'une table sur ({', '.join(names)}) ; les autres entrees sont ignorees"]
        for n, g in zip(names, self.grids):
            out.append(f"  {n} : {len(g)} points de {g[0]:.4g} a {g[-1]:.4g}")
        if len(self.dims) == 1:
            out += [f"  {names[0]} = {x:.4g} -> " + ", ".join(f"{a} = {self.table[i, k]:.4g}" for k, a in enumerate(self.act_names))
                    for i, x in enumerate(self.grids[0])][:: max(1, len(self.grids[0]) // 8)]
        return out

    def code(self, lang, v):
        py, m = lang == "py", len(self.act_names)
        names = [v[i] for i in self.dims]
        if len(self.dims) == 1:
            gx = vec_py(self.grids[0]) if py else vec_m(self.grids[0])
            consts = [f"GX = {gx}" if py else f"GX = {gx};"]
            body = []
            for k in range(m):
                col = self.table[:, k]
                consts.append(f"T{k} = {vec_py(col)}" if py else f"T{k} = {vec_m(col)};")
                body.append(f"u[{k}] = np.interp({names[0]}, GX, T{k})" if py else
                            f"u({k + 1}) = interp1(GX, T{k}, min(max({names[0]}, GX(1)), GX(end)));")
            return consts, body
        consts = [f"GX = {vec_py(self.grids[0])}" if py else f"GX = {vec_m(self.grids[0])};",
                  f"GY = {vec_py(self.grids[1])}" if py else f"GY = {vec_m(self.grids[1])};"]
        body = []
        for k in range(m):
            t = self.table[:, :, k]
            consts.append(f"T{k} = {mat_py(t)}" if py else f"T{k} = {mat_m(t)};")
            body.append(f"u[{k}] = _lerp2(GX, GY, T{k}, {names[0]}, {names[1]})" if py else f"u({k + 1}) = lerp2(GX, GY, T{k}, {names[0]}, {names[1]});")
        return consts, body

    def to_python(self, name="policy"):
        text = super().to_python(name)
        if len(self.dims) == 2:
            helper = ("def _lerp2(gx, gy, t, x, y):\n    x = min(max(x, gx[0]), gx[-1])\n    y = min(max(y, gy[0]), gy[-1])\n"
                      "    i = min(int(np.searchsorted(gx, x, side=\"right\")) - 1, len(gx) - 2)\n"
                      "    j = min(int(np.searchsorted(gy, y, side=\"right\")) - 1, len(gy) - 2)\n"
                      "    fx = (x - gx[i]) / (gx[i + 1] - gx[i])\n    fy = (y - gy[j]) / (gy[j + 1] - gy[j])\n"
                      "    return (t[i, j] * (1 - fx) * (1 - fy) + t[i + 1, j] * fx * (1 - fy)\n"
                      "            + t[i, j + 1] * (1 - fx) * fy + t[i + 1, j + 1] * fx * fy)\n\n\n")
            text = text.replace(f"def {name}(obs):", helper + f"def {name}(obs):", 1)
        return text

    def matlab_local_functions(self):
        if len(self.dims) != 2:
            return []
        return ["", "function v = lerp2(gx, gy, t, x, y)", "x = min(max(x, gx(1)), gx(end));", "y = min(max(y, gy(1)), gy(end));",
                "i = min(find(gx <= x, 1, 'last'), numel(gx) - 1);", "j = min(find(gy <= y, 1, 'last'), numel(gy) - 1);",
                "fx = (x - gx(i)) / (gx(i+1) - gx(i));", "fy = (y - gy(j)) / (gy(j+1) - gy(j));",
                "v = t(i,j)*(1-fx)*(1-fy) + t(i+1,j)*fx*(1-fy) + t(i,j+1)*(1-fx)*fy + t(i+1,j+1)*fx*fy;", "end", ""]

    def to_dict(self):
        d = self._base_dict()
        d.update(dims=self.dims, grids=[g.tolist() for g in self.grids], table=self.table.tolist())
        return d


def fit_table(data, k_dims=2, points=None, label="") -> TableSurrogate:
    """Table sur les k entrees les plus influentes (coefficients lineaires standardises) ; moyenne conditionnelle par cellule."""
    X, Y = data.observations(), data.actions()
    d = X.shape[1]
    k = max(1, min(k_dims, d))
    std = X.std(0) + 1e-12
    lin = fit_linear(data)
    importance = (np.abs(lin.W) * std).sum(0)
    dims = [int(i) for i in np.argsort(-importance)[:k]]
    points = points or (33 if k == 1 else 15)
    grids = []
    for i in dims:
        lo, hi = np.percentile(X[:, i], [0.5, 99.5])
        if hi - lo < 1e-9:
            hi = lo + 1.0
        grids.append(np.linspace(lo, hi, points))
    m = Y.shape[1]
    shape = [points] * k
    sums, counts = np.zeros(shape + [m]), np.zeros(shape)
    cell = tuple(np.clip(np.rint((X[:, i] - g[0]) / (g[1] - g[0])).astype(int), 0, points - 1) for i, g in zip(dims, grids))
    np.add.at(sums, cell, Y)
    np.add.at(counts, cell, 1)
    table = np.where(counts[..., None] > 0, sums / np.maximum(counts[..., None], 1), np.nan)
    # cellules vides : moyenne des voisins deja remplis, par propagation, puis moyenne globale en dernier recours
    for _ in range(points * 2):
        holes = np.isnan(table[..., 0])
        if not holes.any():
            break
        pad = np.pad(table, [(1, 1)] * k + [(0, 0)], constant_values=np.nan)
        neigh = []
        for ax in range(k):
            for s in (-1, 1):
                sl = [slice(1, -1)] * k
                sl[ax] = slice(1 + s, points + 1 + s)
                neigh.append(pad[tuple(sl)])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmean(np.stack(neigh), axis=0)
        table = np.where(holes[..., None], fill, table)
    table = np.where(np.isnan(table), Y.mean(0), table)
    return TableSurrogate(data.obs_names, data.act_names, data.low, data.high, dims, grids, table, label=label)


# ================================================================================================
#  Symbolique : regression parcimonieuse sur une bibliotheque de termes (OMP)
# ================================================================================================
def _term_value(kind, i, j, X):
    x = X[:, i]
    return {"x": lambda: x, "x2": lambda: x ** 2, "x3": lambda: x ** 3, "sin": lambda: np.sin(x), "cos": lambda: np.cos(x),
            "tanh": lambda: np.tanh(x), "abs": lambda: np.abs(x), "prod": lambda: x * X[:, j]}[kind]()


def _term_text(kind, i, j, names, lang):
    a = names[i]
    py = lang == "py"
    if kind == "x":
        return a
    if kind == "x2":
        return f"{a}**2" if py else f"{a}^2"
    if kind == "x3":
        return f"{a}**3" if py else f"{a}^3"
    if kind == "prod":
        return f"{a}*{names[j]}"
    fn = {"sin": ("np.sin", "sin"), "cos": ("np.cos", "cos"), "tanh": ("np.tanh", "tanh"), "abs": ("abs", "abs")}[kind]
    return f"{fn[0] if py else fn[1]}({a})"


class SymbolicSurrogate(Surrogate):
    kind = "symbolic"

    def __init__(self, obs_names, act_names, low, high, models, label=""):
        """models : par action, {"intercept": b, "terms": [(coef, kind, i, j), ...]}"""
        super().__init__(obs_names, act_names, low, high, label or "Equation symbolique")
        self.models = [{"intercept": float(m["intercept"]), "terms": [(float(c), str(k), int(i), int(j)) for c, k, i, j in m["terms"]]} for m in models]

    @property
    def complexity(self):
        return int(sum(len(m["terms"]) + 1 for m in self.models))

    def raw(self, X):
        out = np.zeros((len(X), len(self.models)))
        for k, m in enumerate(self.models):
            out[:, k] = m["intercept"] + sum((c * _term_value(kind, i, j, X) for c, kind, i, j in m["terms"]), np.zeros(len(X)))
        return out

    def _expr(self, k, names, lang, digits=None):
        m = self.models[k]
        fmt = (lambda c: f"{c:.4g}") if digits else num
        parts = [fmt(m["intercept"])] + [f"{fmt(c)}*{_term_text(kind, i, j, names, lang)}" for c, kind, i, j in m["terms"]]
        return " + ".join(parts).replace("+ -", "- ")

    def equations(self):
        return [f"{a} = {self._expr(k, self.obs_names, 'py', digits=True)}   (borne a [{self.low[k]:.4g}, {self.high[k]:.4g}])" for k, a in enumerate(self.act_names)]

    def code(self, lang, v):
        py = lang == "py"
        return [], [(f"u[{k}] = " if py else f"u({k + 1}) = ") + self._expr(k, v, lang) + ("" if py else ";") for k in range(len(self.models))]

    def to_dict(self):
        d = self._base_dict()
        d.update(models=[{"intercept": m["intercept"], "terms": [list(t) for t in m["terms"]]} for m in self.models])
        return d


def _library(d, names=None):
    """Bibliotheque de termes candidats : (kind, i, j)."""
    terms = []
    for i in range(d):
        terms += [("x", i, -1), ("x2", i, -1), ("x3", i, -1), ("sin", i, -1), ("cos", i, -1), ("tanh", i, -1), ("abs", i, -1)]
    terms += [("prod", i, j) for i in range(d) for j in range(i + 1, d)]
    return terms


def fit_symbolic(data, term_counts=(3, 6, 12), label="") -> dict[int, SymbolicSurrogate]:
    """Selection gloutonne de termes (OMP) : renvoie un modele par nombre de termes demande (par action)."""
    X, Y = data.observations(), data.actions()
    lib = _library(X.shape[1])
    F = np.column_stack([_term_value(k, i, j, X) for k, i, j in lib])
    mu, sd = F.mean(0), F.std(0)
    keep = sd > 1e-9
    Fs = (F - mu) / np.where(keep, sd, 1.0)
    Fs[:, ~keep] = 0.0
    kmax = max(term_counts)
    per_action = []  # [action][taille] -> modele
    for a in range(Y.shape[1]):
        y = Y[:, a]
        yc = y - y.mean()
        sel, residual, history = [], yc.copy(), {}
        for step in range(1, kmax + 1):
            corr = np.abs(Fs.T @ residual)
            corr[sel] = -1
            j = int(np.argmax(corr))
            if corr[j] <= 1e-12:
                break
            sel.append(j)
            coef, *_ = np.linalg.lstsq(Fs[:, sel], yc, rcond=None)
            residual = yc - Fs[:, sel] @ coef
            c_orig = coef / sd[sel]
            history[step] = {"intercept": float(y.mean() - (c_orig * mu[sel]).sum()),
                             "terms": [(float(c), *lib[s]) for c, s in zip(c_orig, sel)]}
        per_action.append(history)
    out = {}
    for k in term_counts:
        models = []
        for history in per_action:
            size = max((s for s in history if s <= k), default=None)
            models.append(history[size] if size else {"intercept": 0.0, "terms": []})
        out[k] = SymbolicSurrogate(data.obs_names, data.act_names, data.low, data.high, models, label=(label or "Equation symbolique") + f" ({k} termes max)")
    return out


# ================================================================================================
#  Reseau de neurones exact (poids extraits de Stable-Baselines3)
# ================================================================================================
class NeuralSurrogate(Surrogate):
    """Le reseau de la politique tel quel, en NumPy : exact, mais opaque."""

    kind = "neural"

    def __init__(self, obs_names, act_names, low, high, layers, squash, label=""):
        """layers : [(W (out,in), b (out,), activation 'relu'|'tanh'|'identity')] ; squash : sortie = low + 0.5*(tanh(z)+1)*(high-low)"""
        super().__init__(obs_names, act_names, low, high, label or "Reseau de neurones (exact)")
        self.layers = [(np.asarray(W, float), np.asarray(b, float), str(act)) for W, b, act in layers]
        self.squash = bool(squash)

    @property
    def complexity(self):
        return int(sum(W.size + b.size for W, b, _ in self.layers))

    def raw(self, X):
        h = np.asarray(X, float)
        for W, b, act in self.layers:
            h = h @ W.T + b
            h = np.maximum(h, 0) if act == "relu" else np.tanh(h) if act == "tanh" else h
        return self.low + 0.5 * (np.tanh(h) + 1.0) * (self.high - self.low) if self.squash else h

    def title(self):
        sizes = [self.layers[0][0].shape[1]] + [W.shape[0] for W, _, _ in self.layers]
        return f"{self.label} {'-'.join(map(str, sizes))}"

    def equations(self):
        sizes = [self.layers[0][0].shape[1]] + [W.shape[0] for W, _, _ in self.layers]
        acts = [a for _, _, a in self.layers]
        return [f"reseau {' -> '.join(map(str, sizes))} ({self.complexity} parametres), activations : {', '.join(acts)}",
                ("sortie = low + 0.5*(tanh(z)+1)*(high-low)" if self.squash else "sortie = z, bornee a [low, high]"),
                "(formule explicite dans les fichiers exportes : politique opaque, a distiller pour comprendre)"]

    def code(self, lang, v):
        py = lang == "py"
        consts, body = [], []
        for n, (W, b, act) in enumerate(self.layers, 1):
            consts += [f"W{n} = {mat_py(W)}", f"b{n} = {vec_py(b)}"] if py else [f"W{n} = {mat_m(W)};", f"b{n} = {vec_m(b)};"]
        h0 = "np.array([" + ", ".join(v) + "])" if py else "[" + "; ".join(v) + "]"
        body.append(f"h = {h0}" if py else f"h = {h0};")
        for n, (W, b, act) in enumerate(self.layers, 1):
            z = f"W{n} @ h + b{n}" if py else f"W{n} * h + b{n}"
            if act == "relu":
                body.append(f"h = np.maximum({z}, 0)" if py else f"h = max({z}, 0);")
            elif act == "tanh":
                body.append(f"h = np.tanh({z})" if py else f"h = tanh({z});")
            else:
                body.append(f"h = {z}" if py else f"h = {z};")
        m = len(self.act_names)
        if self.squash:
            low, high = (vec_py(self.low), vec_py(self.high)) if py else (vec_m(self.low), vec_m(self.high))
            expr = f"{low} + 0.5 * (np.tanh(h) + 1.0) * ({high} - {low})" if py else f"{low} + 0.5 * (tanh(h) + 1.0) .* ({high} - {low})"
        else:
            expr = "h"
        body.append(("u[:] = " if py else "u = ") + expr + ("" if py else ";"))
        return consts, body

    def to_dict(self):
        d = self._base_dict()
        d.update(layers=[[W.tolist(), b.tolist(), a] for W, b, a in self.layers], squash=self.squash)
        return d


# ================================================================================================
#  (de)serialisation
# ================================================================================================
def surrogate_from_dict(d: dict) -> Surrogate:
    common = (d["obs_names"], d["act_names"], d["low"], d["high"])
    kind, label = d["kind"], d.get("label", "")
    if kind == "linear":
        return LinearSurrogate(*common, W=d["W"], b=d["b"], integ_idx=d.get("integ_idx", ()), Wi=d.get("Wi"), dt=d.get("dt", 1.0), label=label)
    if kind == "polynomial":
        return PolynomialSurrogate(*common, d["degree"], d["mu"], d["sd"], d["E"], d["C"], d["b"], label=label)
    if kind == "tree":
        return TreeSurrogate(*common, d["feat"], d["thr"], d["left"], d["right"], d["value"], label=label)
    if kind == "table":
        return TableSurrogate(*common, d["dims"], d["grids"], d["table"], label=label)
    if kind == "symbolic":
        return SymbolicSurrogate(*common, d["models"], label=label)
    if kind == "neural":
        return NeuralSurrogate(*common, [(W, b, a) for W, b, a in d["layers"]], d["squash"], label=label)
    raise ValueError(f"type de modele inconnu : {kind!r}")
