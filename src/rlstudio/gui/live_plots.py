"""Graphique temporel multi-courbes (observations, actions) mis a jour en direct."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLayout, QScrollArea, QVBoxLayout, QWidget

from .theme import C, luminance

LEGEND_MAX_HEIGHT = 100  # au-dela, la legende defile au lieu de manger la place du graphique
REDRAW_MS = 40           # les points recus entre deux dessins sont regroupes : au plus ~25 dessins par seconde
MAX_DRAWN = 1500         # points traces par courbe au maximum : le cout du dessin ne depend plus de la duree de l'episode


def palette(n=6):
    """`n` couleurs distinctes : celles du theme d'abord, puis des teintes reparties regulierement."""
    if "accent" not in C:  # theme pas encore applique
        base = ["#4f7cf0", "#e0a030", "#3ecf8e", "#c050d0", "#e05050", "#8b93a7"]
    else:
        base = [C["accent"], C["warning"], C["success"], C["magenta"], C["danger"], C["muted"]]
    if n <= len(base):
        return base[:n]
    from PySide6.QtGui import QColor

    light = "surface" in C and luminance(C["surface"]) > 0.6
    out = list(base)
    for k in range(n - len(base)):
        hue = (0.07 + 0.618034 * k) % 1.0  # nombre d'or : teintes toujours bien espacees, quel que soit n
        out.append(QColor.fromHsvF(hue, 0.60 if light else 0.55, 0.70 if light else 0.98).name())
    return out


def decimate(t, arr, max_points=MAX_DRAWN):
    """Reduit (t, arr) a ~max_points lignes en gardant les extrema de chaque paquet (aucun pic n'est perdu)."""
    n = len(t)
    if n <= max_points:
        return t, arr
    bucket = -(-n // (max_points // 2))          # lignes par paquet (arrondi superieur)
    full = n // bucket * bucket
    k = full // bucket
    tb = t[:full].reshape(k, bucket)
    ab = arr[:full].reshape(k, bucket, arr.shape[1])
    with np.errstate(all="ignore"):
        lo, hi = np.nanmin(ab, axis=1), np.nanmax(ab, axis=1)
    out_t = np.repeat(tb[:, 0], 2)
    out = np.empty((2 * k, arr.shape[1]))
    out[0::2], out[1::2] = lo, hi
    if full < n:                                  # reste : dernier point exact
        out_t, out = np.append(out_t, t[-1]), np.vstack([out, arr[-1:]])
    return out_t, out


class FlowLayout(QLayout):
    """Dispose les elements en lignes, a la ligne quand la largeur manque (exemple classique de Qt)."""

    def __init__(self, parent=None, hspacing=12, vspacing=3):
        super().__init__(parent)
        self._items, self._h, self._v = [], hspacing, vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        return size

    def _layout(self, rect, test_only):
        x, y, line_h = rect.x(), rect.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            if x + hint.width() > rect.right() + 1 and line_h > 0:
                x, y, line_h = rect.x(), y + line_h + self._v, 0
            if not test_only:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


class _Chip(QWidget):
    """Entree de legende : pastille coloree + nom ; un clic masque / affiche la courbe."""

    toggled = Signal(bool)  # True = visible

    def __init__(self, name, color, tooltip=True):
        super().__init__()
        self.name, self.color, self.on = name, color, True
        self.dot, self.text = QLabel("■"), QLabel(name)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.dot)
        lay.addWidget(self.text)
        self.setCursor(Qt.PointingHandCursor)
        if tooltip:
            self.setToolTip("Cliquer pour masquer / afficher cette courbe")
        self.restyle()

    def restyle(self, color=None):
        self.color = color or self.color
        txt, muted = C.get("text", "#dddddd"), C.get("muted", "#888888")
        self.dot.setStyleSheet(f"color: {self.color if self.on else muted}; font-size: 12pt;")
        self.text.setStyleSheet(f"color: {txt if self.on else muted}; font-size: 9pt;"
                                f"{'' if self.on else 'text-decoration: line-through;'}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.on = not self.on
            self.restyle()
            self.toggled.emit(self.on)


class _Legend(QScrollArea):
    """Legende compacte : entrees cliquables qui passent a la ligne, hauteur plafonnee (defilement au-dela)."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }")
        inner = QWidget()
        self.flow = FlowLayout(inner)
        self.setWidget(inner)
        self.chips = []

    def clear(self):
        while self.flow.count():
            item = self.flow.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.chips = []

    def add(self, chip):
        self.chips.append(chip)
        self.flow.addWidget(chip)

    def fit(self):
        h = self.flow.heightForWidth(max(self.viewport().width(), 1))
        self.setFixedHeight(max(min(h + 2, LEGEND_MAX_HEIGHT), 0) if self.chips else 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()


class SeriesPlot(QWidget):
    """Une courbe par signal ; la legende (noms colores, cliquables) est sous le graphique, jamais dessus.
    `normalized` : [-1, 1]. Une couleur distincte par signal, quel que soit leur nombre."""

    def __init__(self, title, xlabel="pas"):
        super().__init__()
        self._title, self._xlabel = title, xlabel
        self.pw = pg.PlotWidget()
        self.pw.setObjectName("plot")
        self.pw.showGrid(x=True, y=True, alpha=0.12)
        self.pw.setMenuEnabled(False)
        self.pw.setMinimumHeight(180)
        self.legend = _Legend()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.pw, 1)
        lay.addWidget(self.legend)
        self.names, self.curves, self.normalized = [], [], False
        # tampon numpy a croissance geometrique : ajouter un point est O(1), jamais de reconstruction complete
        self._t, self._y, self._n, self._width = np.empty(256), np.full((256, 1), np.nan), 0, 1
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDRAW_MS)
        self._timer.timeout.connect(self._redraw)
        self.hidden = set()  # noms des courbes masquees par l'utilisateur
        self.restyle()

    def set_names(self, names, n=None):
        """Declare les signaux (noms ; `n` pour des noms generiques si la liste est vide)."""
        names = list(names) or [f"{self._title.split()[0].lower()}{i}" for i in range(n or 0)]
        if names == self.names and self.curves:
            return
        self.names = names
        self._rebuild()

    def _pen(self, i, cols):
        return pg.mkPen(cols[i], width=2)

    def _rebuild(self):
        for c in self.curves:
            self.pw.removeItem(c)
        self.curves = []
        cols = palette(len(self.names))
        for i, name in enumerate(self.names):
            curve = self.pw.plot(pen=self._pen(i, cols))
            curve.setVisible(name not in self.hidden)
            self.curves.append(curve)
        self._legend()
        self._redraw()

    @property
    def t(self):
        return self._t[:self._n]

    def reset(self):
        self._n = 0
        self._timer.stop()
        self._redraw()

    def append(self, t, values):
        """Ajoute un point ; le dessin est differe de quelques ms et regroupe les points recus entre-temps."""
        values = np.atleast_1d(np.asarray(values, float))
        if len(values) > len(self.curves):  # signaux non declares : noms generiques
            self.names = self.names + [f"{self._title.split()[0].lower()}{i}" for i in range(len(self.names), len(values))]
            self._rebuild()
        if len(values) > self._width:
            self._y = np.hstack([self._y, np.full((len(self._y), len(values) - self._width), np.nan)])
            self._width = len(values)
        if self._n == len(self._t):
            self._t = np.concatenate([self._t, np.empty(len(self._t))])
            self._y = np.vstack([self._y, np.full((len(self._y), self._width), np.nan)])
        self._t[self._n] = t
        self._y[self._n, :] = np.nan
        self._y[self._n, :len(values)] = values
        self._n += 1
        if not self._timer.isActive():
            self._timer.start()

    def set_normalized(self, flag):
        self.normalized = bool(flag)
        self._redraw()

    def _redraw(self):
        self._timer.stop()
        if not self._n:
            for c in self.curves:
                c.setData([], [])
            return
        arr = self._y[:self._n, :self._width]
        if self.normalized:
            scale = np.nanmax(np.abs(arr), axis=0)
            arr = arr / np.where(scale > 1e-12, scale, 1.0)
        t, arr = decimate(self._t[:self._n], arr)
        for i, c in enumerate(self.curves):
            if i < arr.shape[1]:
                c.setData(t, arr[:, i])
        # plus il y a de points, plus on espace les dessins (le rafraichissement reste fluide sur de longs episodes)
        self._timer.setInterval(min(250, max(REDRAW_MS, self._n // 100)))

    def _set_visible(self, name, visible):
        self.hidden.discard(name) if visible else self.hidden.add(name)
        for n, c in zip(self.names, self.curves):
            if n == name:
                c.setVisible(visible)

    def _set_all_visible(self, visible):
        self.hidden = set() if visible else set(self.names)
        for c in self.curves:
            c.setVisible(visible)
        for chip in self.legend.chips:
            if chip.name in self.names:
                chip.on = visible
                chip.restyle()

    def _legend(self):
        cols = palette(len(self.names))
        self.legend.clear()
        for i, n in enumerate(self.names):
            chip = _Chip(n, cols[i])
            chip.on = n not in self.hidden
            chip.restyle()
            chip.toggled.connect(lambda on, n=n: self._set_visible(n, on))
            self.legend.add(chip)
        if len(self.names) > 3:  # raccourci pour isoler un signal : tout masquer, puis cliquer ceux voulus
            all_chip = _Chip("tout masquer / afficher", C.get("muted", "#888888"), tooltip=False)
            all_chip.dot.setText("◌")
            all_chip.toggled.connect(lambda on: self._set_all_visible(on))
            self.legend.add(all_chip)
        self.legend.fit()

    def restyle(self):
        if "surface" not in C:
            return
        self.pw.setBackground(C["surface"])
        self.pw.setTitle(self._title, color=C["text"], size="11pt")
        self.pw.setLabel("bottom", self._xlabel, color=C["muted"])
        for ax in ("left", "bottom"):
            self.pw.getAxis(ax).setPen(C["border2"])
            self.pw.getAxis(ax).setTextPen(C["muted"])
        cols = palette(len(self.names))
        for i, c in enumerate(self.curves):
            c.setPen(self._pen(i, cols))
        self._legend()
