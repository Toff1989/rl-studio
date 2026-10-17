"""Graphique temporel multi-courbes (observations, actions) mis a jour en direct."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .theme import C


def palette():
    if "accent" not in C:  # theme pas encore applique
        return ["#4f7cf0", "#e0a030", "#3ecf8e", "#c050d0", "#e05050", "#8b93a7"]
    return [C["accent"], C["warning"], C["success"], C["magenta"], C["danger"], C["muted"]]


class SeriesPlot(QWidget):
    """Une courbe par signal ; la legende (noms colores) est sous le graphique, jamais dessus. `normalized` : [-1, 1]."""

    def __init__(self, title, xlabel="pas"):
        super().__init__()
        self._title, self._xlabel = title, xlabel
        self.pw = pg.PlotWidget()
        self.pw.setObjectName("plot")
        self.pw.showGrid(x=True, y=True, alpha=0.12)
        self.pw.setMenuEnabled(False)
        self.pw.setMinimumHeight(180)
        self.legend_label = QLabel()
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.pw, 1)
        lay.addWidget(self.legend_label)
        self.names, self.curves, self.t, self.rows, self.normalized = [], [], [], [], False
        self.restyle()

    def set_names(self, names, n=None):
        """Declare les signaux (noms ; `n` pour des noms generiques si la liste est vide)."""
        names = list(names) or [f"{self._title.split()[0].lower()}{i}" for i in range(n or 0)]
        if names == self.names and self.curves:
            return
        self.names = names
        self._rebuild()

    def _rebuild(self):
        for c in self.curves:
            self.pw.removeItem(c)
        self.curves = []
        cols = palette()
        for i, name in enumerate(self.names):
            self.curves.append(self.pw.plot(pen=pg.mkPen(cols[i % len(cols)], width=2, style=Qt.SolidLine if i < len(cols) else Qt.DashLine)))
        self._legend()
        self._redraw()

    def reset(self):
        self.t, self.rows = [], []
        self._redraw()

    def append(self, t, values):
        values = np.atleast_1d(np.asarray(values, float))
        if len(values) > len(self.curves):  # signaux non declares : noms generiques
            self.names = self.names + [f"{self._title.split()[0].lower()}{i}" for i in range(len(self.names), len(values))]
            self._rebuild()
        self.t.append(t)
        self.rows.append(values)
        self._redraw()

    def set_normalized(self, flag):
        self.normalized = bool(flag)
        self._redraw()

    def _redraw(self):
        if not self.rows:
            for c in self.curves:
                c.setData([], [])
            return
        width = max(len(r) for r in self.rows)
        arr = np.array([np.pad(r, (0, width - len(r)), constant_values=np.nan) for r in self.rows])
        if self.normalized:
            scale = np.nanmax(np.abs(arr), axis=0)
            arr = arr / np.where(scale > 1e-12, scale, 1.0)
        for i, c in enumerate(self.curves):
            if i < arr.shape[1]:
                c.setData(self.t, arr[:, i])

    def _legend(self):
        cols = palette()
        text = (C.get("text", "#dddddd"))
        self.legend_label.setText("&nbsp;&nbsp;&nbsp;".join(
            f'<span style="color:{cols[i % len(cols)]}; font-size:14pt">&#9632;</span> <span style="color:{text}">{n}</span>'
            for i, n in enumerate(self.names)))

    def restyle(self):
        if "surface" not in C:
            return
        self.pw.setBackground(C["surface"])
        self.pw.setTitle(self._title, color=C["text"], size="11pt")
        self.pw.setLabel("bottom", self._xlabel, color=C["muted"])
        for ax in ("left", "bottom"):
            self.pw.getAxis(ax).setPen(C["border2"])
            self.pw.getAxis(ax).setTextPen(C["muted"])
        self._legend()
        cols = palette()
        for i, c in enumerate(self.curves):
            c.setPen(pg.mkPen(cols[i % len(cols)], width=2, style=Qt.SolidLine if i < len(cols) else Qt.DashLine))
