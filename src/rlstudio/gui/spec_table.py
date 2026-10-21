"""Table editable <-> liste de dicts (actions, observations, etat initial)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHeaderView, QTableWidget, QTableWidgetItem,
)


# colonnes des tables Simulink : (cle, titre, type)
ACTION_COLS = [("name", "Nom", str), ("block", "Bloc", str), ("param", "Paramètre", str),
               ("variable", "ou Variable", str), ("low", "Min", float), ("high", "Max", float)]
OBS_COLS = [("name", "Nom", str), ("block", "Bloc (sortie)", str), ("port", "Port", int),
            ("size", "Taille", int), ("low", "Min", float), ("high", "Max", float)]
INIT_COLS = [("variable", "Variable", str), ("low", "Min", float), ("high", "Max", float)]


class SpecTable(QTableWidget):
    """Table editable <-> liste de dicts."""

    changed = Signal()

    def __init__(self, cols, defaults=None):
        super().__init__(0, len(cols))
        self.cols, self.defaults = cols, defaults or {}
        self.setHorizontalHeaderLabels([c[1] for c in cols])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(32)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setShowGrid(False)
        self.setMinimumHeight(130)
        self.setMaximumHeight(210)
        widths = {"name": 120, "block": 210, "param": 110, "variable": 110, "port": 60, "size": 60,
                  "low": 80, "high": 80}
        for j, (key, _, _) in enumerate(cols[:-1]):
            self.setColumnWidth(j, widths.get(key, 100))
        self._loading = False
        self.cellChanged.connect(lambda *_: None if self._loading else self.changed.emit())

    def set_rows(self, rows):
        self._loading = True
        self.setRowCount(0)
        for r in rows or []:
            self.add_row(r, emit=False)
        self._loading = False

    def add_row(self, values=None, emit=True):
        values = {**self.defaults, **(values or {})}
        was = self._loading
        self._loading = True
        row = self.rowCount()
        self.insertRow(row)
        for j, (key, _, _) in enumerate(self.cols):
            self.setItem(row, j, QTableWidgetItem("" if values.get(key) is None else str(values[key])))
        self._loading = was
        if emit:
            self.changed.emit()

    def remove_selected(self):
        for idx in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(idx)
        self.changed.emit()

    def rows(self):
        out = []
        for i in range(self.rowCount()):
            d = {}
            for j, (key, _, typ) in enumerate(self.cols):
                txt = (self.item(i, j).text() if self.item(i, j) else "").strip()
                if txt == "":
                    continue
                try:
                    d[key] = typ(float(txt)) if typ is int else typ(txt)
                except ValueError:
                    d[key] = txt  # la validation signalera la valeur invalide
            out.append(d)
        return out
