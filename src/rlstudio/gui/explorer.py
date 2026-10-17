"""Explorateur des blocs d'un modele Simulink : choisir les signaux observes et les parametres commandes."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from .widgets import button, label


class SimulinkExplorer(QDialog):
    """Arbre des blocs du modele : choisir les signaux observes et les parametres commandes."""

    def __init__(self, parent, blocks):
        super().__init__(parent)
        self.setWindowTitle("Explorateur du modele Simulink")
        self.resize(820, 600)
        self.on_observation = self.on_action = None
        self.blocks = [b for b in blocks if b.get("n_out") or b.get("params")]

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filtrer par chemin ou type de bloc...")
        self.filter.textChanged.connect(self._fill)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Bloc / element", "Type / valeur"])
        self.tree.setColumnWidth(0, 500)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.btn_obs = button("Ajouter comme observation", "primary", self._add_obs)
        self.btn_act = button("Ajouter comme action", None, self._add_act)
        row = QHBoxLayout()
        row.addWidget(self.btn_obs)
        row.addWidget(self.btn_act)
        row.addStretch()
        row.addWidget(button("Fermer", "ghost", self.accept))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)
        lay.addWidget(label("Selectionne une sortie de bloc (signal) ou un parametre numerique de bloc.", "hint"))
        lay.addWidget(self.filter)
        lay.addWidget(self.tree, 1)
        lay.addLayout(row)
        self._fill()
        self._update_buttons()

    def _fill(self):
        needle = self.filter.text().lower()
        self.tree.clear()
        for b in self.blocks:
            if needle and needle not in b["path"].lower() and needle not in b["type"].lower():
                continue
            top = QTreeWidgetItem([b["path"].replace("\n", " "), b["type"]])
            for k in range(1, int(b.get("n_out", 0)) + 1):
                it = QTreeWidgetItem([f"Signal : sortie {k}", "signal"])
                it.setData(0, Qt.UserRole, ("signal", b["path"], k))
                top.addChild(it)
            for p in b.get("params", []):
                it = QTreeWidgetItem([f"Parametre : {p['name']}", str(p["value"])])
                it.setData(0, Qt.UserRole, ("param", b["path"], p["name"], p["value"]))
                top.addChild(it)
            self.tree.addTopLevelItem(top)
        if needle:
            self.tree.expandAll()

    def _current(self):
        items = self.tree.selectedItems()
        return items[0].data(0, Qt.UserRole) if items else None

    def _update_buttons(self):
        cur = self._current()
        self.btn_obs.setEnabled(bool(cur and cur[0] == "signal"))
        self.btn_act.setEnabled(bool(cur and cur[0] == "param"))

    @staticmethod
    def _short(path):
        return path.split("/")[-1].replace(" ", "_").replace("\n", "_")

    def _add_obs(self):
        _, path, port = self._current()
        if self.on_observation:
            self.on_observation({"name": self._short(path), "block": path, "port": port, "size": 1})

    def _add_act(self):
        _, path, param, value = self._current()
        if self.on_action:
            v = float(value)
            self.on_action({"name": self._short(path), "block": path, "param": param,
                            "low": min(0.0, v), "high": max(1.0, v * 2)})
