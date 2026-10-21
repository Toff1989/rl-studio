"""Actions sur les projets : liste, creation, enregistrement, partage (.rls), conversion Simulink."""
import os
from pathlib import Path
import sys

import yaml
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFileDialog, QInputDialog, QListWidgetItem, QMessageBox, QVBoxLayout, QWidget,
)

from ..project import Project
from .widgets import label


class ProjectActions:
    """Methodes de MainWindow (actions sur les projets : liste, creation, enregistrement, partage)."""

    def _item_widget(self, p: Project):
        w = QWidget()
        w.setAttribute(Qt.WA_TransparentForMouseEvents)
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(1)
        t = label(p.name)
        t.setStyleSheet("font-weight: 600; font-size: 14px;")
        kind = "Python" if p.kind == "python" else "Simulink"
        lay.addWidget(t)
        lay.addWidget(label(f"{kind}  ·  {p.cfg['algo']['name']}", "muted"))
        return w

    def _refresh_list(self, select=None):
        self.plist.blockSignals(True)
        self.plist.clear()
        for p in self.store.list():
            it = QListWidgetItem()
            it.setData(Qt.UserRole, p.slug)
            it.setSizeHint(QSize(0, 56))
            self.plist.addItem(it)
            self.plist.setItemWidget(it, self._item_widget(p))
            if p.slug == select:
                self.plist.setCurrentItem(it)
        self.plist.blockSignals(False)
        if select is None and self.plist.count():
            self.plist.setCurrentRow(0)
        elif self.plist.count() == 0:
            self._load_project(None)

    def _on_select(self, cur, prev):
        if self.worker:  # pas de changement pendant un entrainement
            return
        if self.dirty and not self._ask_save():
            self.plist.blockSignals(True)
            for i in range(self.plist.count()):
                if self.project and self.plist.item(i).data(Qt.UserRole) == self.project.slug:
                    self.plist.setCurrentRow(i)
            self.plist.blockSignals(False)
            return
        self._load_project(self.store.get(cur.data(Qt.UserRole)) if cur else None)

    def _ask_save(self):
        r = QMessageBox.question(self, "Modifications non enregistrees", "Enregistrer les modifications ?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self._save()
        self._set_dirty(False)
        return True

    def _new(self):
        if self.dirty and not self._ask_save():
            return
        templates = self.store.list_templates()
        labels = [f"{n} - {d}" for _, n, d in templates]
        choice, ok = QInputDialog.getItem(self, "Nouveau projet", "Modele de depart :", labels, 0, False)
        if not ok:
            return
        tid = templates[labels.index(choice)][0]
        name, ok = QInputDialog.getText(self, "Nouveau projet", "Nom du projet :")
        if not ok or not name.strip():
            return
        try:
            p = self.store.create(name, tid)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
            return
        self._set_dirty(False)
        self._refresh_list(select=p.slug)
        self._load_project(p)

    def _duplicate(self):
        if self.project and (not self.dirty or self._ask_save()):
            p = self.store.duplicate(self.project)
            self._refresh_list(select=p.slug)
            self._load_project(p)

    def _rename(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "Renommer", "Nouveau nom :", text=self.project.name)
        if ok and name.strip():
            self.store.rename(self.project, name)
            self._refresh_list(select=self.project.slug)
            self._load_project(self.store.get(self.project.slug))

    def _delete(self):
        if not self.project:
            return
        r = QMessageBox.question(
            self, "Supprimer", f"Supprimer le projet « {self.project.name} » ?\n"
            "Il sera deplace dans projects/.trash/ et pourra etre recupere.")
        if r == QMessageBox.Yes:
            self.store.delete(self.project)
            self.project = None
            self._set_dirty(False)
            self._refresh_list()

    def _convert_dialog(self):
        from .convert_dialog import ConvertDialog

        if self.dirty and not self._ask_save():
            return
        if self.worker or any(getattr(t, "runner", None) for t in (self.distill, self.evaltab, self.identify)):
            QMessageBox.information(self, "Convertir un modele", "Attends la fin du calcul en cours avant de convertir un modele.")
            return
        dlg = ConvertDialog(self, self.settings, self.store)
        dlg.exec()
        if dlg.created_slug:
            self._set_dirty(False)
            self._refresh_list(select=dlg.created_slug)
            self._load_project(self.store.get(dlg.created_slug))

    def _export_dialog(self):
        from ..archive import export_project

        if not self.project or (self.dirty and not self._save()):
            return
        labels = ["Sans resultats d'entrainement (le plus leger)", "Avec le dernier entrainement et son modele",
                  "Avec tous les entrainements"]
        choice, ok = QInputDialog.getItem(self, "Exporter le projet", "Contenu de l'archive :", labels, 0, False)
        if not ok:
            return
        mode = ("none", "last", "all")[labels.index(choice)]
        ident = "last"
        if (self.project.path / "identify").is_dir() and any((self.project.path / "identify").glob("*/report.json")):
            id_labels = ["Avec la derniere identification (reglages inclus)", "Avec toutes les identifications", "Sans resultats d'identification"]
            id_choice, ok = QInputDialog.getItem(self, "Exporter le projet", "Identification de modele reduit :", id_labels, 0, False)
            if not ok:
                return
            ident = ("last", "all", "none")[id_labels.index(id_choice)]
        default = str(Path.home() / "Documents" / f"{self.project.slug}.rls")
        path, _ = QFileDialog.getSaveFileName(self, "Exporter le projet", default, "Projet RL Studio (*.rls)")
        if not path:
            return
        try:
            out = export_project(self.project, path, runs=mode, identify=ident)
        except Exception as e:
            QMessageBox.critical(self, "Export impossible", str(e))
            return
        QMessageBox.information(self, "Projet exporte", f"{out}\n\nTaille : {out.stat().st_size / 1024:.0f} Ko")

    def _import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importer un projet", str(Path.home()), "Projet RL Studio (*.rls)")
        if path:
            self.import_file(path)

    def _confirm_import(self, report) -> bool:
        m = report["manifest"]
        lines = [f"<b>{m['name']}</b> ({m['kind']}) - {report['files']} fichiers, {report['size'] / 1024:.0f} Ko"]
        if report["runs"]:
            lines.append(f"Resultats d'entrainement inclus : {len(report['runs'])}")
        if report.get("identifications"):
            lines.append(f"Resultats d'identification inclus : {len(report['identifications'])}")
        if report["code_files"]:
            shown = ", ".join(report["code_files"][:8]) + (" ..." if len(report["code_files"]) > 8 else "")
            lines.append("<br><b>Attention :</b> ce projet contient du <b>code qui s'executera sur votre ordinateur</b> "
                         f"(environnement Python, modele Simulink, scripts) :<br><i>{shown}</i><br><br>"
                         "N'importez que des projets provenant de personnes de confiance.")
        box = QMessageBox(self)
        box.setWindowTitle("Importer un projet")
        box.setTextFormat(Qt.RichText)
        box.setText("<br>".join(lines))
        box.setIcon(QMessageBox.Warning if report["code_files"] else QMessageBox.Question)
        yes = box.addButton("Importer", QMessageBox.AcceptRole)
        box.addButton("Annuler", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is yes

    def import_file(self, path, confirm=None):
        """Importe un .rls (menu, glisser-deposer ou double-clic). `confirm(report) -> bool` remplace la boite de dialogue."""
        from ..archive import ArchiveError, import_project, inspect_archive

        try:
            report = inspect_archive(path)
            if report["problems"]:
                raise ArchiveError("Import refuse :\n- " + "\n- ".join(report["problems"]))
            if not (confirm or self._confirm_import)(report):
                return None
            if self.dirty and not self._ask_save():
                return None
            project = import_project(path, self.store)
        except ArchiveError as e:
            QMessageBox.warning(self, "Import impossible", str(e))
            return None
        self._set_dirty(False)
        self._refresh_list(select=project.slug)
        self._load_project(project)
        return project

    def dragEnterEvent(self, event):
        if any(u.toLocalFile().lower().endswith(".rls") for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for u in event.mimeData().urls():
            if u.toLocalFile().lower().endswith(".rls"):
                self.import_file(u.toLocalFile())

    def _open_folder(self):
        target = self.project.path if self.project else self.store.root
        if sys.platform == "win32":
            os.startfile(target)

    def _set_enabled(self, on):
        self.tabs.setEnabled(on)
        self.btn_actions.setEnabled(on)
        for a in (self.act_dup, self.act_ren, self.act_dir, self.act_exp, self.act_del):
            a.setEnabled(on)

    def _set_dirty(self, flag):
        self.dirty = flag
        self.lbl_dirty.setVisible(flag)
        self.btn_save.setEnabled(flag)

    def _load_project(self, project: Project | None):
        self.project = project
        self._loading = True
        self._set_enabled(project is not None)
        if project is None:
            self.top_title.setText("Aucun projet")
            self.top_badge.setText("")
            self._set_dirty(False)
            self._loading = False
            return
        c, env = project.cfg, project.cfg["env"]
        self.f_name.setText(project.name)
        self.f_desc.setPlainText(c.get("description", ""))
        is_py = project.kind == "python"
        self.env_stack.setCurrentIndex(0 if is_py else 1)
        if is_py:
            self.py_entry.setText(env.get("entry", "env.py:make_env"))
            self.py_params.setPlainText(yaml.safe_dump(env.get("params") or {}, sort_keys=False, allow_unicode=True))
        else:
            self.sl_model.setText(env.get("model", ""))
            self.sl_step.setValue(float(env.get("step_s", 60)))
            self.sl_episode.setValue(float(env.get("episode_s", 3600)))
            self.sl_reward.setText(env.get("reward", ""))
            self.sl_terminate.setText(env.get("terminate", "") or "")
            self.t_actions.set_rows(env.get("actions"))
            self.t_obs.set_rows(env.get("observations"))
            self.t_init.set_rows(env.get("initial"))
            self.sl_status.hide()
            self._update_vars_hint()
        a = c["algo"]
        self.a_name.setCurrentText(a["name"])
        self.a_steps.setValue(int(a["total_timesteps"]))
        self.a_seed.setValue(int(a["seed"]))
        self.a_device.setCurrentText(a["device"])
        self.a_base.setValue(int(a.get("baseline_episodes", 0)))
        self.a_kwargs.setPlainText(yaml.safe_dump(a.get("kwargs") or {}, sort_keys=False) if a.get("kwargs") else "")
        self._refresh_runs()
        self.distill.set_project(project)
        self.identify.set_project(project)
        self.evaltab.set_project(project)
        self._reset_train_state()
        self._update_dashboard()
        self._set_dirty(False)
        self._loading = False
        self.setWindowTitle(f"RL Studio - {project.name}")

    def _mark_dirty(self, *_):
        if not self._loading and self.project:
            self._set_dirty(True)

    def _collect(self):
        """Ecrit les widgets dans project.cfg ; leve ValueError si une saisie est invalide."""
        c = self.project.cfg
        c["name"] = self.f_name.text().strip() or self.project.slug
        c["description"] = self.f_desc.toPlainText()
        try:
            if self.project.kind == "python":
                c["env"]["entry"] = self.py_entry.text().strip()
                c["env"]["params"] = yaml.safe_load(self.py_params.toPlainText()) or {}
                if not isinstance(c["env"]["params"], dict):
                    raise ValueError("Les parametres doivent etre un dictionnaire YAML.")
            else:
                c["env"].update({
                    "model": self.sl_model.text().strip(), "step_s": self.sl_step.value(),
                    "episode_s": self.sl_episode.value(), "reward": self.sl_reward.text().strip(),
                    "terminate": self.sl_terminate.text().strip(), "actions": self.t_actions.rows(),
                    "observations": self.t_obs.rows(), "initial": self.t_init.rows(),
                })
            kw = yaml.safe_load(self.a_kwargs.toPlainText()) or {}
            if not isinstance(kw, dict):
                raise ValueError("Les hyperparametres doivent etre un dictionnaire YAML.")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML invalide : {e}")
        c["algo"].update(name=self.a_name.currentText(), total_timesteps=self.a_steps.value(),
                         seed=self.a_seed.value(), device=self.a_device.currentText(),
                         baseline_episodes=self.a_base.value(), kwargs=kw)

    def _save(self):
        if not self.project:
            return True
        try:
            self._collect()
        except ValueError as e:
            QMessageBox.warning(self, "Saisie invalide", str(e))
            return False
        self.project.save()
        self._update_dashboard()
        self._set_dirty(False)
        self.setWindowTitle(f"RL Studio - {self.project.name}")
        self._refresh_list(select=self.project.slug)
        return True
