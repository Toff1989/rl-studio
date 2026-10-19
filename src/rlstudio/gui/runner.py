"""Lancement des processus de calcul depuis l'interface (QProcess) : entrainement, explorateur, outils."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .. import runtime
from ..settings import Settings


class RuntimeMissing(RuntimeError):
    pass


def _worker_process(settings: Settings, args: list[str], parent) -> QProcess:
    py = runtime.worker_python(settings)
    if py is None:
        raise RuntimeMissing("Composants de calcul non installes : ouvre Parametres > Composants.")
    proc = QProcess(parent)
    env = QProcessEnvironment()
    for k, v in runtime.worker_env(settings).items():
        env.insert(k, v)
    proc.setProcessEnvironment(env)
    proc.setProcessChannelMode(QProcess.MergedChannels)
    proc.setProgram(str(py))
    proc.setArguments(["-m", "rlstudio.worker", *args])
    return proc


class _LineReader:
    """Decoupe le flux du processus en lignes completes."""

    def __init__(self):
        self.buf = ""

    def feed(self, data: bytes):
        self.buf += data.decode("utf-8", "replace")
        *lines, self.buf = self.buf.split("\n")
        return [ln.rstrip("\r") for ln in lines]


class TrainRunner(QObject):
    episode = Signal(int, float, int, int)
    state = Signal(object)
    baseline = Signal(float, str)
    meta = Signal(list, list)  # noms des observations, noms des actions
    log = Signal(str)
    progress = Signal(str, float)  # phase ("baseline"), avancement 0..1
    finished = Signal(str)  # "Termine..." ou "ERREUR..."

    GRACE_MS = 3000  # delai laisse a un arret propre quand il apporte quelque chose (modele partiel enregistre)

    def __init__(self, project, settings: Settings, parent=None):
        super().__init__(parent)
        self.project, self.settings = project, settings
        self.proc = None
        self._reader = _LineReader()
        self._final = None
        self._stopping = self._killed = False
        self._graceful = False  # arret propre utile ? (seul l'apprentissage en cours enregistre un modele partiel)
        self.run_dir = None

    def start(self):
        self.proc = _worker_process(self.settings, ["train", "--project", str(self.project.path),
                                                    "--settings", str(self.settings.path)], self)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.start()

    def _on_error(self, error):
        if self._killed:  # « Crashed » est le signal normal d'un process que nous venons de tuer
            return
        self._fail(f"Impossible de lancer le processus de calcul ({error}).")

    def stop(self):
        """Arret immediat : le process est tue sur-le-champ (avec MATLAB, qu'il a lance), sauf pendant l'apprentissage,
        ou l'on laisse quelques secondes au modele partiel pour s'enregistrer."""
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            return
        self._stopping = True
        if self._graceful:
            self.proc.write(b"stop\n")
            QTimer.singleShot(self.GRACE_MS, self._kill)
        else:
            self._kill()

    def _kill(self):
        """Tue le process sans bloquer l'interface. Sous Windows, `taskkill /T` s'execute d'abord (en arriere-plan) pour
        tuer l'arbre entier (le python du venv est un lanceur, MATLAB un process fils) : le pere n'est tue qu'apres,
        sinon les fils deviennent orphelins et ne sont plus retrouves."""
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            return
        self._killed = True
        pid = self.proc.processId()
        if sys.platform == "win32" and pid:
            killer = QProcess(self)
            killer.setProgram("taskkill")
            killer.setArguments(["/F", "/T", "/PID", str(pid)])
            killer.finished.connect(lambda *_: self._hard_kill())
            killer.errorOccurred.connect(lambda *_: self._hard_kill())
            killer.start()
            QTimer.singleShot(5000, self._hard_kill)  # filet de securite si taskkill ne repond pas
        else:
            self.proc.kill()

    def _hard_kill(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def _on_output(self):
        for line in self._reader.feed(bytes(self.proc.readAllStandardOutput())):
            self._handle(line)

    def _handle(self, line: str):
        if not line.startswith("{"):
            if line.strip():
                self.log.emit(line)
            return
        try:
            ev = json.loads(line)
        except ValueError:
            self.log.emit(line)
            return
        kind = ev.get("type")
        if kind == "run":
            self.run_dir = Path(ev["run_dir"])
        elif kind == "state":
            self._graceful = True  # l'apprentissage a commence : un arret propre enregistre le modele partiel
            self.state.emit({"timesteps": ev["timesteps"], "obs": np.asarray(ev["obs"]),
                             "action": np.asarray(ev["action"]), "reward": ev["reward"]})
        elif kind == "episode":
            self.episode.emit(ev["n"], ev["reward"], ev["length"], ev["steps"])
        elif kind == "baseline":
            self.baseline.emit(ev["mean"], ev["label"])
        elif kind == "meta":
            self.meta.emit(list(ev.get("obs_names") or []), list(ev.get("action_names") or []))
        elif kind == "log":
            self.log.emit(ev["text"])
        elif kind == "progress":
            self.progress.emit(ev.get("stage", ""), float(ev.get("fraction", 0)))
        elif kind == "done":
            self._final = ("Termine. Resultats : " + ev["run_dir"]) if not ev.get("stopped") else \
                          "Arrete. Resultats partiels : " + ev["run_dir"]
        elif kind == "error":
            self._final = "ERREUR :\n" + ev.get("trace", ev.get("text", ""))

    def _fail(self, text):
        if self._final is None:
            self._final = "ERREUR : " + text

    def _on_finished(self, code, _status):
        self._on_output()
        if self._reader.buf.strip():
            self._handle(self._reader.buf)
        if self._killed and self.run_dir and not (self.run_dir / "model.zip").exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)  # run interrompu avant l'enregistrement : rien a garder
            self.log.emit("Run interrompu avant l'enregistrement du modele : dossier supprime.")
        if self._stopping and self._final is None:
            self._final = "Arrete."
        self.finished.emit(self._final or f"ERREUR : le processus de calcul s'est arrete (code {code}).")


class EvalRunner(TrainRunner):
    """Evaluation en direct (commande `evaluate` du worker) : signaux `step` et `episode`."""

    step = Signal(object)
    episode_done = Signal(object)

    def __init__(self, project, settings: Settings, run_name: str, policy: str, episodes: int, seed: int, delay: float, parent=None):
        super().__init__(project, settings, parent)
        self.args = ["evaluate", "--project", str(project.path), "--run", run_name, "--policy", policy,
                     "--episodes", str(episodes), "--seed", str(seed), "--delay", str(delay)]

    def start(self):
        self.proc = _worker_process(self.settings, self.args, self)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.start()

    def _handle(self, line: str):
        if line.startswith("{"):
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {}
            kind = ev.get("type")
            if kind == "step":
                self.step.emit({"episode": ev["episode"], "step": ev["step"], "obs": np.asarray(ev["obs"]),
                                "action": np.asarray(ev["action"]), "reward": ev["reward"], "total": ev["total"]})
                return
            if kind == "episode":
                self.episode_done.emit(ev)
                return
            if kind == "done":
                self._final = "Arrete." if ev.get("stopped") else "Termine."
                return
        super()._handle(line)


class DistillRunner(TrainRunner):
    """Extraction d'un modele mathematique (commande `distill` du worker) ; meme protocole que l'entrainement + `progress`."""

    progress = Signal(str, float)

    def __init__(self, project, settings: Settings, run_name: str, options: dict, parent=None):
        super().__init__(project, settings, parent)
        self.run_name, self.options = run_name, options

    def start(self):
        self.proc = _worker_process(self.settings, ["distill", "--project", str(self.project.path), "--run", self.run_name,
                                                    "--options", json.dumps(self.options)], self)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.start()

    def _handle(self, line: str):
        if line.startswith("{"):
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {}
            if ev.get("type") == "progress":
                self.progress.emit(ev.get("stage", ""), float(ev.get("fraction", 0)))
                return
            if ev.get("type") == "done":
                self._final = "Arrete." if ev.get("stopped") else "Termine."
                return
        super()._handle(line)


class WorkerJob(QObject):
    """Commande ponctuelle du worker (inspect, matlab-test, env-info) ; resultat via `done`."""

    done = Signal(object, str)  # (dict resultat | None, message d'erreur)

    def __init__(self, settings: Settings, args: list[str], parent=None):
        super().__init__(parent)
        self.settings, self.args = settings, args
        self._reader = _LineReader()
        self._lines = []

    def start(self):
        try:
            self.proc = _worker_process(self.settings, self.args, self)
        except RuntimeMissing as e:
            QTimer.singleShot(0, lambda: self.done.emit(None, str(e)))
            return
        self.proc.readyReadStandardOutput.connect(self._on_ready)
        self.proc.finished.connect(self._finish)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.start()

    def _on_ready(self):
        self._lines.extend(self._reader.feed(bytes(self.proc.readAllStandardOutput())))

    def _on_error(self, error):
        self.done.emit(None, f"Lancement impossible ({error})")

    def _finish(self, code, _status):
        self._lines.extend(self._reader.feed(bytes(self.proc.readAllStandardOutput())))
        if self._reader.buf:
            self._lines.append(self._reader.buf)
        events = []
        for ln in self._lines:
            if ln.startswith("{"):
                try:
                    events.append(json.loads(ln))
                except ValueError:
                    pass
        err = next((e for e in events if e.get("type") == "error"), None)
        res = next((e for e in events if e.get("type") == "result"), None)
        if res is not None:
            self.done.emit(res, "")
        else:
            self.done.emit(None, (err or {}).get("text") or "\n".join(self._lines[-5:]) or f"code de sortie {code}")


class ToolProcess(QObject):
    """Outils console (scan, provision) : sortie ligne a ligne."""

    line = Signal(str)
    finished = Signal(int)

    def __init__(self, args: list[str], parent=None):
        super().__init__(parent)
        program, arguments = runtime.tools_command(args)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setProgram(program)
        self.proc.setArguments(arguments)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self._reader = _LineReader()
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(lambda code, _s: (self._read(), self.finished.emit(code)))
        self.proc.errorOccurred.connect(lambda e: self.line.emit(f"Erreur de lancement : {e}"))

    def _on_finished(self, code, _status):
        self._read()
        self.finished.emit(code)

    def _on_error(self, error):
        self.line.emit(f"Erreur de lancement : {error}")

    def start(self):
        self.proc.start()

    def kill(self):
        self.proc.kill()

    def _read(self):
        for ln in self._reader.feed(bytes(self.proc.readAllStandardOutput())):
            self.line.emit(ln)
