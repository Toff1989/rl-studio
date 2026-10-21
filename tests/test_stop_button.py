"""Le bouton Arreter doit repondre tout de suite : process tue (avec ses fils), pas d'attente ni de faux message d'erreur."""
import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _runner(monkeypatch, tmp_path, script):
    from PySide6.QtCore import QProcess

    from rlstudio.gui import runner as runner_mod
    from rlstudio.settings import Settings

    def fake_worker(settings, args, parent):
        proc = QProcess(parent)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setProgram(sys.executable)
        proc.setArguments(["-u", "-c", script])
        return proc

    monkeypatch.setattr(runner_mod, "_worker_process", fake_worker)
    project = type("P", (), {"path": tmp_path})()
    return runner_mod.TrainRunner(project, Settings(tmp_path / "s.json"))


def _run_until_finished(qapp, r, action, timeout=10):
    done = []
    r.finished.connect(done.append)
    r.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:  # laisse le process demarrer
        qapp.processEvents()
        time.sleep(0.01)
    t_stop = time.monotonic()
    action()
    while not done and time.monotonic() - t_stop < timeout:
        qapp.processEvents()
        time.sleep(0.005)
    return done, time.monotonic() - t_stop


def test_stop_before_learning_kills_immediately(qapp, monkeypatch, tmp_path):
    r = _runner(monkeypatch, tmp_path, "import time; print('{\"type\": \"log\", \"text\": \"phase de reference\"}'); time.sleep(60)")
    done, elapsed = _run_until_finished(qapp, r, lambda: r.stop())
    assert done == ["Arrêté."]  # pas « ERREUR : Crashed »
    assert elapsed < 2.0


def test_stop_during_learning_lets_the_model_save(qapp, monkeypatch, tmp_path):
    script = ("import sys, time, json\n"
              "print(json.dumps({'type': 'state', 'timesteps': 1, 'obs': [0.0], 'action': [0.0], 'reward': 0.0}))\n"
              "sys.stdin.readline()\n"
              "print(json.dumps({'type': 'done', 'run_dir': 'x', 'stopped': True}))\n")
    r = _runner(monkeypatch, tmp_path, script)
    r.log.connect(lambda m: None)
    done, elapsed = _run_until_finished(qapp, r, lambda: r.stop())
    assert r._graceful and done and done[0].startswith("Arrêté. Résultats partiels")
    assert elapsed < 2.0


def test_unresponsive_learning_worker_is_killed_after_grace(qapp, monkeypatch, tmp_path):
    script = ("import time, json\n"
              "print(json.dumps({'type': 'state', 'timesteps': 1, 'obs': [0.0], 'action': [0.0], 'reward': 0.0}))\n"
              "time.sleep(60)\n")
    r = _runner(monkeypatch, tmp_path, script)
    r.GRACE_MS = 400
    done, elapsed = _run_until_finished(qapp, r, lambda: r.stop())
    assert done == ["Arrêté."] and 0.3 < elapsed < 3.0
