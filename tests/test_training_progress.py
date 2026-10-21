"""Suivi detaille de l'entrainement : evaluation de reference observable, durees lisibles."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rlstudio.training import evaluate, fmt_duration


class _TenSteps(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        self.k = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.k += 1
        return np.zeros(1, np.float32), 1.0, False, self.k >= 10, {}


def test_fmt_duration():
    assert fmt_duration(5) == "5 s"
    assert fmt_duration(62) == "1 min 02 s"
    assert fmt_duration(3 * 3600 + 5 * 60) == "3 h 05 min"
    assert fmt_duration(-3) == "0 s"


def test_evaluate_reports_episodes_and_steps():
    episodes, steps = [], []
    mean, std = evaluate(_TenSteps(), lambda obs: np.zeros(1, np.float32), 3,
                         on_episode=lambda i, n, ret, dur: episodes.append((i, n, ret)),
                         on_step=lambda i, n, k: steps.append((i, k)))
    assert (mean, std) == (10.0, 0.0)
    assert episodes == [(0, 3, 10.0), (1, 3, 10.0), (2, 3, 10.0)]
    assert len(steps) == 30 and steps[0] == (0, 1) and steps[-1] == (2, 10)
