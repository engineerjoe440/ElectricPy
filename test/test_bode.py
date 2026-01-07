import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from electricpy import bode as ep_bode


def test_sys_condition_feedback_padding():
    num = np.array([1.0, 0.0])
    den = np.array([1.0, 1.0, 0.0])
    conditioned_num, conditioned_den = ep_bode._sys_condition((num, den), True)

    assert conditioned_den.shape[0] >= conditioned_num.shape[0]
    assert np.allclose(conditioned_num, np.array([1.0, 0.0]))
    assert np.allclose(conditioned_den, np.array([1.0, 2.0, 0.0]))


def test_bode_validates_frequency_range():
    system = (np.array([1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        ep_bode.bode(system, mn=10, mx=1, magnitude=False, angle=False)


def test_sbode_returns_plot_module():
    func = lambda s: 1 / (s + 1)
    plot_module = ep_bode.sbode(func, NN=10, mn=1, mx=10, magnitude=False, angle=False)
    assert plot_module is None
