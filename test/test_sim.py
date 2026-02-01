import matplotlib
import numpy as np

matplotlib.use("Agg")

from electricpy import sim


def test_digifiltersim_basic():
    plot_module = sim.digifiltersim(
        lambda t, f: np.sin(2 * np.pi * f * t),
        [0.0, 0.0, 1.0, 0.0, 0.0],
        freqs=[1],
        NN=10,
        dt=0.1,
        legend=False
    )

    assert plot_module is not None


def test_step_response_returns_plot():
    system = (np.array([1.0]), np.array([1.0, 1.0]))
    plot_module = sim.step_response(system, npts=5, dt=0.1, combine=False, xlim=(0, 1))
    assert plot_module is not None
