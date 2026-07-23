import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from electricpy import sim


def test_digifiltersim_basic():
    """Validate digifiltersim basic behavior."""
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
    """Step response helper should return a plotting module."""
    system = (np.array([1.0]), np.array([1.0, 1.0]))
    plot_module = sim.step_response(system, npts=5, dt=0.1, combine=False, xlim=(0, 1))
    assert plot_module is not None


def test_nr_pq_rejects_non_square_ybus() -> None:
    """Power-flow generator requires a square Y-bus matrix."""
    ybus = np.array([[1 + 0j, 2 + 0j, 3 + 0j], [4 + 0j, 5 + 0j, 6 + 0j]])
    with pytest.raises(ValueError, match="Invalid Y-Bus Shape"):
        sim.nr_pq(ybus, V_set=[[1, 0], [None, None]], P_set=[None, -1.0], Q_set=[None, -0.5])
