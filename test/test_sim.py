"""Tests for the simulation module."""

import matplotlib
import numpy as np
import pytest

from electricpy import sim

matplotlib.use("Agg")


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


def test_nr_pq_includes_diagonal_conductance() -> None:
    """A lossy line with equal bus voltages has zero power injection."""
    ybus = np.array([[1 - 10j, -1 + 10j], [-1 + 10j, 1 - 10j]])
    residual = sim.nr_pq(ybus, [[1, 0], [None, None]], [None, 0], [None, 0])

    np.testing.assert_allclose(residual([0, 1]), [0, 0], atol=1e-12)


def test_nr_pq_mixed_buses_and_return_formats() -> None:
    """Mixed PV and PQ buses use bus order for unknowns and mismatches."""
    ybus = np.array([
        [3 - 12j, -1 + 4j, -1 + 4j, -1 + 4j],
        [-1 + 4j, 2 - 8j, -1 + 4j, 0],
        [-1 + 4j, -1 + 4j, 3 - 12j, -1 + 4j],
        [-1 + 4j, 0, -1 + 4j, 2 - 8j],
    ])
    voltages = [[1.03, 0.12], [None, None], [1.05, None], [None, None]]
    p_set = [None, -1.0, 0.5, -0.4]
    q_set = [None, -0.3, None, -0.2]
    variables = [0.1, -0.04, 0.2, 0.95, 1.02]
    phasors = np.array([1.03, 0.95, 1.05, 1.02]) * np.exp(
        1j * np.array([0.12, 0.1, -0.04, 0.2])
    )
    power = phasors * np.conj(ybus @ phasors)
    expected = np.concatenate((power.real[[1, 2, 3]] - p_set[1:],
                               power.imag[[1, 3]] - np.array([q_set[1], q_set[3]])))

    residual, shape = sim.nr_pq(ybus, voltages, p_set, q_set, argshape=True)
    (p_funcs, q_funcs), split_shape = sim.nr_pq(
        ybus, voltages, p_set, q_set, extend=False, argshape=True
    )

    assert shape == split_shape == (3, 2)
    assert len(p_funcs) == 3
    assert len(q_funcs) == 2
    np.testing.assert_allclose(residual(variables), expected)
    np.testing.assert_allclose([func(variables) for func in p_funcs + q_funcs], expected)


def test_nr_pq_calculators_keep_independent_inputs() -> None:
    """Each calculator retains its own network and specified powers."""
    ybus = np.array([[1 - 10j, -1 + 10j], [-1 + 10j, 1 - 10j]])
    voltages = [[1, 0], [None, None]]
    p_set = [None, 0]
    q_set = [None, 0]
    first = sim.nr_pq(ybus, voltages, p_set, q_set)
    second = sim.nr_pq(ybus, voltages, [None, -2], [None, -1])

    ybus[1, 1] = 100
    voltages[0][0] = 2
    p_set[1] = -3
    q_set[1] = -4

    np.testing.assert_allclose(first([0, 1]), [0, 0], atol=1e-12)
    np.testing.assert_allclose(second([0, 1]), [2, 1], atol=1e-12)
