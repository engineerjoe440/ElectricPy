import numpy as np
from numpy.testing import assert_allclose

from electricpy import fault


def test_single_phase_to_ground_fault_sequence():
    v_th = 1
    z_seq = (1, 1, 1)
    result = fault.single_phase_to_ground_fault(v_th, z_seq)

    expected = np.array([1 / (3j)] * 3)
    assert_allclose(result, expected)


def test_phase_to_phase_fault_sequence():
    v_th = 1
    z_seq = (1, 1, 1)
    result = fault.phase_to_phase_fault(v_th, z_seq)
    assert_allclose(result, np.array([0, 1 / (2j), -1 / (2j)]))


def test_three_phase_fault_sequence():
    v_th = 1
    z_seq = (1, 1, 1)
    result = fault.three_phase_fault(v_th, z_seq)
    assert_allclose(result, np.array([0, 1 / (1j), 0]))
