import numpy as np
import pytest

from electricpy import conversions


def test_sequencez_reference_invariant_for_symmetric_matrix():
    Zabc = np.array([[1 + 0j, 0, 0], [0, 2 + 0j, 0], [0, 0, 3 + 0j]])
    ref_a = conversions.sequencez(Zabc, reference="A")
    ref_b = conversions.sequencez(Zabc, reference="B")
    ref_c = conversions.sequencez(Zabc, reference="C")
    assert np.allclose(np.diag(ref_a), np.diag(ref_b))
    assert np.allclose(np.diag(ref_a), np.diag(ref_c))
    assert np.allclose(ref_a, ref_a.T.conjugate())


def test_abc_seq_round_trip_with_reference():
    data = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    seq = conversions.abc_to_seq(data, reference="B")
    abc = conversions.seq_to_abc(seq, reference="B")
    assert np.allclose(abc, data)


def test_abc_to_seq_invalid_reference():
    data = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    with pytest.raises(ValueError):
        conversions.abc_to_seq(data, reference="D")
