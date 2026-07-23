import numpy as np
import pytest

from electricpy import conversions


def test_sequencez_reference_invariant_for_symmetric_matrix():
    """Validate sequencez reference invariant for symmetric matrix behavior."""
    Zabc = np.array([[1 + 0j, 0, 0], [0, 2 + 0j, 0], [0, 0, 3 + 0j]])
    ref_a = conversions.sequencez(Zabc, reference="A")
    ref_b = conversions.sequencez(Zabc, reference="B")
    ref_c = conversions.sequencez(Zabc, reference="C")
    assert np.allclose(np.diag(ref_a), np.diag(ref_b))
    assert np.allclose(np.diag(ref_a), np.diag(ref_c))
    assert np.allclose(ref_a, ref_a.T.conjugate())


def test_abc_seq_round_trip_with_reference():
    """Validate round-trip behavior for abc seq round trip with reference."""
    data = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    seq = conversions.abc_to_seq(data, reference="B")
    abc = conversions.seq_to_abc(seq, reference="B")
    assert np.allclose(abc, data)


def test_abc_to_seq_invalid_reference():
    """Validate error handling for abc to seq invalid reference."""
    data = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    with pytest.raises(ValueError):
        conversions.abc_to_seq(data, reference="D")


@pytest.mark.parametrize("hz_value", [0.1, 1.0, 60.0, 1234.5])
def test_hz_rad_round_trip(hz_value: float) -> None:
    """Frequency should round-trip between Hz and rad/s."""
    radians = conversions.hz_to_rad(hz_value)
    assert conversions.rad_to_hz(radians) == pytest.approx(hz_value, rel=1e-12)


@pytest.mark.parametrize("rpm_value", [1.0, 900.0, 1800.0, 3600.0])
def test_rpm_hz_round_trip(rpm_value: float) -> None:
    """Frequency should round-trip between RPM and Hz."""
    hz = conversions.rpm_to_hz(rpm_value)
    assert conversions.hz_to_rpm(hz) == pytest.approx(rpm_value, rel=1e-12)


@pytest.mark.parametrize("watts", [1e-6, 1.0, 10.0, 5e3])
def test_dbw_watts_round_trip(watts: float) -> None:
    """Positive power should round-trip through dBW conversion."""
    dbw = conversions.watts_to_dbw(watts)
    assert conversions.dbw_to_watts(dbw) == pytest.approx(watts, rel=1e-12)
