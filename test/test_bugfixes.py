import numpy as np
import pytest

import electricpy as ep
from electricpy import conversions
from electricpy import thermal


def test_thermocouple_cjt_units_match_equivalent_voltage():
    cjt = 25.0
    Vcj = thermal.coldjunction(cjt, coupletype="K")
    temp_from_v = thermal.thermocouple(Vcj, coupletype="K")
    temp_from_cjt = thermal.thermocouple(0.0, coupletype="K", cjt=cjt)
    assert temp_from_v == pytest.approx(temp_from_cjt, rel=1e-6)


def test_ic_555_astable_roundtrip_from_timing():
    R1, R2, C = 1000.0, 2000.0, 1e-6
    result = ep.ic_555_astable(R=(R1, R2), C=C)
    t_high = result["t_high"]
    t_low = result["t_low"]

    back = ep.ic_555_astable(t_high=t_high, t_low=t_low, C=C)
    assert back["R1"] == pytest.approx(R1, rel=1e-6)
    assert back["R2"] == pytest.approx(R2, rel=1e-6)
    assert back["duty_cycle"] == pytest.approx(result["duty_cycle"], rel=1e-9)


def test_ic_555_astable_freq_branch_returns_combined_resistance():
    C = 1e-6
    freq = 10.0
    expected = (1 / freq) / (np.log(2) * C)
    result = ep.ic_555_astable(freq=freq, C=C)
    assert result["R1_plus_2R2"] == pytest.approx(expected, rel=1e-9)


def test_ic_555_monostable_uses_single_pulse_width():
    R = 1000.0
    C = 1e-6
    T = R * C * np.log(3)
    result = ep.ic_555_monostable(R=R, C=C, t_high=T)
    assert result == pytest.approx(T, rel=1e-9)


def test_abc_seq_roundtrip_respects_reference():
    abc = np.array([ep.phasor(1, 0), ep.phasor(1, -120), ep.phasor(1, 120)])
    for reference in ("A", "B", "C"):
        seq = conversions.abc_to_seq(abc, reference=reference)
        back = conversions.seq_to_abc(seq, reference=reference)
        assert np.allclose(back, abc)


def test_phasordata_accepts_float_npts():
    data = ep.phasors.phasordata(0, 1, npts=5.7)
    assert len(data) == 5
