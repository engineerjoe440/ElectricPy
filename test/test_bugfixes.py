import numpy as np
import pytest

import electricpy as ep
from electricpy import conversions
from electricpy import thermal


def test_thermocouple_cjt_units_match_equivalent_voltage():
    """Validate thermocouple cjt units match equivalent voltage behavior."""
    cjt = 25.0
    Vcj = thermal.coldjunction(cjt, coupletype="K")
    temp_from_v = thermal.thermocouple(Vcj, coupletype="K")
    temp_from_cjt = thermal.thermocouple(0.0, coupletype="K", cjt=cjt)
    assert temp_from_v == pytest.approx(temp_from_cjt, rel=1e-6)


def test_ic_555_astable_roundtrip_from_timing():
    """Validate round-trip behavior for ic 555 astable roundtrip from timing."""
    R1, R2, C = 1000.0, 2000.0, 1e-6
    result = ep.ic_555_astable(R=(R1, R2), C=C)
    t_high = result["t_high"]
    t_low = result["t_low"]

    back = ep.ic_555_astable(t_high=t_high, t_low=t_low, C=C)
    assert back["R1"] == pytest.approx(R1, rel=1e-6)
    assert back["R2"] == pytest.approx(R2, rel=1e-6)
    assert back["duty_cycle"] == pytest.approx(result["duty_cycle"], rel=1e-9)


def test_ic_555_astable_freq_branch_returns_combined_resistance():
    """Validate ic 555 astable freq branch returns combined resistance behavior."""
    C = 1e-6
    freq = 10.0
    expected = (1 / freq) / (np.log(2) * C)
    result = ep.ic_555_astable(freq=freq, C=C)
    assert result["R1_plus_2R2"] == pytest.approx(expected, rel=1e-9)


def test_ic_555_monostable_uses_single_pulse_width():
    """Validate ic 555 monostable uses single pulse width behavior."""
    R = 1000.0
    C = 1e-6
    T = R * C * np.log(3)
    result = ep.ic_555_monostable(R=R, C=C, t_high=T)
    assert result == pytest.approx(T, rel=1e-9)


def test_ic_555_monostable_solves_for_resistance_and_capacitance():
    """Validate ic 555 monostable solves for resistance and capacitance behavior."""
    R = 1500.0
    C = 2e-6
    T = R * C * np.log(3)
    calc_r = ep.ic_555_monostable(R=None, C=C, t_high=T)
    calc_c = ep.ic_555_monostable(R=R, C=None, t_high=T)
    assert calc_r == pytest.approx(R, rel=1e-9)
    assert calc_c == pytest.approx(C, rel=1e-9)


def test_short_circuit_rms_branch_uses_provided_frequency():
    """Validate short circuit rms branch uses provided frequency behavior."""
    Z = 1 + 1j
    V = 1
    t = 0.01
    f = 50
    Irms, IAC, K = ep.short_circuit_current(V, Z, t=t, f=f, mxcurrent=False)

    R = abs(Z.real)
    X = abs(Z.imag)
    tau = t * f
    expected_k = np.sqrt(1 + 2 * np.exp(-4 * np.pi * tau / (X / R)))

    assert IAC == pytest.approx(abs(V / Z), rel=1e-12)
    assert K == pytest.approx(expected_k, rel=1e-12)
    assert Irms == pytest.approx(expected_k * abs(V / Z), rel=1e-12)


def test_abc_seq_roundtrip_respects_reference():
    """Validate round-trip behavior for abc seq roundtrip respects reference."""
    abc = np.array([ep.phasor(1, 0), ep.phasor(1, -120), ep.phasor(1, 120)])
    for reference in ("A", "B", "C"):
        seq = conversions.abc_to_seq(abc, reference=reference)
        back = conversions.seq_to_abc(seq, reference=reference)
        assert np.allclose(back, abc)


def test_phasordata_accepts_float_npts():
    """Validate phasordata accepts float npts behavior."""
    data = ep.phasors.phasordata(0, 1, npts=5.7)
    assert len(data) == 5
