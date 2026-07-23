import pytest

from electricpy import passive


def test_vcap_charge_discharge_initial():
    """Validate vcap charge discharge initial behavior."""
    vs = 5
    assert passive.vcapcharge(0, vs, 1, 1) == 0
    assert passive.vcapdischarge(0, vs, 1, 1) == vs


def test_capbacktoback_vln():
    """Validate capbacktoback vln behavior."""
    imax, ifreq = passive.capbacktoback(1e-6, 1e-6, 1e-3, VLN=120)
    assert imax > 0
    assert ifreq > 0


def test_captransfer_validation():
    """Validate error handling for captransfer validation."""
    with pytest.raises(ValueError):
        passive.captransfer(1, 1, 1, 0, 0)


def test_captransfer_negative_time_raises():
    """Validate error handling for captransfer negative time raises."""
    with pytest.raises(ValueError):
        passive.captransfer(-1, 1, 1, 1, 1)


def test_capbacktoback_requires_voltage_input():
    """Validate error handling for capbacktoback requires voltage input."""
    with pytest.raises(ValueError):
        passive.capbacktoback(1e-6, 1e-6, 1e-3)


def test_farads_matches_formula():
    """Validate farads matches formula behavior."""
    var = 1000
    voltage = 120
    freq = 60
    capacitance = passive.farads(var, voltage, freq=freq)
    assert capacitance > 0


def test_vcapcharge_negative_time():
    """Validate vcapcharge negative time behavior."""
    with pytest.raises(ValueError):
        passive.vcapcharge(-1, 1, 1, 1)


def test_capenergy_matches_formula() -> None:
    """Capacitor energy should follow 1/2*C*V^2."""
    c_val = 2.0e-6
    voltage = 400.0
    expected = 0.5 * c_val * voltage * voltage
    assert passive.capenergy(c_val, voltage) == pytest.approx(expected, rel=1e-12)


def test_inductorenergy_matches_formula() -> None:
    """Inductor energy should follow 1/2*L*I^2."""
    inductance = 5.0e-3
    current = 12.0
    expected = 0.5 * inductance * current * current
    assert passive.inductorenergy(inductance, current) == pytest.approx(expected, rel=1e-12)
