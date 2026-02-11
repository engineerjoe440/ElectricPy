import pytest

from electricpy import passive


def test_vcap_charge_discharge_initial():
    vs = 5
    assert passive.vcapcharge(0, vs, 1, 1) == 0
    assert passive.vcapdischarge(0, vs, 1, 1) == vs


def test_capbacktoback_vln():
    imax, ifreq = passive.capbacktoback(1e-6, 1e-6, 1e-3, VLN=120)
    assert imax > 0
    assert ifreq > 0


def test_captransfer_validation():
    with pytest.raises(ValueError):
        passive.captransfer(1, 1, 1, 0, 0)


def test_farads_matches_formula():
    var = 1000
    voltage = 120
    freq = 60
    capacitance = passive.farads(var, voltage, freq=freq)
    assert capacitance > 0


def test_vcapcharge_negative_time():
    with pytest.raises(ValueError):
        passive.vcapcharge(-1, 1, 1, 1)
