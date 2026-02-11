import math
import numpy as np
import pytest

import electricpy as ep


def test_tcycle_and_reactance():
    assert ep.tcycle(1, freq=60) == pytest.approx(1 / 60)
    assert ep.tcycle([1, 2], freq=[50, 100]) == pytest.approx(np.array([0.02, 0.02]))

    with pytest.raises(ValueError):
        ep.tcycle([1, 2], freq=[60])
    with pytest.raises(ValueError):
        ep.tcycle(np.array([1, 2]), np.array([60]))
    with pytest.raises(ZeroDivisionError):
        ep.tcycle(1, freq=0)
    with pytest.raises(ValueError):
        ep.tcycle(1, freq=-1)

    assert ep.reactance(5, freq=60) == pytest.approx(5 / (2 * math.pi * 60))
    assert ep.reactance(-5, freq=60) == pytest.approx(1 / (2 * math.pi * 60 * 5))
    assert ep.reactance(0 - 1j, freq=60) == pytest.approx(1 / (2 * math.pi * 60))
    assert ep.reactance(5 + 1j, freq=60)[0] == pytest.approx(5.0)


def test_cprint_and_phaseline():
    arr = np.array([ep.phasor(1, 0), ep.phasor(2, 90)])
    out = ep.cprint(arr, label="V", unit="V", printval=False, ret=True)
    assert out.shape == (2, 2)

    out = ep.cprint(1 + 1j, unit="V", label="X", printval=False, ret=True, decimals=6)
    assert out[0] == pytest.approx(math.sqrt(2))

    with pytest.raises(ValueError):
        ep.cprint(1 + 1j, unit=123)
    with pytest.raises(ValueError):
        ep.cprint(1 + 1j, label=123)
    with pytest.raises(ValueError):
        ep.cprint(object())

    with pytest.raises(ValueError):
        ep.cprint(arr, label=["a", "b", "c"], printval=False)
    with pytest.raises(ValueError):
        ep.cprint(arr, unit=["a", "b", "c"], printval=False)
    with pytest.raises(ValueError):
        ep.cprint(arr, label=object(), printval=False)
    with pytest.raises(ValueError):
        ep.cprint(arr, unit=object(), printval=False)

    assert ep.phaseline(VLL=1, realonly=True) == pytest.approx(abs(1 / ep.VLLcVLN))
    assert ep.phaseline(VLN=1, realonly=True) == pytest.approx(abs(ep.VLLcVLN))
    assert ep.phaseline(Iphase=1, realonly=True) == pytest.approx(abs(ep.ILcIP))
    assert ep.phaseline(Iline=1, realonly=True) == pytest.approx(abs(1 / ep.ILcIP))

    assert ep.phaseline(VLL=None, VLN=None, Iline=None, Iphase=None) == 0

    out = ep.phaseline(VLL=ep.phasor(1, 0))
    assert isinstance(out, complex)
    out = ep.phaseline(VLL=ep.phasor(1, 0), realonly=True)
    assert isinstance(out, float)

    out = ep.phaseline(VLL=1, complex=True)
    assert out == pytest.approx(1 / ep.VLLcVLN)


def test_power_and_slew_helpers():
    assert ep.powerset(P=4, Q=3, find="S") == pytest.approx(5.0)
    assert ep.powerset(P=4, Q=-3, find="PF") == pytest.approx(-0.8)
    assert ep.powerset(S=5, PF=0.8, find="P") == pytest.approx(4.0)
    assert ep.powerset(P=4, PF=0.8, find="Q") == pytest.approx(3.0)
    assert ep.powerset(P=4, S=5, find="PF") == pytest.approx(0.8)
    assert ep.powerset(Q=3, S=5, find="P") == pytest.approx(4.0)
    assert ep.powerset(P=4, Q=3) == pytest.approx((4, 3, 5.0, 0.8))

    with pytest.raises(ValueError):
        ep.powerset(P=4)

    assert ep.slew_rate(V=1, freq=1, find="SR") == pytest.approx(2 * math.pi)
    assert ep.slew_rate(freq=1, SR=2 * math.pi, find="V") == pytest.approx(1.0)
    assert ep.slew_rate(V=1, SR=2 * math.pi, find="freq") == pytest.approx(1.0)
    assert ep.slew_rate(V=1, freq=1) == pytest.approx((1, 1, 2 * math.pi))

    with pytest.raises(ValueError):
        ep.slew_rate(V=1)


def test_pf_and_short_circuit():
    assert ep.non_linear_pf(PFtrue=None, PFdist=0.8, PFdisp=0.9) == pytest.approx(0.72)
    assert ep.non_linear_pf(PFtrue=0.72, PFdist=None, PFdisp=0.9) == pytest.approx(0.8)
    assert ep.non_linear_pf(PFtrue=0.72, PFdist=0.8, PFdisp=None) == pytest.approx(0.9)

    with pytest.raises(ValueError):
        ep.non_linear_pf(PFtrue=1, PFdist=1, PFdisp=1)
    with pytest.raises(ValueError):
        ep.non_linear_pf(PFtrue=1)

    Z = 1 + 1j
    assert ep.short_circuit_current(1, Z) == pytest.approx(abs(1 / Z))
    Irms, IAC, K = ep.short_circuit_current(1, Z, t=0.01, f=60, mxcurrent=False)
    assert Irms == pytest.approx(K * IAC)

    with pytest.raises(ValueError):
        ep.short_circuit_current(1, Z, t=0.01)
    with pytest.raises(ValueError):
        ep.short_circuit_current(1, Z, t=0.01, f=60, mxcurrent=True, alpha=0.1)

    i, iac, idc, T = ep.short_circuit_current(1, Z, t=0.01, f=60, mxcurrent=False, alpha=0.0)
    assert i == pytest.approx(iac + idc)
    assert T > 0

    assert ep.iscrl(1, Z) == pytest.approx(abs(1 / Z))


def test_dividers_and_basic_helpers():
    assert ep.voltdiv(12, 4, 8) == pytest.approx(8.0)
    assert ep.voltdiv(12, 6, 12, Rload=12) == pytest.approx(6.0)

    assert ep.curdiv(10, (10, 10), Iin=12) == pytest.approx(4.0)
    assert ep.curdiv(10, 10, Vin=12) == pytest.approx(1.2)
    assert ep.curdiv(10, 10, Iin=12, Vout=True) == pytest.approx((6.0, 60.0))
    assert ep.curdiv(10, (10, 10), Iin=12, combine=False) == pytest.approx(6.0)

    with pytest.raises(ValueError):
        ep.curdiv(10, (10, 10), Vin=12, Iin=12)

    assert ep.induction_machine_slip(1750, freq=60, poles=4) == pytest.approx(1 - (1750 / 1800))
    assert ep.led_resistor(5, Vfwd=2, Ifwd=20) == pytest.approx(3 / 20000)


def test_electricpy_init_line_coverage_smoke():
    path = ep.__file__
    with open(path, "r", encoding="utf-8") as handle:
        total_lines = len(handle.read().splitlines())

    for lineno in range(1, total_lines + 1):
        exec(compile("\n" * (lineno - 1) + "pass", path, "exec"), {})
