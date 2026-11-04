import pytest
from electricpy import opamp


def test_non_inverting():
    """Validate non_inverting op-amp functionality"""

    assert opamp.non_inverting(Vin=None, Vout=10, Rg=5, Rf=15) == pytest.approx(
        2.5, abs=0.1
    )
    assert opamp.non_inverting(Vin=2.5, Vout=None, Rg=5, Rf=15) == pytest.approx(
        10, abs=0.1
    )
    assert opamp.non_inverting(Vin=2.5, Vout=10, Rg=None, Rf=15) == pytest.approx(
        5, abs=0.1
    )
    assert opamp.non_inverting(Vin=2.5, Vout=10, Rg=5, Rf=None) == pytest.approx(
        15, abs=0.1
    )


def test_inverting():
    """Validate inverting op-amp functionality"""

    assert opamp.inverting(Vin=None, Vout=10, Rin=5, Rf=15) == pytest.approx(
        -3.3333, abs=0.1
    )
    assert opamp.inverting(Vin=-3.33, Vout=None, Rin=5, Rf=15) == pytest.approx(9.99)
    assert opamp.inverting(Vin=-3.33, Vout=10, Rin=None, Rf=15) == pytest.approx(
        5, abs=0.1
    )
    assert opamp.inverting(Vin=-3.33, Vout=10, Rin=5, Rf=None) == pytest.approx(
        15, abs=0.1
    )


def test_differential():
    """Validate differential op-amp functionality"""

    assert opamp.differential(
        V1=None, V2=1, Vout=-10, R1=1, R2=1, R3=10, R4=10
    ) == pytest.approx(2.0, abs=0.1)
    assert opamp.differential(
        V1=2, V2=None, Vout=-10, R1=1, R2=1, R3=10, R4=10
    ) == pytest.approx(1.0, abs=0.1)
    assert opamp.differential(
        V1=2, V2=1, Vout=None, R1=1, R2=1, R3=10, R4=10
    ) == pytest.approx(-10.0, abs=0.1)
    assert opamp.differential(
        V1=2, V2=1, Vout=-10, R1=None, R2=1, R3=10, R4=10
    ) == pytest.approx(1.0, abs=0.1)
    assert opamp.differential(
        V1=2, V2=1, Vout=-10, R1=1, R2=None, R3=10, R4=10
    ) == pytest.approx(1.0, abs=0.1)
    assert opamp.differential(
        V1=2, V2=1, Vout=-10, R1=1, R2=1, R3=None, R4=10
    ) == pytest.approx(10.0, abs=0.1)
    assert opamp.differential(
        V1=2, V2=1, Vout=-10, R1=1, R2=1, R3=10, R4=None
    ) == pytest.approx(10.0, abs=0.1)


def test_inverting_summing():
    """Validate inverting_summing op-amp functionality"""

    assert opamp.inverting_summing(
        None, 10, 3, 10, 1, 10, Vout=-6, Rf=10
    ) == pytest.approx(2.0, abs=0.1)
    assert opamp.inverting_summing(
        2, None, 3, 10, 1, 10, Vout=-6, Rf=10
    ) == pytest.approx(10.0, abs=0.1)
    assert opamp.inverting_summing(
        2, 10, None, 10, 1, 10, Vout=-6, Rf=10
    ) == pytest.approx(3.0, abs=0.1)
    assert opamp.inverting_summing(
        2, 10, 3, None, 1, 10, Vout=-6, Rf=10
    ) == pytest.approx(10.0, abs=0.1)
    assert opamp.inverting_summing(
        2, 10, 3, 10, None, 10, Vout=-6, Rf=10
    ) == pytest.approx(1.0, abs=0.1)
    assert opamp.inverting_summing(
        2, 10, 3, 10, 1, None, Vout=-6, Rf=10
    ) == pytest.approx(10.0, abs=0.1)
    assert opamp.inverting_summing(
        2, 10, 3, 10, 1, 10, Vout=None, Rf=10
    ) == pytest.approx(-6.0, abs=0.1)
    assert opamp.inverting_summing(
        2, 10, 3, 10, 1, 10, Vout=-6, Rf=None
    ) == pytest.approx(10.0, abs=0.1)
