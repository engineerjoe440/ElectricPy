import electricpy.thermal as thermal
import pytest


def test_rtdtemp_allows_no_rounding():
    """RTD helper should support returning an unrounded float."""
    temp = thermal.rtdtemp(100.0, rtdtype="PT100", round=None)
    assert isinstance(temp, float)


def test_coldjunction_invalid_coupletype_raises() -> None:
    """Unknown thermocouple types should be rejected."""
    with pytest.raises(ValueError):
        thermal.coldjunction(25.0, coupletype="X")


def test_coldjunction_type_b_has_stricter_temperature_range() -> None:
    """Type B enforces 0C lower bound while type K allows subzero values."""
    with pytest.raises(ValueError):
        thermal.coldjunction(-1.0, coupletype="B")
    assert isinstance(thermal.coldjunction(-1.0, coupletype="K"), float)


def test_thermocouple_invalid_coupletype_raises() -> None:
    """Unknown thermocouple types should be rejected in temperature solve."""
    with pytest.raises(ValueError):
        thermal.thermocouple(0.0, coupletype="X")


def test_thermocouple_voltage_above_upper_bound_raises() -> None:
    """Clearly out-of-range voltages should fail fast."""
    with pytest.raises(ValueError):
        thermal.thermocouple(1.0, coupletype="K")


def test_thermocouple_fahrenheit_conversion_matches_celsius() -> None:
    """Fahrenheit option should apply the standard linear conversion."""
    celsius = thermal.thermocouple(0.0, coupletype="K", round=None)
    fahrenheit = thermal.thermocouple(0.0, coupletype="K", fahrenheit=True, round=None)
    assert fahrenheit == pytest.approx((celsius * 9.0 / 5.0) + 32.0, rel=1e-12)
