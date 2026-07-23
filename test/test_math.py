import numpy as np
import pytest
from numpy.testing import assert_allclose

from electricpy import math as epmath


def test_gaussian_at_zero():
    """Validate gaussian at zero behavior."""
    expected = 1 / np.sqrt(2 * np.pi)
    assert_allclose(epmath.gaussian(0, mu=0, sigma=1), expected)


def test_gausdist_midpoint():
    """Validate gausdist midpoint behavior."""
    assert_allclose(epmath.gausdist(0, mu=0, sigma=1), 0.5, atol=1e-6)


def test_step_function():
    """Validate step function behavior."""
    data = np.array([-1.0, 0.0, 1.0])
    assert_allclose(epmath.step(data), np.array([0.0, 1.0, 1.0]))


def test_gaussian_sigma_zero():
    """Validate gaussian sigma zero behavior."""
    with pytest.raises(ValueError):
        epmath.gaussian(0, sigma=0)


def test_funcrms_validation():
    """Validate error handling for funcrms validation."""
    with pytest.raises(ValueError):
        epmath.funcrms(lambda x: x, T=0)
