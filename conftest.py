# Pytest configuration

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Use Legacy Numpy Printing Options for consistent test outputs."""
    np.set_printoptions(legacy='1.25')
