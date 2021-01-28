# Import ElectricPy modules just to make sure they load correctly

import os
import sys
import pytest

# Test importing the package itself
def test_import_by_name():
    try:
        import electricpy
        assert True
    except:
        assert False

# Test importing the `bode` module
def test_import_bode():
    try:
        from electricpy import bode
        assert True
    except:
        assert False

# Test importing the `constants` module
def test_import_constants():
    try:
        from electricpy import constants
        assert True
    except:
        assert False

# Test importing the `fault` module
def test_import_fault():
    try:
        from electricpy import fault
        assert True
    except:
        assert False

# Test importing the `sim` module
def test_import_sim():
    try:
        from electricpy import sim
        assert True
    except:
        assert False
