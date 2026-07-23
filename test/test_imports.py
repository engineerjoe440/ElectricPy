# Import ElectricPy modules just to make sure they load correctly

import importlib

# Test importing the package itself
def test_import_by_name():
    """Validate import behavior for import by name."""
    assert importlib.import_module("electricpy") is not None

# Test importing the `bode` module
def test_import_bode():
    """Validate import behavior for import bode."""
    assert importlib.import_module("electricpy.bode") is not None

# Test importing the `constants` module
def test_import_constants():
    """Validate import behavior for import constants."""
    assert importlib.import_module("electricpy.constants") is not None

# Test importing the `fault` module
def test_import_fault():
    """Validate import behavior for import fault."""
    assert importlib.import_module("electricpy.fault") is not None

# Test importing the `sim` module
def test_import_sim():
    """Validate import behavior for import sim."""
    assert importlib.import_module("electricpy.sim") is not None

# Test importing the `visu` module
def test_import_visu():
    """Validate import behavior for import visu."""
    assert importlib.import_module("electricpy.visu") is not None

# Testing Imports of geometry submodule

# Testing geometry import
def test_Geometry():
    """Validate geometry behavior."""
    assert importlib.import_module("electricpy.geometry") is not None

# Testing circle import from electricpy.geometry
def test_circle():
    """Validate circle behavior."""
    module = importlib.import_module("electricpy.geometry.circle")
    assert module.Circle is not None

# Testing triangle import from electricpy.geometry
def test_triangle():
    """Validate triangle behavior."""
    module = importlib.import_module("electricpy.geometry.triangle")
    assert module.Triangle is not None
