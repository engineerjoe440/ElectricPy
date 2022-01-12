# Import ElectricPy modules just to make sure they load correctly

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

# Test importing the `visu` module
def test_import_visu():
    try:
        from electricpy import visu
        assert True
    except:
        assert False

# Testing Imports of geometry submodule

# Testing geometry import
def test_Geometry():
    try:
        from electricpy import geometry
        assert True
    except ImportError:
        assert False

# Testing circle import from electricpy.geometry
def test_circle():
    try:
        from electricpy.geometry.circle import Circle
        assert True
    except ImportError:
        assert False

# Testing triangle import from electricpy.geometry
def test_triangle():
    try:
        from electricpy.geometry.triangle import Triangle
        assert True
    except ImportError:
        assert False

