# Import SELProtoPy Just to Verify No Major Issues

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.getcwd(), 'electricpy'))

def test_import_by_name():
    try:
        import electricpy
        assert True
    except:
        assert False

def test_import_bode():
    try:
        from electricpy import bode
        assert True
    except:
        assert False

def test_import_constants():
    try:
        from electricpy import constants
        assert True
    except:
        assert False

def test_import_fault():
    try:
        from electricpy import fault
        assert True
    except:
        assert False

def test_import_sim():
    try:
        from electricpy import sim
        assert True
    except:
        assert False