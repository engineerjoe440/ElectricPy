from Geometry import Point
import numpy as np
from numpy.testing import assert_almost_equal

def compare_points(p1: Point, p2: Point) -> bool:
    try:
        assert_almost_equal(p1.x, p2.x)
        assert_almost_equal(p1.y, p2.y)
    except AssertionError:
        return False
    return True
    