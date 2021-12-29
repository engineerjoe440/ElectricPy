from Geometry import Point
from Geometry import Line
import numpy as np
from numpy.testing import assert_almost_equal

def compare_points(p1: Point, p2: Point) -> bool:
    try:
        assert_almost_equal(p1.x, p2.x)
        assert_almost_equal(p1.y, p2.y)
    except AssertionError:
        return False
    return True

def compare_lines(l1: Line, l2: Line) -> bool:
    try:
        assert_almost_equal(l1.a / l2.a , l1.b / l2.b)
        assert_almost_equal(l1.b / l2.b , l1.c / l2.c)
        assert_almost_equal(l1.c / l2.c , l1.a / l2.a)
    except AssertionError:
        return False
    return True
    