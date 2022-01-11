from electricpy.geometry import Point
from electricpy.geometry import Line
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
    except ZeroDivisionError:
        if l1.a == 0 and l2.a == 0:
            try:
                assert_almost_equal(l1.c / l2.c , l1.b / l2.b)
            except ZeroDivisionError:
                return False
            except AssertionError:
                return False
        if l1.b == 0 and l2.b == 0:
            try:
                assert_almost_equal(l1.c / l2.c , l1.a / l2.a)
            except ZeroDivisionError:
                return False
            except AssertionError:
                return False
        if l1.c == 0 and l2.c == 0:
            try:
                assert_almost_equal(l1.a / l2.a , l1.b / l2.b)
            except ZeroDivisionError:
                return False
            except AssertionError:
                return False
    finally:
        return True
    