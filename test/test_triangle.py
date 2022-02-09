from electricpy.geometry import triangle
from electricpy.geometry import Point
from test import compare_points
import cmath
from numpy.testing import assert_almost_equal

def test_centroid():

    def test_0():
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert t.centroid() == Point(1/3, 1/3)

    def test_1():
        p1 = Point(1.1, 2.2)
        p2 = Point(3.1, 4.2)
        p3 = Point(5.1, 6.7)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.centroid(), Point(3.1, 131/30))

    for i in range(2):
        exec("test_{}()".format(i))


def test_in_center():

    def test_0():
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.in_center(), Point(1/(2 + cmath.sqrt(2)), 1/(2 + cmath.sqrt(2))))

    def test_1():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.in_center(), Point(0.5, cmath.sqrt(3)/6))

    for i in range(2):
        exec("test_{}()".format(i))

def test_ortho_center():
    
    def test_0():
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.ortho_center(), Point(0, 0))

    def test_1():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.ortho_center(), Point(0.5, cmath.sqrt(3)/6))

    for i in range(2):
        exec("test_{}()".format(i))

def test_circum_center(): 
    def test_0():
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.circum_center(), Point(0.5, 0.5))

    def test_1():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.circum_center(), Point(0.5, cmath.sqrt(3)/6))

    for i in range(2):
        exec("test_{}()".format(i))

def test_area():
    def test_0():
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert_almost_equal(t.area(), 0.5, decimal=3)

    def test_1():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert_almost_equal(t.area(), cmath.sqrt(3)/4)

    for i in range(2):
        exec("test_{}()".format(i))

def test_in_radius():
    def test_0():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)

        assert_almost_equal(t.in_radius(), 0.5/cmath.sqrt(3))

    def test_1():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(0, 1)
        t = triangle.Triangle(p1, p2, p3)

        assert_almost_equal(t.in_radius(), 1/(2 + cmath.sqrt(2)))

    for i in range(2):
        exec("test_{}()".format(i))

def test_circum_radius():
    def test_0():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(0, 1)
        t = triangle.Triangle(p1, p2, p3)

        assert_almost_equal(t.circum_radius(), 1/(2 ** 0.5))

    def test_1():
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)

        assert_almost_equal(t.circum_radius(), 1/cmath.sqrt(3))

    for i in range(2):
        exec("test_{}()".format(i))
