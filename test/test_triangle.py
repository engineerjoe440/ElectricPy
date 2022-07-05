import cmath
from electricpy.geometry import triangle
from electricpy.geometry import Point
from test import compare_points

class TestCentroid():

    def test_0(self):
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert t.centroid() == Point(1/3, 1/3)

    def test_1(self):
        p1 = Point(1.1, 2.2)
        p2 = Point(3.1, 4.2)
        p3 = Point(5.1, 6.7)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.centroid(), Point(3.1, 131/30))


class TestInCenter():

    def test_0(self):
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.in_center(), Point(1/(2 + cmath.sqrt(2)), 1/(2 + cmath.sqrt(2))))

    def test_1(self):
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.in_center(), Point(0.5, cmath.sqrt(3)/6))

class TestOrthoCenter():

    def test_0(self):
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.ortho_center(), Point(0, 0))

    def test_1(self):
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.ortho_center(), Point(0.5, cmath.sqrt(3)/6))


class TestCircumCenter():
    def test_0(self):
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.circum_center(), Point(0.5, 0.5))

    def test_1(self):
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.circum_center(), Point(0.5, cmath.sqrt(3)/6))