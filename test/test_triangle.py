import cmath
import math
import pytest

from electricpy.geometry import triangle
from electricpy.geometry import Point
from test import compare_points

class TestCentroid():

    def test_0(self):
        """Validate centroid scenario 0."""
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert t.centroid() == Point(1/3, 1/3)

    def test_1(self):
        """Validate centroid scenario 1."""
        p1 = Point(1.1, 2.2)
        p2 = Point(3.1, 4.2)
        p3 = Point(5.1, 6.7)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.centroid(), Point(3.1, 131/30))


class TestInCenter():

    def test_0(self):
        """Validate in center scenario 0."""
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.in_center(), Point(1/(2 + cmath.sqrt(2)), 1/(2 + cmath.sqrt(2))))

    def test_1(self):
        """Validate in center scenario 1."""
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.in_center(), Point(0.5, cmath.sqrt(3)/6))

class TestOrthoCenter():

    def test_0(self):
        """Validate ortho center scenario 0."""
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.ortho_center(), Point(0, 0))

    def test_1(self):
        """Validate ortho center scenario 1."""
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.ortho_center(), Point(0.5, cmath.sqrt(3)/6))


class TestCircumCenter():
    def test_0(self):
        """Validate circum center scenario 0."""
        p1 = Point(0, 1)
        p2 = Point(1, 0)
        p3 = Point(0, 0)
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.circum_center(), Point(0.5, 0.5))

    def test_1(self):
        """Validate circum center scenario 1."""
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(1*cmath.cos(cmath.pi/3), 1*cmath.sin(cmath.pi/3))
        t = triangle.Triangle(p1, p2, p3)
        assert compare_points(t.circum_center(), Point(0.5, cmath.sqrt(3)/6))


def test_triangle_perimeter_and_area():
    """Validate triangle perimeter and area behavior."""
    p1 = Point(0, 0)
    p2 = Point(3, 0)
    p3 = Point(0, 4)
    t = triangle.Triangle(p1, p2, p3)
    assert t.perimeter() == 12
    assert t.perimeters() == 12
    assert t.area() == 6


def test_triangle_radii():
    """Validate triangle radii behavior."""
    p1 = Point(0, 0)
    p2 = Point(3, 0)
    p3 = Point(0, 4)
    t = triangle.Triangle(p1, p2, p3)
    assert t.in_radius() == 1
    assert t.circum_radius() == 2.5


def test_triangle_invalid_points():
    """Validate error handling for triangle invalid points."""
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p3 = Point(2, 2)
    with pytest.raises(ValueError):
        triangle.Triangle(p1, p2, p3)


def test_triangle_init_validation_and_helpers():
    """Validate error handling for triangle init validation and helpers."""
    with pytest.raises(ValueError):
        triangle.Triangle(Point(0, 0), Point(1, 0))

    with pytest.raises(TypeError):
        triangle.Triangle(Point(0, 0), (1, 0), Point(0, 1))

    with pytest.raises(ValueError):
        triangle.Triangle(Point(0, 0), Point(0, 0), Point(1, 1))

    assert triangle._as_float(1) == 1.0
    assert triangle._as_float(1 + 1e-13j) == 1.0
    with pytest.raises(ValueError):
        triangle._as_float(1 + 1e-3j)

    assert triangle._is_close(1.0, 1.0 + 1e-10)
    assert not triangle._is_close(1.0, 1.01, rel_tol=1e-6, abs_tol=1e-6)

    p0 = Point(0, 0)
    p1 = Point(2, 0)
    p2 = Point(0, 2)
    assert triangle._triangle_twice_area(p0, p1, p2) == 4.0


def test_triangle_area_and_centers_errors():
    """Validate error handling for triangle area and centers errors."""
    tri = object.__new__(triangle.Triangle)
    tri.points = (Point(0, 0), Point(1, 0), Point(0, 1))
    tri._tol = 1e-12
    tri.a = 0.0
    tri.b = 0.0
    tri.c = 0.0

    with pytest.raises(ValueError):
        tri.in_center()

    with pytest.raises(ValueError):
        tri.in_radius()

    tri2 = object.__new__(triangle.Triangle)
    tri2.points = (Point(0, 0), Point(1, 0), Point(0, 1))
    tri2._tol = 1e-12
    tri2.a = 1.0
    tri2.b = 1.0
    tri2.c = 3.0

    with pytest.raises(ValueError):
        tri2.area()

    tri3 = object.__new__(triangle.Triangle)
    tri3.points = (Point(0, 0), Point(1, 0), Point(0, 1))
    tri3._tol = 1e-12
    tri3.a = 1.0
    tri3.b = 1.0
    tri3.c = 2.0

    with pytest.raises(ValueError):
        tri3.circum_radius()

    tri4 = object.__new__(triangle.Triangle)
    tri4.points = (Point(0, 0), Point(1, 0), Point(0, 1))
    tri4._tol = 1e-12
    tri4.a = 1.0
    tri4.b = 1.0
    tri4.c = 2.0 + 1e-13
    assert tri4.area() == 0.0


def test_triangle_validation_helper():
    """Validate error handling for triangle validation helper."""
    tri = object.__new__(triangle.Triangle)
    tri.points = (Point(0, 0), Point(1, 0), Point(2, 0))
    tri._tol = 1e-12
    tri.a = 1.0
    tri.b = 1.0
    tri.c = 2.1
    assert not tri._Triangle__is_valid()

    tri.c = 2.0
    assert not tri._Triangle__is_valid()

    tri.a = 1.0
    tri.b = 3.0
    tri.c = 1.0
    assert not tri._Triangle__is_valid()

    tri.a = 3.0
    tri.b = 1.0
    tri.c = 1.0
    assert not tri._Triangle__is_valid()

    tri.a = 1.0
    tri.b = 1.0
    tri.c = 1.0
    assert not tri._Triangle__is_valid()

    tri.points = (Point(0, 0), Point(1, 0), Point(0, 1))
    tri.a = math.sqrt(2)
    tri.b = 1.0
    tri.c = 1.0
    assert tri._Triangle__is_valid()
