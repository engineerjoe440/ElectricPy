import cmath
import math
import pytest

import electricpy.geometry.circle as circle_mod
from electricpy.geometry.circle import Circle, construct
from electricpy.geometry import Line, Point


class TestArea:
    def test_0(self):

        c = Circle((0, 0), 1)
        assert c.area() == cmath.pi

        c = Circle((0, 0), 2)
        assert c.area() == cmath.pi * 4

    def test_1(self):

        c = Circle((0, 0), 1.1)
        assert c.area() == cmath.pi * 1.1**2

        c = Circle((0, 0), 2.2)
        assert c.area() == cmath.pi * 2.2**2


class TestCircumference:
    def test_0(self):

        c = Circle((0, 0), 1)
        assert c.circumference() == cmath.pi * 2

        c = Circle((0, 0), 2)
        assert c.circumference() == cmath.pi * 4

    def test_1(self):

        c = Circle((0, 0), 1.1)
        assert c.circumference() == cmath.pi * 2.2

        c = Circle((0, 0), 2.2)
        assert c.circumference() == cmath.pi * 4.4


class TestTangent:
    def test_0(self):
        c = Circle((0, 0), 1)

        assert c.tangent(Point(0, 1)) == Line(0, 1, -1)
        assert c.tangent(Point(0, -1)) == Line(0, -1, -1)
        assert c.tangent(Point(1, 0)) == Line(1, 0, -1)
        assert c.tangent(Point(-1, 0)) == Line(-1, 0, -1)

    def test_1(self):

        from test import compare_lines

        c = Circle((0, 0), 1)

        p = Point(cmath.cos(cmath.pi / 4), cmath.sin(cmath.pi / 4))
        p1 = Point(cmath.sqrt(2), 0)
        p2 = Point(0, cmath.sqrt(2))

        assert compare_lines(c.tangent(p), Line.construct(p1, p2))


class TestNormal:
    def test_0(self):
        c = Circle((0, 0), 1)

        assert c.normal(Point(0, 1)) == Line(1, 0, 0)
        assert c.normal(Point(0, -1)) == Line(1, 0, 0)
        assert c.normal(Point(1, 0)) == Line(0, 1, 0)
        assert c.normal(Point(-1, 0)) == Line(0, 1, 0)

    def test_1(self):

        from test import compare_lines

        c = Circle((0, 0), 1)
        p0 = Point(cmath.cos(cmath.pi / 4), cmath.sin(cmath.pi / 4))
        p1 = Point(-cmath.cos(cmath.pi / 4), cmath.sin(cmath.pi / 4))

        assert compare_lines(c.normal(p0), Line(1, -1, 0))
        assert compare_lines(c.normal(p1), Line(1, 1, 0))


def test_contains_point_and_tangent_validation():
    c = Circle((0, 0), 2)
    assert c.contains_point(Point(2, 0))
    with pytest.raises(ValueError):
        c.tangent(Point(1, 1))
    assert c.tangent(Point(1, 1), require_on_circle=False) == Line(1, 1, -4)

    assert c.power(Point(0, 0)) == -4
    assert c.power(Point(2, 0)) == 0
    assert c.is_tangent(Line(1, 0, -2))
    assert not c.is_tangent(Line(1, 0, -3))
    assert c.is_normal(Line(1, 0, 0))
    assert not c.is_normal(Line(1, 0, -1))


def test_equation_and_parametric_errors():
    c = Circle((1, 2), 3)
    eq = c.equation()
    assert "x^2 + y^2" in eq
    assert " - 2*x" in eq
    assert " - 4*y" in eq
    assert " - 4" in eq
    assert c.radius == 3
    
    with pytest.raises(ValueError):
        list(c.parametric_equation(theta_resolution=0))
    with pytest.raises(ValueError):
        list(c.parametric_equation(theta_resolution=-0.1))

    points = list(c.parametric_equation(theta_resolution=math.pi))
    assert points[0] == (4.0, 2.0)
    assert points[-1][0] == pytest.approx(4.0)
    semi = list(c.parametric_equation(theta_resolution=math.pi / 2, semi=True))
    assert semi[0] == (4.0, 2.0)
    assert semi[-1][0] == pytest.approx(-2.0)

    eq_no_linear = Circle((0, 0), math.pi).equation()
    assert "x^2 + y^2" in eq_no_linear
    assert "*x" not in eq_no_linear
    assert "*y" not in eq_no_linear
    assert " - " in eq_no_linear

    eq_float = Circle((0.3, 0), 1).equation()
    assert " - 0.6*x" in eq_float


def test_sector_and_intersection_cases():
    c1 = Circle((0, 0), 1)
    c2 = Circle((2, 0), 1)
    assert c1.sector_length(cmath.pi) == cmath.pi
    assert c1.sector_area(cmath.pi) == cmath.pi / 2
    inter = c1.intersection(c2)
    assert isinstance(inter, Point)
    assert c1.intersetion(c2) == inter

    assert c1.intersection(Circle((0, 0), 1)) == "infinite"
    assert c1.intersection(Circle((0, 0), 2)) is None
    assert c1.intersection(Circle((5, 0), 1)) is None
    assert c1.intersection(Circle((0.5, 0), 5)) is None

    two = c1.intersection(Circle((1, 0), 1))
    assert isinstance(two, tuple)
    assert len(two) == 2
    assert c1.contains_point(two[0])
    assert c1.contains_point(two[1])

    with pytest.raises(TypeError):
        c1.intersection("not-a-circle")

    near_tangent = c1.intersection(Circle((2 + 1e-6, 0), 1), tol=1e-5)
    assert isinstance(near_tangent, Point)

    assert c1.intersection(Circle((2.5, 0), 1), tol=0.5) is None


def test_construct_circle():
    p0 = Point(1, 0)
    p1 = Point(0, 1)
    p2 = Point(-1, 0)
    circ = construct(p0, p1, p2)
    assert circ.contains_point(p1)

    with pytest.raises(AssertionError):
        construct(Point(0, 0), Point(1, 1), Point(2, 2))


def test_point_coercion_and_repr():
    pt = circle_mod._as_point((1, 2))
    assert isinstance(pt, Point)
    assert pt == Point(1, 2)
    pt = circle_mod._as_point([3, 4])
    assert pt == Point(3, 4)
    pt = circle_mod._as_point(Point(5, 6))
    assert pt == Point(5, 6)

    with pytest.raises(ValueError):
        circle_mod._as_point((1, 2, 3))
    with pytest.raises(TypeError):
        circle_mod._as_point("bad")

    assert circle_mod._is_close(1.0, 1.0 + 1e-10)
    assert not circle_mod._is_close(1.0, 1.1, rel_tol=1e-6, abs_tol=1e-6)

    circle = Circle((0, 0), 1)
    assert repr(circle) == "Circle(center=(0.0, 0.0), radius=1.0)"
    assert str(circle) == "Circle(center=(0.0, 0.0), radius=1.0)"
    assert circle == Circle((0, 0), 1)
    assert circle != Circle((0, 0), 2)
    assert circle != "not-a-circle"
    assert hash(circle) == hash((0.0, 0.0, 1.0))


def test_circle_init_validation_and_normal_error():
    with pytest.raises(TypeError):
        Circle((0, 0), "bad")
    with pytest.raises(ValueError):
        Circle((0, 0), -1)
    with pytest.raises(TypeError):
        Circle("bad", 1)

    circle = Circle((1, 1), 1)
    with pytest.raises(ValueError):
        circle.normal(Point(1, 1))
