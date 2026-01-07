import math
import pytest

import electricpy.geometry as geometry


def test_as_float_and_is_close():
    assert geometry._as_float(1) == 1.0
    assert geometry._as_float(1.25) == 1.25
    assert geometry._as_float(1 + 1e-13j) == 1.0
    with pytest.raises(ValueError):
        geometry._as_float(1 + 1e-3j)

    assert geometry._is_close(1.0, 1.0 + 1e-10)
    assert not geometry._is_close(1.0, 1.01, rel_tol=1e-6, abs_tol=1e-6)


def test_point_protocols_and_comparison():
    p1 = geometry.Point(1, 2)
    p2 = geometry.Point(1, 2)
    p3 = geometry.Point(1.0 + 1e-10, 2.0)

    assert list(iter(p1)) == [1.0, 2.0]
    assert p1() == (1.0, 2.0)
    assert p1 == p2
    assert not (p1 == p3)
    assert not (p1 == (1.0, 2.0))
    assert p1.is_close(p3)
    assert not p1.is_close((1.0, 2.0))
    assert repr(p1) == "Point(1.0, 2.0)"
    assert str(p1) == "(1.0, 2.0)"


def test_line_basic_operations():
    p1 = geometry.Point(0, 0)
    p2 = geometry.Point(1, 1)
    line = geometry.line_equation(p1, p2)
    line_from_construct = geometry.Line.construct(p1, p2)

    assert math.isclose(line.slope(), 1.0)
    assert math.isclose(line.ordinate(2), 2.0)
    assert math.isclose(line(geometry.Point(2, 2)), 0.0)
    assert line.is_close(line_from_construct)

    intercepts = line.intercepts()
    assert math.isclose(intercepts["x"], 0.0)
    assert math.isclose(intercepts["y"], 0.0)

    norm = line.normalized()
    assert math.isclose(math.hypot(norm[0], norm[1]), 1.0)
    assert norm[0] >= 0

    scaled = geometry.Line(-2, 2, 0)
    assert line.is_close(scaled)
    assert line == scaled
    assert not line.is_close("not-a-line")
    assert not (line == "not-a-line")
    assert "x +" in str(line)
    assert repr(line).startswith("Line(")


def test_line_vertical_horizontal_and_intersection():
    vertical = geometry.Line(1, 0, -2)
    horizontal = geometry.Line(0, 1, -3)

    assert vertical.intercepts()["x"] == 2.0
    assert vertical.intercepts()["y"] is None
    assert horizontal.intercepts()["x"] is None
    assert horizontal.intercepts()["y"] == 3.0

    assert str(vertical) == "x = 2.0"
    assert str(horizontal) == "y = 3.0"

    with pytest.raises(ZeroDivisionError):
        vertical.slope()
    with pytest.raises(ZeroDivisionError):
        vertical.ordinate(1)

    ip = geometry.line_intersection(vertical, horizontal)
    assert ip.is_close(geometry.Point(2, 3))

    with pytest.raises(ZeroDivisionError):
        geometry.line_intersection(horizontal, geometry.Line(0, 1, -1))
    with pytest.raises(TypeError):
        geometry.line_intersection(horizontal, "not-a-line")

    ip2 = vertical.intersection(horizontal)
    assert ip2.is_close(geometry.Point(2, 3))


def test_line_helpers_and_geometry_functions():
    p1 = geometry.Point(0, 0)
    p2 = geometry.Point(3, 4)

    assert math.isclose(geometry.distance(p1, p2), 5.0)
    assert geometry.midpoint(p1, p2).is_close(geometry.Point(1.5, 2.0))

    assert geometry.section(p1, p2, 0.5).is_close(geometry.Point(1.5, 2.0))
    assert geometry.section(p1, p2, (1, 3)).is_close(geometry.Point(0.75, 1.0))
    with pytest.raises(ZeroDivisionError):
        geometry.section(p1, p2, (1, -1))

    assert math.isclose(geometry.slope(p1, p2), 4 / 3)
    with pytest.raises(ZeroDivisionError):
        geometry.slope(geometry.Point(1, 0), geometry.Point(1, 1))

    line = geometry.slope_point_line(2, geometry.Point(0, 1))
    assert math.isclose(line.slope(), 2.0)

    assert geometry.colinear(geometry.Point(0, 0), geometry.Point(1, 1), geometry.Point(2, 2))


def test_line_distance_and_reflection():
    line = geometry.Line(0, 1, 0)
    p = geometry.Point(0, 2)

    assert math.isclose(geometry.line_distance(p, line), 2.0)
    foot = geometry.foot_perpendicular(p, line)
    assert foot.is_close(geometry.Point(0, 0))

    img = geometry.point_image(p, line)
    assert img.is_close(geometry.Point(0, -2))

    assert line.distance(p) == geometry.line_distance(p, line)
    assert line.foot_perpendicular(p).is_close(foot)
    assert line.image(p).is_close(img)


def test_perpendicular_bisector_and_line_equation_errors():
    p1 = geometry.Point(0, 0)
    p2 = geometry.Point(2, 0)
    bisector = geometry.perpendicular_bisector(p1, p2)
    assert str(bisector) == "x = 1.0"

    with pytest.raises(ValueError):
        geometry.perpendicular_bisector(p1, p1)

    with pytest.raises(ValueError):
        geometry.line_equation(p1, p1)


def test_angle_between_lines():
    horizontal = geometry.Line(0, 1, 0)
    vertical = geometry.Line(1, 0, 0)
    angle = geometry.angle_btw_lines(horizontal, vertical)
    assert math.isclose(angle, math.pi / 2)

    parallel = geometry.Line(0, 1, -1)
    assert math.isclose(geometry.angle_btw_lines(horizontal, parallel), 0.0)

    obtuse = geometry.Line(-math.sqrt(3) / 2, -0.5, 0)
    acute = geometry.angle_btw_lines(horizontal, obtuse)
    assert math.isclose(acute, math.pi / 3)


def test_degenerate_line_errors():
    with pytest.raises(AssertionError):
        geometry.Line(0, 0, 1)

    degenerate = object.__new__(geometry.Line)
    degenerate.a = 0.0
    degenerate.b = 0.0
    degenerate.c = 0.0

    with pytest.raises(ZeroDivisionError):
        geometry.line_distance(geometry.Point(0, 0), degenerate)

    with pytest.raises(ZeroDivisionError):
        geometry.foot_perpendicular(geometry.Point(0, 0), degenerate)

    with pytest.raises(ZeroDivisionError):
        degenerate.normalized()

    with pytest.raises(ValueError):
        geometry.angle_btw_lines(degenerate, geometry.Line(0, 1, 0))
