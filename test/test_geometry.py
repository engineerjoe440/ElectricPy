import cmath
from electricpy import geometry as Geometry
from electricpy.geometry import Point
from electricpy.geometry import Line
from numpy.testing import assert_array_almost_equal

class TestDistance():

    def test_0(self):
        """Validate distance scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        assert Geometry.distance(p1, p2) == 2*(2**0.5)

        p1 = Point(4, -6)
        p2 = Point(-2, -5)
        assert Geometry.distance(p2, p1) ==  (37**0.5)

        p1 = Point(1.3, 2.3)
        p2 = Point(1.4, 2.4)

        d_output = Geometry.distance(p1, p2)
        d_actual = 0.1*(2**0.5)

        assert_array_almost_equal(d_output, d_actual, decimal=6)

    def test_1(self):
        """Validate distance scenario 1."""
        p1 = Point(1, 2)
        p2 = Point(1, 3)
        assert Geometry.distance(p1, p2) == 1

        p1 = Point(2.0, 1)
        p2 = Point(3.0, 1)
        assert Geometry.distance(p1, p2) == 1

class Testslope():

    def test_0(self):
        """Validate slope scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        assert Geometry.slope(p1, p2) == 1

        p1 = Point(4, -6)
        p2 = Point(-2, -5)
        assert Geometry.slope(p2, p1) == -1/6

    def test_1(self):
        """Validate slope scenario 1."""

        p1 = Point(1, 2)
        p2 = Point(2, 2)

        assert Geometry.slope(p1, p2) == 0

        p1 = Point(1, 2)
        p2 = Point(1, 3)
        try:
            Geometry.slope(p1, p2)
        except ZeroDivisionError:
            assert True

class Testsection():

    def test_0(self):
        """Validate section scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)

        p = Geometry.section(p1, p2, 0.5)
        assert p == Point(2, 3)

    def test_1(self):
        """Validate section scenario 1."""
        p1 = Point(-1, 3)
        p2 = Point(1, -3)

        p_computed = Geometry.section(p1, p2, (2, 3))
        p_actual = Point(-1/5, 3/5)

        assert_array_almost_equal(p_computed(), p_actual(), decimal=6)

class Testline_equaltion():

    def test_0(self):
        """Validate line equaltion scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        assert Geometry.line_equation(p1, p2) == Line(1, -1, 1)

        p1 = Point(4, -6)
        p2 = Point(-2, -5)
        assert Geometry.line_equation(p1, p2) == Line(1, 6, 32)

    def test_1(self):
        """Validate line equaltion scenario 1."""
        p1 = Point(1, 2)
        p2 = Point(1, 3)
        assert Geometry.line_equation(p1, p2) == Line(1, 0, -1)

        p1 = Point(1, 2)
        p2 = Point(2, 2)
        assert Geometry.line_equation(p1, p2) == Line(0, 1, -2)

    def test_2(self):
        """Validate line equaltion scenario 2."""
        assert Line(1, 2, 3) == Line(2, 4, 6)
        assert Line(1, -1, 0) == Line(3, -3, 0)
        assert Line(1, 0, -1) == Line(3, 0, -3)

class Testline_distance():

    def test_0(self):
        """Validate line distance scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        line = Line.construct(p1, p2)
        assert Geometry.line_distance(p1, line) == 0
        assert Geometry.line_distance(p2, line) ==  0

    def test_1(self):
        """Validate line distance scenario 1."""
        p1 = Point(2, 0)
        p2 = Point(2, 4)
        p = Point(0, 0)
        line = Line.construct(p1, p2)
        assert Geometry.line_distance(p, line) == 2
        assert line.distance(p) == 2

        line = Line(0, 1, -3)
        assert line.distance(p) == 3

class Testfoot_perpendicular():

    def test_0(self):
        """Validate foot perpendicular scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        line = Line.construct(p1, p2)
        p = Point(2, 2)
        assert Geometry.foot_perpendicular(p, line) == Point(1.5, 2.5)

        p = Point(2, 3)
        assert Geometry.foot_perpendicular(p, line) == Point(2, 3)

    def test_1(self):
        """Validate foot perpendicular scenario 1."""
        p = Point(-1, 3)
        line = Line(3, -4, -16)

        p_actual = line.foot_perpendicular(p)
        p_image = line.image(p)

        p_desired = Point(68/25, -49/25)

        assert_array_almost_equal(p_actual(), p_desired(), decimal=6)
        assert Geometry.midpoint(p, p_image) == p_actual

class TestPerpendicularBisector():

    def test_0(self):
        """Validate perpendicular bisector scenario 0."""
        p1 = Point(3, 0)
        p2 = Point(0, 3)
        line = Geometry.perpendicular_bisector(p1, p2)
        assert line == Line(1, -1, 0)

    def test_1(self):
        """Validate perpendicular bisector scenario 1."""
        p1 = Point(-3, 0)
        p2 = Point(0, 3)
        line = Geometry.perpendicular_bisector(p1, p2)
        assert line == Line(1, 1, 0)

    def test_2(self):
        """Validate perpendicular bisector scenario 2."""
        p1 = Point(3, 0)
        p2 = Point(5, 0)
        line = Geometry.perpendicular_bisector(p1, p2)
        assert line == Line(1, 0, -4)

    def test_3(self):
        """Validate perpendicular bisector scenario 3."""
        p1 = Point(0, 3)
        p2 = Point(0, 5)
        line = Geometry.perpendicular_bisector(p1, p2)
        assert line == Line(0, 1, -4)

class Testcolinear():

    def test_0(self):
        """Validate colinear scenario 0."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        p3 = Point(5, 6)
        assert Geometry.colinear(p1, p2, p3)

    def test_1(self):
        """Validate colinear scenario 1."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        p3 = Point(5, 7)
        assert not Geometry.colinear(p1, p2, p3)

    def test_2(self):
        """Validate colinear scenario 2."""
        p1 = Point(1, 0)
        p2 = Point(2, 0)
        p3 = Point(3, 0)
        assert Geometry.colinear(p1, p2, p3)

class TestAngleBtwLines():

    def test_0(self):
        """Validate angle btw lines scenario 0."""
        l1 = Line(3, 4, 7)
        l2 = Line(4, -3, 5)

        assert cmath.pi/2 == Geometry.angle_btw_lines(l1, l2)

    def test_1(self):
        """Validate angle btw lines scenario 1."""
        l1 = Line(3, 0, 0)
        l2 = Line(4, 0, 0)

        assert 0 == Geometry.angle_btw_lines(l1, l2)

    def test_3(self):
        """Validate angle btw lines scenario 3."""
        l1 = Line(0, 4, 0)
        l2 = Line(0, 3, 0)
        assert 0 == Geometry.angle_btw_lines(l1, l2)
