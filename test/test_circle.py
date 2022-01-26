from electricpy.geometry.circle import Circle
from electricpy.geometry import Line, Point
import cmath

def test_area():

    def test_0():

        c = Circle((0, 0), 1)
        assert c.area() == cmath.pi

        c = Circle((0, 0), 2)
        assert c.area() == cmath.pi * 4

    def test_1():

        c = Circle((0, 0), 1.1)
        assert c.area() == cmath.pi * 1.1**2

        c = Circle((0, 0), 2.2)
        assert c.area() == cmath.pi * 2.2**2

    for i in range(2):
        exec("test_{}()".format(i))

def test_circumference():

    def test_0():

        c = Circle((0, 0), 1)
        assert c.circumference() == cmath.pi * 2

        c = Circle((0, 0), 2)
        assert c.circumference() == cmath.pi * 4

    def test_1():

        c = Circle((0, 0), 1.1)
        assert c.circumference() == cmath.pi * 2.2

        c = Circle((0, 0), 2.2)
        assert c.circumference() == cmath.pi * 4.4

    for i in range(2):
        exec("test_{}()".format(i))

def test_tangent():

    def test_0():
        c = Circle((0, 0), 1)

        assert c.tangent(Point(0, 1)) == Line(0, 1, -1)
        assert c.tangent(Point(0, -1)) == Line(0, -1, -1)
        assert c.tangent(Point(1, 0)) == Line(1, 0, -1)
        assert c.tangent(Point(-1, 0)) == Line(-1, 0, -1)


    def test_1():

        from test import compare_lines

        c = Circle((0, 0), 1)

        p = Point(cmath.cos(cmath.pi/4), cmath.sin(cmath.pi/4))
        p1 = Point(cmath.sqrt(2), 0)
        p2 = Point(0, cmath.sqrt(2))

        assert compare_lines(c.tangent(p), Line.construct(p1, p2))

    for i in range(2):
        exec("test_{}()".format(i))

def test_normal():

    def test_0():
        c = Circle((0, 0), 1)

        assert c.normal(Point(0, 1)) == Line(1, 0, 0)
        assert c.normal(Point(0, -1)) == Line(1, 0, 0)
        assert c.normal(Point(1, 0)) == Line(0, 1, 0)
        assert c.normal(Point(-1, 0)) == Line(0, 1, 0)

    def test_1():

        from test import compare_lines

        c = Circle((0, 0), 1)
        p0 = Point(cmath.cos(cmath.pi/4), cmath.sin(cmath.pi/4))
        p1 = Point(-cmath.cos(cmath.pi/4), cmath.sin(cmath.pi/4))

        assert compare_lines(c.normal(p0), Line(1, -1, 0))
        assert compare_lines(c.normal(p1), Line(1, 1, 0))

    for i in range(2):
        exec("test_{}()".format(i))
