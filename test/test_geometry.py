# helps vscode to find the path for source directory
# import sys
# sys.path.append("D:\ElectricProject\Geometry")
import Geometry
from Geometry import Point
from Geometry import Line
from numpy.testing import assert_array_almost_equal

def test_distance():
    
    def test_0():
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

    def test_1():
        p1 = Point(1, 2)
        p2 = Point(1, 3)
        assert Geometry.distance(p1, p2) == 1

        p1 = Point(2.0, 1)
        p2 = Point(3.0, 1)
        assert Geometry.distance(p1, p2) == 1

    for i in range(2):
        exec("test_{}()".format(i))

def test_slope():

    def test_0():
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        assert Geometry.slope(p1, p2) == 1

        p1 = Point(4, -6)
        p2 = Point(-2, -5)
        assert Geometry.slope(p2, p1) == -1/6

    def test_1():

        p1 = Point(1, 2)
        p2 = Point(2, 2)

        assert Geometry.slope(p1, p2) == 0

        p1 = Point(1, 2)
        p2 = Point(1, 3)
        try:
            Geometry.slope(p1, p2)
        except ZeroDivisionError:
            assert True

    for i in range(2):
        exec("test_{}()".format(i))

def test_section():
        
    def test_0():
        p1 = Point(1, 2)
        p2 = Point(3, 4)

        p = Geometry.section(p1, p2, 0.5)
        assert p == Point(2, 3)

    def test_1():
        p1 = Point(-1, 3)
        p2 = Point(1, -3)

        p_computed = Geometry.section(p1, p2, (2, 3))
        p_actual = Point(-1/5, 3/5)

        assert_array_almost_equal(p_computed(), p_actual(), decimal=6)
    
    for i in range(2):
        exec("test_{}()".format(i))

def test_line_equaltion():
    
    def test_0():
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        assert Geometry.line_equation(p1, p2) == Line(1, -1, 1)

        p1 = Point(4, -6)
        p2 = Point(-2, -5)
        assert Geometry.line_equation(p1, p2) == Line(1, 6, 32)

    def test_1():
        p1 = Point(1, 2)
        p2 = Point(1, 3)
        assert Geometry.line_equation(p1, p2) == Line(1, 0, -1)

        p1 = Point(1, 2)
        p2 = Point(2, 2)
        assert Geometry.line_equation(p1, p2) == Line(0, 1, -2)

    def test_2():
        assert Line(1, 2, 3) == Line(2, 4, 6)
        assert Line(1, -1, 0) == Line(3, -3, 0)
        assert Line(1, 0, -1) == Line(3, 0, -3)

    for i in range(3):
        exec("test_{}()".format(i))

def test_line_distance():

    def test_0():
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        l = Line.construct(p1, p2)
        assert Geometry.line_distance(p1, l) == 0
        assert Geometry.line_distance(p2, l) ==  0

    def test_1():
        p1 = Point(2, 0)
        p2 = Point(2, 4)
        p = Point(0, 0)
        l = Line.construct(p1, p2)
        assert Geometry.line_distance(p, l) == 2
        assert l.distance(p) == 2

        l = Line(0, 1, -3)
        assert l.distance(p) == 3

    for i in range(2):
        exec("test_{}()".format(i))

def test_foot_perpendicular():
    
    def test_0():
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        l = Line.construct(p1, p2)
        p = Point(2, 2)
        assert Geometry.foot_perpendicular(p, l) == Point(1.5, 2.5)

        p = Point(2, 3)
        assert Geometry.foot_perpendicular(p, l) == Point(2, 3)

    def test_1():
        p = Point(-1, 3)
        l = Line(3, -4, -16)

        p_actual = l.foot_perpendicular(p)
        p_image = l.image(p)

        p_desired = Point(68/25, -49/25)

        assert_array_almost_equal(p_actual(), p_desired(), decimal=6)
        assert Geometry.midpoint(p, p_image) == p_actual 

    for i in range(2):
        exec("test_{}()".format(i))

def test_perpendicular_bisector():

    from Geometry import  perpendicular_bisector
    
    def test_0():
        p1 = Point(3, 0)
        p2 = Point(0, 3)
        l = perpendicular_bisector(p1, p2)
        assert l == Line(1, -1, 0)

    def test_1():
        p1 = Point(-3, 0)
        p2 = Point(0, 3)
        l = perpendicular_bisector(p1, p2)
        assert l == Line(1, 1, 0)

    def test_2():
        p1 = Point(3, 0)
        p2 = Point(5, 0)
        l = perpendicular_bisector(p1, p2)
        assert l == Line(1, 0, -4)

    def test_3():
        p1 = Point(0, 3)
        p2 = Point(0, 5)
        l = perpendicular_bisector(p1, p2)
        assert l == Line(0, 1, -4)


    for i in range(4):
        exec("test_{}()".format(i))

def test_colinear():

    def test_0():
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        p3 = Point(5, 6)
        assert Geometry.colinear(p1, p2, p3)

    def test_1():
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        p3 = Point(5, 7)
        assert not Geometry.colinear(p1, p2, p3)
    
    def test_2():
        p1 = Point(1, 0)
        p2 = Point(2, 0)
        p3 = Point(3, 0)
        assert Geometry.colinear(p1, p2, p3)

    for i in range(3):
        exec("test_{}()".format(i))
    