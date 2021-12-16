from Geometry import circle
import cmath

def test_area():

    def test_0():

        c = circle.Circle(1)
        assert c.area() == cmath.pi

        c = circle.Circle(2)
        assert c.area() == cmath.pi * 4

    def test_1():

        c = circle.Circle(1.1)
        assert c.area() == cmath.pi * 1.1**2

        c = circle.Circle(2.2)
        assert c.area() == cmath.pi * 4.4**2

def test_perimeter():

    def test_0():

        c = circle.Circle(1)
        assert c.perimeter() == cmath.pi * 2

        c = circle.Circle(2)
        assert c.perimeter() == cmath.pi * 4

    def test_1():

        c = circle.Circle(1.1)
        assert c.perimeter() == cmath.pi * 2.2

        c = circle.Circle(2.2)
        assert c.perimeter() == cmath.pi * 4.4


def test_parametric_equation():

    def test_0():
        pass


    def test_1():
        pass

    for i in range(2):
        exec("test_{}()".format(i))

