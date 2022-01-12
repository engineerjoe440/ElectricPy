################################################################################
"""
`electricpy.geometry`  Geometry Sub Module.

>>> import electricpy.geometry as geometry

This Package help to handle coordinate geometry calculatios which are required 
for plotting various graphs in electrical engineering. 

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################
import cmath
from typing import Tuple, Union
from typing import Tuple


class Point:
    """A point in 2D space.
    
    Parameters
    ----------
    x : float
        The x coordinate of the point
    y : float
        The y coordinate of the point
    """

    def __init__(self, x: float, y: float):
        """Initialize the point."""
        self.x = x
        self.y = y

    def __iter__(self) -> float:
        """Return an iterator for the point."""
        yield self.x
        yield self.y

    def __call__(self) -> Tuple:
        """Return the coordinates of the point."""
        return (self.x, self.y)

    def __ne__(self, other):
        """Return true if the points are not equal."""
        return not self == other

    def __eq__(self, __o: object) -> bool:
        """Return true if the points are equal."""
        if isinstance(__o, Point):
            return self.x == __o.x and self.y == __o.y
        else:
            return False

    def __repr__(self) -> str:
        """Return the representation of the point."""
        return f"Point({self.x}, {self.y})"

    def __str__(self) -> str:
        """Return the string representation of the point."""
        return f"({self.x}, {self.y})"


class Line:
    """A line in 2D space in the form .

    math:: ax + by + c = 0
    
    Parameters
    ----------
    a : float
        The a coefficient of the line
    b : float
        The b coefficient of the line
    c : float
        The c coefficient of the line
    """

    def __init__(self, a: float, b: float, c: float):
        """Initialize the line."""
        self.a = a
        self.b = b
        self.c = c
        try:
            assert self.a or self.b != 0
        except AssertionError:
            raise AssertionError("line can not have all co-efficients zeros")

    def ordinate(self, x):
        """Return the ordinate of the line at the given x value."""
        try:
            return -1*(self.a * x + self.c)/self.b
        except ZeroDivisionError:
            raise ZeroDivisionError(
                "ordinate is not defined for vertical lines")

    @staticmethod
    def construct(p1: Point, p2: Point):
        """Construct a line from two points."""
        return line_equation(p1, p2)

    def __call__(self, p: Point) -> float:
        """Return the value of the point when subsituted in a line."""
        return self.a * p.x + self.b * p.y + self.c

    def __str__(self):
        """Return the string representation of the line."""
        if self.a == 0:
            return f"y = {-self.c/self.b}"
        elif self.b == 0:
            return f"x = {-self.c/self.a}"
        elif self.c == 0:
            if self.a < 0:
                return f'{-self.a}x + {-self.b}y = 0'
            else:
                return f"{self.a}x + {self.b}y = 0"
        elif self.a < 0:
            return f'{-self.a}x + {-self.b}y + {-self.c} = 0'
        else:
            return f"{self.a}x + {self.b}y + {self.c} = 0"

    def __repr__(self) -> str:
        """Return the representation of the line."""
        return f"Line({self.a}, {self.b}, {self.c})"

    def slope(self):
        """Return the slope of the line."""
        try:
            return -self.a / self.b
        except ZeroDivisionError:
            raise ZeroDivisionError("slope is not defined for vertical lines")

    def intercepts(self):
        """Return the intercepts of the line."""
        data = dict()
        try:
            data['x'] = -self.c / self.a
        except ZeroDivisionError:
            data['x'] = None
        try:
            data['y'] = -self.c / self.b
        except ZeroDivisionError:
            data['y'] = None
        return data

    def __eq__(self, __o: object) -> bool:
        """Return true if the lines are equal."""
        if self.a == 0 and __o.a == 0:
            if self.c == 0 and __o.c == 0:
                return True
            else:
                try:
                    return self.b / __o.b == self.c / __o.c
                except ZeroDivisionError:
                    return False

        if self.b == 0 and __o.b == 0:
            if self.c == 0 and __o.c == 0:
                return True
            else:
                try:
                    return self.a / __o.a == self.c / __o.c
                except ZeroDivisionError:
                    return False

        if self.c == 0 and __o.c == 0:
            try:
                return self.a / __o.a == self.b / __o.b
            except ZeroDivisionError:
                return False
        try:
            return self.a / __o.a == self.b / __o.b == self.c / __o.c
        except ZeroDivisionError:
            return False

    def distance(self, p: Point) -> float:
        """Return the distance of the point from the line."""
        return line_distance(p, self)

    def foot_perpendicular(self, p: Point) -> Point:
        """Return the foot of the perpendicular from a point to line."""
        return foot_perpendicular(p, self)

    def image(self, p: Point) -> Point:
        """Return the image of a point with respect to line."""
        return point_image(p, self)

    def intersection(self, l1: object) -> Union[Point, None]:
        """Return the intersection of two lines."""
        return line_intersection(self, l1)


def line_intersection(l1: Line, l2: Line) -> Point:
    """Calculate the intersection point of two lines."""
    a1, b1, c1 = l1.a, l1.b, -l1.c
    a2, b2, c2 = l2.a, l2.b, -l2.c

    try:
        x = (b2*c1 - b1*c2) / (a1*b2 - a2*b1)
        y = (a1*c2 - a2*c1) / (a1*b2 - a2*b1)
    except ZeroDivisionError:
        raise ZeroDivisionError("lines are parallel")
    return Point(x, y)


def distance(p1: Point, p2: Point) -> float:
    """Calculate the distance between two points."""
    d = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    if isinstance(d, complex):
        return d.real
    else:
        return d


def section(p1: Point, p2: Point, ratio: Union[Tuple, float]) -> Point:
    """Calculate the point on a line section."""
    if isinstance(ratio, float):
        return Point(p1.x + ratio * (p2.x - p1.x), p1.y + ratio * (p2.y - p1.y))
    else:
        (x, y) = ratio
        m = x / (x + y)
        n = y / (x + y)
        return Point(n*p1.x + m*p2.x, n*p1.y + m*p2.y)


def midpoint(p1: Point, p2: Point) -> Point:
    """Calculate the midpoint between two points."""
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def slope(p1: Point, p2: Point) -> float:
    """Calculate the slope between two points."""
    try:
        return (p2.y - p1.y) / (p2.x - p1.x)
    except ZeroDivisionError:
        raise ZeroDivisionError("slope is not defined for vertical lines")


def line_equation(p1: Point, p2: Point) -> Line:
    """Calculate the line equation between two points."""
    # try:
    #     a = slope(p1, p2)
    # except ZeroDivisionError:
    #     assert p1.x == p2.x
    #     a = 1
    #     c = -p1.x
    #     return Line(a, 0, c)
    # b = -1
    # c = p1.y - a * p1.x
    # return Line(a, b, c)
    return Line(p2.y - p1.y, p1.x - p2.x, p1.x*(p1.y - p2.y) + p1.y*(p2.x - p1.x))


def slope_point_line(slope: float, p: Point) -> Line:
    """Calculate the line equation from a slope and a point."""
    a = slope
    b = -1
    c = p.y - a * p.x
    return Line(a, b, c)


def line_distance(p: Point, line: Line) -> float:
    """Calculate the distance between a point and a line."""
    return abs(line(p)) / (line.a ** 2 + line.b ** 2) ** 0.5


def foot_perpendicular(p: Point, line: Line) -> Point:
    """Calculate the foot perpendicular from a point to a line."""
    d = -line(p)/(line.a**2+line.b**2)
    x = p.x + line.a * d
    y = p.y + line.b * d
    return Point(x, y)


def point_image(p: Point, line: Line) -> Point:
    """Calculate the image of a point when a line is acting like a mirror."""
    p1: Point = foot_perpendicular(p, line)
    return Point(2*p1.x - p.x, 2*p1.y - p.y)


def perpendicular_bisector(p1: Point, p2: Point) -> Line:
    """Calculate the perpendicular bisector of two points."""
    try:
        m = slope(p1, p2)
    except ZeroDivisionError:
        return Line(0, 1, -(p1.y + p2.y) / 2)

    if m == 0:
        return Line(1, 0, -(p1.x + p2.x) / 2)
    else:
        x1, x2 = p1.x, p2.x
        y1, y2 = p1.y, p2.y
        return Line(2*(x2 - x1), 2*(y2 - y1), -x1**2 + -y1**2 + x2**2 + y2**2)


def colinear(p1: Point, p2: Point, p3: Point) -> bool:
    """Determine whether 3 points are colinear or not ."""
    l1 = line_equation(p1, p2)
    return l1(p3) == 0
