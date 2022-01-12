################################################################################
"""
`electricpy.geometry.circle` - Collection of methods which operate on \
cartesial Circle.

>>> import electricpy.geometry.circle as circle

This sub package help to handle coordinate geometry calculatios on circles 
which are required for plotting various graphs in electrical engineering. 
"""
################################################################################
from typing import Union
from electricpy import geometry
from electricpy.geometry import Line
from electricpy.geometry import Point
import cmath

class Circle:
    r"""
    Circle in cartesian plane.
    
    Parameters
    ----------
    center : Point
        The center of the circle
    radius : float
        The radius of the circle
    """

    def __init__(self, center: Union[Point, tuple, list], radius: float):
        """Initialize the circle."""
        if isinstance(center, tuple) or isinstance(center, list):
            assert len(center) == 2, "Center must be a 2-tuple or list"
            center = Point(center[0], center[1])

        self.center = center
        self.radius = radius

    def area(self) -> float:
        """Return the area of the circle."""
        return cmath.pi * self.radius ** 2

    def circumference(self) -> float:
        """Return the circumference of the circle."""
        return 2 * cmath.pi * self.radius

    def tangent(self, p: Point) -> Line:
        """Return the tangent line to the circle at point p."""
        # try:
        #     m = Geometry.slope(self.center, p)
        # except ZeroDivisionError:
        #     return Line(0, 1, -p.y)

        # if m == 0:
        #     return Line(1, 0, -p.x)  
        
        # m = -1/m
        # return Geometry.slope_point_line(m, p)
        x, y = p.x, p.y
        c_x , c_y = -self.center.x, -self.center.y
        return Line(x + c_x, y + c_y, x*c_x + y*c_y + c_x**2 + c_y**2 - self.radius**2)

    def normal(self, p: Point) -> Line:
        """Return the normal line to the circle at point p."""
        return Line.construct(p, self.center)

    def power(self, p: Point) -> float:
        """Return the power of the circle at point p."""
        return geometry.distance(self.center, p) ** 2 - self.radius ** 2

    def is_tangent(self, l: Line) -> bool:
        """Return True if the line is tangent to the circle."""
        return l.distance(self.center) == self.radius

    def is_normal(self, l: Line) -> bool:
        """Return True if the line is normal to the circle."""
        return l(self.center) == 0

    def equation(self) -> str:
        """Return the equation of the circle."""
        (x, y) = self.center
        return f"x^2 + 2*{-x}*x + 2*{-y}*y + y^2 + {x**2 + y**2 - self.radius**2} = 0"

    def parametric_equation(self, theta_resolution: float = 0.01, semi=False):
        """Return the parametric equation of the circle."""
        i = 0
        if semi:
            k = cmath.pi
        else:
            k = 2 * cmath.pi
        while i < k:
            yield self.center.x + self.radius * cmath.cos(i), self.center.y + self.radius * cmath.sin(i)
            i += theta_resolution

    def sector_length(self, theta: float) -> float:
        """Return the length of a sector of the circle which subtended angle theta(radians) at center."""
        return self.radius * theta

    def sector_area(self, theta: float) -> float:
        """Return the area of a sector of the circle which subtended angle theta(radians) at center."""
        return self.radius ** 2 * theta / 2

    def intersetion(self, other) -> Union[Point, None]:
        """Return the intersection point of the circle and the other circle."""
        if isinstance(other, Circle):
            c1 = self.center
            c2 = other.center

            m = geometry.slope(c1, c2)
            theta = cmath.atan(m)

            d = geometry.distance(c1, c2)

            if d == self.radius + other.radius:
                """Two circles are touching each other"""
                x = c1.x + self.radius * cmath.cos(theta)
                y = c1.y + self.radius * cmath.sin(theta)
                return Point(x, y)

            elif d < self.radius + other.radius:
                """Two circles intersect"""
                r1 = self.radius
                r2 = other.radius

                theta = cmath.asin(r2 / d)

                x = c1.x + r1 * cmath.cos(theta)
                y = c1.y + r1 * cmath.sin(theta)

                p1 = Point(x, y)
                l = Line.construct(c1, c2)
                p2 = l.image(p1)

                return (p1, p2)
            else:
                return None

        else:
            raise ValueError("Can only intersect with another circle")

    def __repr__(self):
        """Return a string representation of the circle."""
        return 'Circle(center={0}, radius={1})'.format(self.center, self.radius)

    def __eq__(self, other):
        """Return True if the circles are equal."""
        if isinstance(other, Circle):
            return self.center == other.center and self.radius == other.radius
        else:
            return False

    def __ne__(self, other):
        """Return True if the circles are not equal."""
        return not self == other

    def __hash__(self):
        """Return a hash of the circle."""
        return hash((self.center, self.radius))

    def __str__(self):
        """Return a string representation of the circle."""
        return 'Circle(center={0}, radius={1})'.format(self.center, self.radius)

def construct(p0: Point, p1: Point, p2: Point) -> Circle:
    """Return a circle which passes through the three points."""
    try:
        assert not geometry.colinear(p0, p1, p2)
    except AssertionError:
        raise AssertionError("Circle can not be constructed from three points that are colinear")
    
    l1 = geometry.perpendicular_bisector(p0, p1)
    l2 = geometry.perpendicular_bisector(p1, p2)

    center = l1.intersection(l2)
    radius = geometry.distance(center, p0)

    return Circle(center, radius)
