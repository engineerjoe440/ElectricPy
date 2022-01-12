################################################################################
"""
`electricpy.geometry.triangle` - Collection of methods which operate on \
cartesial Triangle.

>>> import electricpy.geometry.triangle as triangle

This sub package help to handle coordinate geometry calculatios on triangle 
which are required for plotting various graphs in electrical engineering. 
"""
################################################################################
from typing import List
from electricpy.geometry import Point
from electricpy.geometry import Line
from electricpy import geometry

#Type cast complex to float

class Triangle:
    r"""
    Triangle in cartesian plane.

    Parameters
    ----------
    points : list of Point, List[Point]
        The points of the triangle in cartesian plane.
    """

    def __init__(self, *points: List[Point]):
        """Initialize the triangle."""
        if len(points) != 3:
            raise ValueError('Triangle must have 3 points')

        self.a = geometry.distance(points[0], points[1])
        self.b = geometry.distance(points[1], points[2])
        self.c = geometry.distance(points[0], points[2])

        self.l1 = geometry.line_equation(points[0], points[1])
        self.l2 = geometry.line_equation(points[1], points[2])
        self.l3 = geometry.line_equation(points[0], points[2])

        if not self.__is_valid():
            raise ValueError("Invalid triangle")
        else:
            self.points = points

    def centroid(self):
        """Return the centroid of the triangle."""
        x = (self.points[0].x + self.points[1].x + self.points[2].x) / 3
        y = (self.points[0].y + self.points[1].y + self.points[2].y) / 3
        return Point(x, y)

    def in_center(self):
        """Return the inCenter of the triangle."""
        s = self.perimeters()
        i = (self.a * self.points[2].x + self.b * self.points[0].x + self.c * self.points[1].x) / s, \
                (self.a * self.points[2].y + self.b * self.points[0].y + self.c * self.points[1].y) / s
        return Point(i[0], i[1])

    def in_radius(self):
        """Return the inRadius of the triangle."""
        return self.area() / (self.perimeters()/2)

    def ortho_center(self):
        """Return the orthoCenter of the triangle."""
        d1 = self.l2.foot_perpendicular(self.points[0])
        d2 = self.l3.foot_perpendicular(self.points[1])

        alt_1 = Line.construct(self.points[0], d1)
        alt_2 = Line.construct(self.points[1], d2)

        return alt_1.intersection(alt_2)

    def circum_center(self):
        """Return the circumCenter of the triangle."""
        pb_1 = geometry.perpendicular_bisector(self.points[0], self.points[1])
        pb_2 = geometry.perpendicular_bisector(self.points[1], self.points[2])

        return pb_1.intersection(pb_2)

    def circum_radius(self):
        """Return the circumRadius of the triangle."""
        return (self.a*self.b*self.c) / (4 * self.area())

    def area(self):
        """Return the area of the triangle."""
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5

    def perimeters(self):
        """Return the perimeters of the triangle."""
        return self.a + self.b + self.c

    def __is_valid(self):
        return (self.a + self.b > self.c) and (self.a + self.c > self.b) and (self.b + self.c > self.a)