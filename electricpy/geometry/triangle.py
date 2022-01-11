from typing import List

from electricpy.geometry import Point
from electricpy.geometry import Line
import electricpy.geometry as geometry

#Type cast complex to float

class Triangle:

    def __init__(self, *points: List[Point]):

        if len(points) != 3:
            raise ValueError('Triangle must have 3 points')

        self.a = geometry.distance(points[0], points[1])
        self.b = geometry.distance(points[1], points[2])
        self.c = geometry.distance(points[0], points[2])

        self.l1 = geometry.line_equation(points[0], points[1])
        self.l2 = geometry.line_equation(points[1], points[2])
        self.l3 = geometry.line_equation(points[0], points[2])

        if not self.is_valid():
            raise ValueError("Invalid triangle")
        else:
            self.points = points

    def centroid(self):
        x = (self.points[0].x + self.points[1].x + self.points[2].x) / 3
        y = (self.points[0].y + self.points[1].y + self.points[2].y) / 3
        return Point(x, y)

    def in_center(self):
        s = self.perimeters()
        i = (self.a * self.points[2].x + self.b * self.points[0].x + self.c * self.points[1].x) / s, \
                (self.a * self.points[2].y + self.b * self.points[0].y + self.c * self.points[1].y) / s
        return Point(i[0], i[1])

    def in_radius(self):
        return self.area() / (self.perimeters()/2)

    def ortho_center(self):

        d1 = self.l2.foot_perpendicular(self.points[0])
        d2 = self.l3.foot_perpendicular(self.points[1])

        alt_1 = Line.construct(self.points[0], d1)
        alt_2 = Line.construct(self.points[1], d2)

        return alt_1.intersection(alt_2)

    def circum_center(self):

        pb_1 = geometry.perpendicular_bisector(self.points[0], self.points[1])
        pb_2 = geometry.perpendicular_bisector(self.points[1], self.points[2])

        return pb_1.intersection(pb_2)

    def circum_radius(self):
        return (self.a*self.b*self.c) / (4 * self.area())

    def area(self):
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5

    def perimeters(self):
        return self.a + self.b + self.c

    def is_valid(self):
        return (self.a + self.b > self.c) and (self.a + self.c > self.b) and (self.b + self.c > self.a)