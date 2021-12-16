from Geometry import triangle
from Geometry import Point
import cmath

class Circle:
    
    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    def area(self):
        return cmath.pi * self.radius ** 2

    def circumference(self):
        return 2 * cmath.pi * self.radius

    def parametric_equation(self, theta_resolution: float = 0.01, semi=False):
        i = 0
        if semi:
            k = cmath.pi
        else:
            k = 2 * cmath.pi
        while i < k:
            yield self.center.x + self.radius * cmath.cos(i), self.center.y + self.radius * cmath.sin(i)
            i += theta_resolution

    def __repr__(self):
        return 'Circle(center={0}, radius={1})'.format(self.center, self.radius)

    def __eq__(self, other):
        return self.center == other.center and self.radius == other.radius

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.center, self.radius))

    def __str__(self):
        return 'Circle(center={0}, radius={1})'.format(self.center, self.radius)
