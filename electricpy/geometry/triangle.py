################################################################################
"""
electricpy.geometry.triangle - Collection of methods for Cartesian triangles.

>>> import electricpy.geometry.triangle as triangle

This subpackage helps handle coordinate geometry calculations on triangles
which are required for plotting various graphs in electrical engineering.
"""
################################################################################

from __future__ import annotations

from typing import Tuple, Union
import math

from electricpy.geometry import Point, Line
from electricpy import geometry


Number = Union[int, float]


def _as_float(x) -> float:
    """
    Coerce numeric-like values (including complex with ~0 imag) to float.

    This defends against upstream helpers that may return complex numbers
    (e.g., if cmath was used somewhere).
    """
    if isinstance(x, complex):
        # If imaginary part is negligible, drop it; otherwise fail loudly.
        if not math.isclose(x.imag, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Expected a real-valued result, got complex: {x}")
        return float(x.real)
    return float(x)


def _is_close(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _triangle_twice_area(p0: Point, p1: Point, p2: Point) -> float:
    """Return twice the signed area (cross product magnitude)."""
    return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)


class Triangle:
    r"""
    Triangle in Cartesian plane.

    Parameters
    ----------
    points : Point, Point, Point
        The 3 vertices of the triangle in the Cartesian plane.

    Notes
    -----
    - Rejects degenerate (collinear or near-collinear) triangles.
    - Uses tolerance-aware validation for robust behavior with floats.
    """

    def __init__(self, *points: Point, tol: float = 1e-12):
        """
        Initialize the triangle.

        Parameters
        ----------
        *points : Point
            Exactly three Point objects.
        tol : float
            Tolerance used to reject degenerate triangles and to stabilize
            numeric comparisons. Default is 1e-12.

        Raises
        ------
        ValueError
            If not exactly 3 points or if triangle is degenerate/invalid.
        """
        if len(points) != 3:
            raise ValueError("Triangle must have 3 points")

        if not all(isinstance(p, Point) for p in points):
            raise TypeError("All inputs must be electricpy.geometry.Point")

        self.points: Tuple[Point, Point, Point] = (points[0], points[1], points[2])
        self._tol = float(tol)

        # Side lengths:
        # a = |p0 - p1|, b = |p1 - p2|, c = |p0 - p2|
        self.a = _as_float(geometry.distance(self.points[0], self.points[1]))
        self.b = _as_float(geometry.distance(self.points[1], self.points[2]))
        self.c = _as_float(geometry.distance(self.points[0], self.points[2]))

        # Basic sanity on lengths
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("Invalid triangle: side lengths must be positive")

        # Lines representing the sides (whatever geometry.line_equation returns)
        self.l1 = geometry.line_equation(self.points[0], self.points[1])
        self.l2 = geometry.line_equation(self.points[1], self.points[2])
        self.l3 = geometry.line_equation(self.points[0], self.points[2])

        if not self.__is_valid():
            raise ValueError("Invalid triangle")

    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------
    def perimeter(self) -> float:
        """Return the perimeter of the triangle."""
        return self.a + self.b + self.c

    def perimeters(self) -> float:
        """Backward-compatible alias for perimeter()."""
        return self.perimeter()

    def area(self) -> float:
        """
        Return the area of the triangle.

        Uses Heron's formula with numeric clamping to handle tiny negative
        values due to floating-point round-off.
        """
        s = 0.5 * self.perimeter()
        radicand = s * (s - self.a) * (s - self.b) * (s - self.c)

        # Clamp tiny negatives caused by floating-point error.
        if radicand < 0 and radicand > -self._tol:
            radicand = 0.0

        if radicand < -self._tol:
            # This would imply an invalid triangle numerically; treat as invalid.
            raise ValueError("Triangle area became complex; triangle may be invalid numerically.")

        return math.sqrt(max(0.0, radicand))

    # -------------------------------------------------------------------------
    # Triangle centers
    # -------------------------------------------------------------------------
    def centroid(self) -> Point:
        """Return the centroid of the triangle."""
        x = (self.points[0].x + self.points[1].x + self.points[2].x) / 3.0
        y = (self.points[0].y + self.points[1].y + self.points[2].y) / 3.0
        return Point(x, y)

    def in_center(self) -> Point:
        """
        Return the in_center of the triangle.

        The in_center is the weighted average of vertices by the lengths of
        the opposite sides. With our naming:
          a = |p0-p1| opposite vertex p2
          b = |p1-p2| opposite vertex p0
          c = |p0-p2| opposite vertex p1

        So:
          I = (a*p2 + b*p0 + c*p1) / (a+b+c)
        """
        s = self.perimeter()
        if _is_close(s, 0.0, abs_tol=self._tol):
            raise ValueError("Degenerate triangle: perimeter is zero")

        x = (self.a * self.points[2].x + self.b * self.points[0].x + self.c * self.points[1].x) / s
        y = (self.a * self.points[2].y + self.b * self.points[0].y + self.c * self.points[1].y) / s
        return Point(x, y)

    def in_radius(self) -> float:
        """
        Return the in_radius of the triangle.

        r = A / s where s is semiperimeter.
        """
        semi = self.perimeter() / 2.0
        A = self.area()
        if _is_close(semi, 0.0, abs_tol=self._tol):
            raise ValueError("Degenerate triangle: semiperimeter is zero")
        return A / semi

    def ortho_center(self) -> Point:
        """
        Return the ortho_center of the triangle.

        Construct two altitudes:
        - altitude from p0 to line through p1-p2
        - altitude from p1 to line through p0-p2
        Their intersection is the ortho_center.

        Requires Line objects returned by geometry.line_equation to support:
        - foot_perpendicular(Point) -> Point
        And electricpy.geometry.Line to support:
        - Line.construct(Point, Point)
        - intersection(Line) -> Point
        """
        d1 = self.l2.foot_perpendicular(self.points[0])
        d2 = self.l3.foot_perpendicular(self.points[1])

        alt_1 = Line.construct(self.points[0], d1)
        alt_2 = Line.construct(self.points[1], d2)

        return alt_1.intersection(alt_2)

    def circum_center(self) -> Point:
        """
        Return the circum_center of the triangle.

        Intersection of perpendicular bisectors of two sides.
        """
        pb_1 = geometry.perpendicular_bisector(self.points[0], self.points[1])
        pb_2 = geometry.perpendicular_bisector(self.points[1], self.points[2])
        return pb_1.intersection(pb_2)

    def circum_radius(self) -> float:
        """
        Return the circum_radius of the triangle.

        R = abc / (4A)
        """
        A = self.area()
        if _is_close(A, 0.0, abs_tol=self._tol):
            raise ValueError("Degenerate triangle: area is zero, circum_radius undefined")
        return (self.a * self.b * self.c) / (4.0 * A)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def __is_valid(self) -> bool:
        """
        Validate triangle.

        Checks:
        - triangle inequality with tolerance
        - non-collinear (non-degenerate) using cross-product area test
        """
        # Triangle inequality (tolerant)
        if (self.a + self.b) <= self.c + self._tol:
            return False
        if (self.a + self.c) <= self.b + self._tol:
            return False
        if (self.b + self.c) <= self.a + self._tol:
            return False

        # Non-collinear check (area not ~ 0)
        twoA = _triangle_twice_area(self.points[0], self.points[1], self.points[2])
        if _is_close(twoA, 0.0, abs_tol=self._tol):
            return False

        return True
