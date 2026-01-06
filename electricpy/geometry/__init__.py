################################################################################
"""
electricpy.geometry  Geometry Sub Module.

>>> import electricpy.geometry as geometry

This package helps handle coordinate geometry calculations which are required
for plotting various graphs in electrical engineering.

Built to support operations similar to NumPy/SciPy, this package is designed
to aid in scientific calculations.
"""
################################################################################

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple, Union


Number = Union[int, float]


def _as_float(x) -> float:
    """
    Coerce numeric-like values (including complex with ~0 imag) to float.
    This defends against accidental cmath usage elsewhere in the library.
    """
    if isinstance(x, complex):
        if not math.isclose(x.imag, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Expected real value, got complex: {x}")
        return float(x.real)
    return float(x)


def _is_close(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


@dataclass(frozen=False)
class Point:
    """A point in 2D space.

    Parameters
    ----------
    x : float
        The x coordinate of the point
    y : float
        The y coordinate of the point
    """
    x: float
    y: float

    def __init__(self, x: Number, y: Number):
        self.x = _as_float(x)
        self.y = _as_float(y)

    def __iter__(self) -> Iterator[float]:
        """Return an iterator over (x, y)."""
        yield self.x
        yield self.y

    def __call__(self) -> Tuple[float, float]:
        """Return (x, y)."""
        return (self.x, self.y)

    def __eq__(self, other: object) -> bool:
        """Exact equality (kept for backward compatibility)."""
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def is_close(self, other: "Point", *, tol: float = 1e-9) -> bool:
        """Tolerance-based point comparison."""
        if not isinstance(other, Point):
            return False
        return _is_close(self.x, other.x, rel_tol=tol, abs_tol=tol) and _is_close(self.y, other.y, rel_tol=tol, abs_tol=tol)

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


@dataclass(frozen=False)
class Line:
    """A line in 2D space in the form:

        ax + by + c = 0
    """
    a: float
    b: float
    c: float

    def __init__(self, a: Number, b: Number, c: Number):
        self.a = _as_float(a)
        self.b = _as_float(b)
        self.c = _as_float(c)

        if _is_close(self.a, 0.0) and _is_close(self.b, 0.0):
            raise AssertionError("Line cannot have both a and b equal to zero.")

    @staticmethod
    def construct(p1: Point, p2: Point) -> "Line":
        """Construct a line from two points."""
        return line_equation(p1, p2)

    def ordinate(self, x: Number) -> float:
        """Return y for a given x (only if the line is not vertical)."""
        if _is_close(self.b, 0.0):
            raise ZeroDivisionError("ordinate is not defined for vertical lines")
        x = _as_float(x)
        return -1.0 * (self.a * x + self.c) / self.b

    def __call__(self, p: Point) -> float:
        """Evaluate ax + by + c at point p."""
        return self.a * p.x + self.b * p.y + self.c

    def slope(self) -> float:
        """Return the slope of the line (only if not vertical)."""
        if _is_close(self.b, 0.0):
            raise ZeroDivisionError("slope is not defined for vertical lines")
        return -self.a / self.b

    def intercepts(self) -> dict:
        """Return dict with x- and y-intercepts (None if undefined)."""
        data = {}
        data["x"] = None if _is_close(self.a, 0.0) else (-self.c / self.a)
        data["y"] = None if _is_close(self.b, 0.0) else (-self.c / self.b)
        return data

    def distance(self, p: Point) -> float:
        """Perpendicular distance from point to this line."""
        return line_distance(p, self)

    def foot_perpendicular(self, p: Point) -> Point:
        """Foot of perpendicular from p to this line."""
        return foot_perpendicular(p, self)

    def image(self, p: Point) -> Point:
        """Reflection of p across this line."""
        return point_image(p, self)

    def intersection(self, other: object) -> Point:
        """Intersection point with another line (raises if parallel)."""
        return line_intersection(self, other)

    def normalized(self) -> Tuple[float, float, float]:
        """
        Return a normalized (a,b,c) such that sqrt(a^2+b^2)=1 and sign is stable.
        This helps with comparisons and distances.
        """
        norm = math.hypot(self.a, self.b)
        if _is_close(norm, 0.0):
            raise ZeroDivisionError("Cannot normalize a degenerate line.")
        a = self.a / norm
        b = self.b / norm
        c = self.c / norm

        # Stabilize sign: make first nonzero among (a,b) positive
        if a < 0 or (_is_close(a, 0.0) and b < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def is_close(self, other: "Line", *, tol: float = 1e-9) -> bool:
        """Tolerance-based equality: normalized coefficients are close."""
        if not isinstance(other, Line):
            return False
        a1, b1, c1 = self.normalized()
        a2, b2, c2 = other.normalized()
        return (
            _is_close(a1, a2, rel_tol=tol, abs_tol=tol)
            and _is_close(b1, b2, rel_tol=tol, abs_tol=tol)
            and _is_close(c1, c2, rel_tol=tol, abs_tol=tol)
        )

    def __eq__(self, other: object) -> bool:
        """
        Exact-ish equality (proportional coefficients), but using normalization for
        better behavior than raw ratio checks. This is safer than the original.
        """
        if not isinstance(other, Line):
            return False
        return self.is_close(other, tol=0.0)  # exact compare after normalization

    def __repr__(self) -> str:
        return f"Line({self.a}, {self.b}, {self.c})"

    def __str__(self) -> str:
        # Keep a readable form; avoid division by zero when possible.
        if _is_close(self.a, 0.0):
            # by + c = 0 => y = -c/b
            return f"y = {-self.c / self.b}"
        if _is_close(self.b, 0.0):
            # ax + c = 0 => x = -c/a
            return f"x = {-self.c / self.a}"
        # General form
        return f"{self.a}x + {self.b}y + {self.c} = 0"


def angle_btw_lines(l1: Line, l2: Line) -> float:
    """
    Return the acute angle (in radians) between two lines l1 and l2.

    Uses direction vectors; robust and stays in real math:
        tan(theta) = |(m1 - m2) / (1 + m1*m2)|
    but avoids slope singularities by using coefficients.
    """
    # Direction vector for line ax+by+c=0 is (b, -a).
    d1x, d1y = l1.b, -l1.a
    d2x, d2y = l2.b, -l2.a

    dot = d1x * d2x + d1y * d2y
    n1 = math.hypot(d1x, d1y)
    n2 = math.hypot(d2x, d2y)

    if _is_close(n1, 0.0) or _is_close(n2, 0.0):
        raise ValueError("Cannot compute angle for degenerate line.")

    # Clamp to [-1,1] to protect against tiny numeric drift
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    ang = math.acos(cosang)

    # Return acute angle
    if ang > math.pi / 2:
        ang = math.pi - ang
    return ang


def line_intersection(l1: Line, l2: object) -> Point:
    """Calculate the intersection point of two lines (raises if parallel)."""
    if not isinstance(l2, Line):
        raise TypeError("line_intersection requires two Line objects.")

    # Solve:
    # a1 x + b1 y + c1 = 0
    # a2 x + b2 y + c2 = 0
    a1, b1, c1 = l1.a, l1.b, l1.c
    a2, b2, c2 = l2.a, l2.b, l2.c

    det = a1 * b2 - a2 * b1
    if _is_close(det, 0.0):
        raise ZeroDivisionError("lines are parallel")

    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return Point(x, y)


def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points (real float)."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def section(p1: Point, p2: Point, ratio: Union[Tuple[Number, Number], float]) -> Point:
    """
    Calculate the point dividing segment p1->p2.

    If ratio is float t: returns p1 + t*(p2-p1).
    If ratio is (m,n): returns internal division in ratio m:n (convention preserved).
    """
    if isinstance(ratio, (int, float)):
        t = _as_float(ratio)
        return Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
    else:
        m, n = ratio
        m = _as_float(m)
        n = _as_float(n)
        if _is_close(m + n, 0.0):
            raise ZeroDivisionError("Invalid ratio: m+n must be nonzero.")
        # Preserve original convention: m corresponds to p2 weight, n to p1 weight via:
        # p = (n*p1 + m*p2)/(m+n)
        return Point((n * p1.x + m * p2.x) / (m + n), (n * p1.y + m * p2.y) / (m + n))


def midpoint(p1: Point, p2: Point) -> Point:
    """Calculate midpoint of segment p1-p2."""
    return Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)


def slope(p1: Point, p2: Point) -> float:
    """Calculate slope between two points (raises if vertical)."""
    dx = p2.x - p1.x
    if _is_close(dx, 0.0):
        raise ZeroDivisionError("slope is not defined for vertical lines")
    return (p2.y - p1.y) / dx


def line_equation(p1: Point, p2: Point) -> Line:
    """
    Return the line through p1 and p2 in ax + by + c = 0 form.

    Uses determinant form:
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1
    which avoids slope division and handles vertical/horizontal lines cleanly.
    """
    if p1 == p2:
        raise ValueError("Cannot define a line from two identical points.")

    a = p1.y - p2.y
    b = p2.x - p1.x
    c = (p1.x * p2.y) - (p2.x * p1.y)
    return Line(a, b, c)


def slope_point_line(slope_val: Number, p: Point) -> Line:
    """Return line with given slope passing through point p."""
    m = _as_float(slope_val)
    # y - y0 = m(x - x0) => m x - y + (y0 - m x0)=0
    a = m
    b = -1.0
    c = p.y - m * p.x
    return Line(a, b, c)


def line_distance(p: Point, line: Line) -> float:
    """Perpendicular distance between point and line."""
    denom = math.hypot(line.a, line.b)
    if _is_close(denom, 0.0):
        raise ZeroDivisionError("Distance undefined for degenerate line.")
    return abs(line(p)) / denom


def foot_perpendicular(p: Point, line: Line) -> Point:
    """Foot of perpendicular from point p to line."""
    denom = (line.a * line.a + line.b * line.b)
    if _is_close(denom, 0.0):
        raise ZeroDivisionError("Foot undefined for degenerate line.")
    d = -line(p) / denom
    return Point(p.x + line.a * d, p.y + line.b * d)


def point_image(p: Point, line: Line) -> Point:
    """Reflection of p across line."""
    fp = foot_perpendicular(p, line)
    return Point(2.0 * fp.x - p.x, 2.0 * fp.y - p.y)


def perpendicular_bisector(p1: Point, p2: Point) -> Line:
    """
    Perpendicular bisector of segment p1-p2.

    Construct using midpoint M and direction perpendicular to segment:
      segment direction v = (dx, dy)
      perpendicular direction w = (-dy, dx)
    Line through M with direction w gives equation:
      A x + B y + C = 0 with (A,B) = (dx, dy) and C = -(A*mx + B*my)
    because perpendicular bisector has normal vector equal to segment direction.
    """
    if p1 == p2:
        raise ValueError("Perpendicular bisector undefined for identical points.")
    mx, my = (p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0
    dx, dy = (p2.x - p1.x), (p2.y - p1.y)

    # Normal vector to bisector is (dx, dy)
    a = dx
    b = dy
    c = -(a * mx + b * my)
    return Line(a, b, c)


def colinear(p1: Point, p2: Point, p3: Point, *, tol: float = 1e-12) -> bool:
    """
    Determine whether 3 points are collinear (tolerance-based).

    Uses twice-area (cross product):
      (p2-p1) x (p3-p1) == 0
    """
    area2 = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    return _is_close(area2, 0.0, abs_tol=tol, rel_tol=tol)
