################################################################################
"""
electricpy.geometry.circle - Collection of methods which operate on Cartesian
circles.

>>> import electricpy.geometry.circle as circle

This subpackage helps handle coordinate geometry calculations on circles,
which are required for plotting various graphs in electrical engineering.
"""
################################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple, Union, overload
import math

from electricpy import geometry
from electricpy.geometry import Line, Point


Number = Union[int, float]
PointLike = Union[Point, Tuple[Number, Number], list]


def _as_point(center: PointLike) -> Point:
    """Coerce tuple/list/Point into a Point."""
    if isinstance(center, Point):
        return center
    if isinstance(center, (tuple, list)):
        if len(center) != 2:
            raise ValueError("Center must be a 2-tuple or 2-list.")
        return Point(center[0], center[1])
    raise TypeError("Center must be a Point, tuple, or list.")


def _is_close(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Robust float comparison."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


@dataclass(frozen=False)
class Circle:
    r"""
    Circle in Cartesian plane.

    Parameters
    ----------
    center : Point | (x,y) tuple | [x,y] list
        The center of the circle
    radius : float
        The radius of the circle (must be >= 0)
    """
    center: Point
    radius: float

    def __init__(self, center: PointLike, radius: float):
        center_pt = _as_point(center)
        if not isinstance(radius, (int, float)):
            raise TypeError("Radius must be a real number (int or float).")
        if radius < 0:
            raise ValueError("Radius must be non-negative.")
        object.__setattr__(self, "center", center_pt)
        object.__setattr__(self, "radius", float(radius))

    # -------------------------------------------------------------------------
    # Basic properties
    # -------------------------------------------------------------------------
    def area(self) -> float:
        """Return the area of the circle."""
        return math.pi * (self.radius ** 2)

    def circumference(self) -> float:
        """Return the circumference of the circle."""
        return 2.0 * math.pi * self.radius

    # -------------------------------------------------------------------------
    # Core geometry helpers
    # -------------------------------------------------------------------------
    def contains_point(self, p: Point, *, tol: float = 1e-9) -> bool:
        """Return True if point p lies on the circle (within tolerance)."""
        d = geometry.distance(self.center, p)
        return _is_close(d, self.radius, rel_tol=tol, abs_tol=tol)

    def power(self, p: Point) -> float:
        """Return the power of the circle at point p: |CP|^2 - r^2."""
        return (geometry.distance(self.center, p) ** 2) - (self.radius ** 2)

    # -------------------------------------------------------------------------
    # Lines related to the circle
    # -------------------------------------------------------------------------
    def tangent(self, p: Point, *, require_on_circle: bool = True, tol: float = 1e-9) -> Line:
        """
        Return the tangent line to the circle at point p.

        Notes
        -----
        Tangent at p is defined only when p lies on the circle. If you want
        tangents *from* an external point to the circle, that is a different
        construction and not what this method computes.

        The tangent line at point (x1, y1) on circle centered at (h, k) with
        radius r is:
            (x1 - h)(x - h) + (y1 - k)(y - k) = r^2
        which expands to A x + B y + C = 0 where:
            A = (x1 - h)
            B = (y1 - k)
            C = -(A*h + B*k + r^2)  with A,B as above.
        """
        if require_on_circle and not self.contains_point(p, tol=tol):
            raise ValueError("Point p must lie on the circle to define the tangent at p.")

        h, k = self.center.x, self.center.y
        x1, y1 = p.x, p.y

        A = (x1 - h)
        B = (y1 - k)
        C = -(A * h + B * k + (self.radius ** 2))

        # Line is assumed to be in the form A*x + B*y + C = 0
        return Line(A, B, C)

    def normal(self, p: Point) -> Line:
        """
        Return the normal line to the circle at point p.

        Notes
        -----
        The normal at a point on a circle passes through the circle's center,
        so it is simply the line through p and center.
        """
        return Line.construct(p, self.center)

    def is_tangent(self, l: Line, *, tol: float = 1e-9) -> bool:
        """Return True if the line is tangent to the circle (within tolerance)."""
        d = l.distance(self.center)
        return _is_close(d, self.radius, rel_tol=tol, abs_tol=tol)

    def is_normal(self, l: Line, *, tol: float = 1e-9) -> bool:
        """
        Return True if the line passes through the circle's center (within tolerance).

        IMPORTANT
        ---------
        A line being a "normal to the circle" is only well-defined at a specific
        point of contact. This method keeps backward compatibility with the
        original code's behavior: it checks whether the line goes through the center.

        If you want the *normal at a specific point p*, use:
            l == circle.normal(p)   (or compare directionally)
        """
        # l(self.center) should evaluate A*h + B*k + C
        val = l(self.center)
        return _is_close(float(val), 0.0, rel_tol=tol, abs_tol=tol)

    # -------------------------------------------------------------------------
    # Representations / equations
    # -------------------------------------------------------------------------
    def equation(self) -> str:
        """
        Return the expanded standard-form equation string.

        For center (h,k) and radius r:
            (x - h)^2 + (y - k)^2 = r^2
        expands to:
            x^2 + y^2 - 2h x - 2k y + (h^2 + k^2 - r^2) = 0
        """
        h, k, r = self.center.x, self.center.y, self.radius
        const = (h ** 2) + (k ** 2) - (r ** 2)

        # Format cleanly with explicit signs
        # x^2 + y^2 + Ax + By + C = 0 where A = -2h, B = -2k
        A = -2.0 * h
        B = -2.0 * k
        C = const

        def _term(coeff: float, var: str) -> str:
            if _is_close(coeff, 0.0):
                return ""
            sign = " + " if coeff > 0 else " - "
            mag = abs(coeff)
            # Prefer integer-like display when possible
            if _is_close(mag, round(mag)):
                mag_str = str(int(round(mag)))
            else:
                mag_str = repr(mag)
            return f"{sign}{mag_str}*{var}"

        parts = ["x^2 + y^2"]
        parts.append(_term(A, "x"))
        parts.append(_term(B, "y"))

        # Constant term
        if not _is_close(C, 0.0):
            sign = " + " if C > 0 else " - "
            mag = abs(C)
            if _is_close(mag, round(mag)):
                mag_str = str(int(round(mag)))
            else:
                mag_str = repr(mag)
            parts.append(f"{sign}{mag_str}")

        return "".join(parts) + " = 0"

    def parametric_equation(
        self,
        theta_resolution: float = 0.01,
        semi: bool = False
    ) -> Generator[Tuple[float, float], None, None]:
        """
        Yield (x,y) points along the circle as theta advances.

        Parameters
        ----------
        theta_resolution : float
            Step size in radians (> 0).
        semi : bool
            If True, yield only a semicircle from 0 to pi (inclusive end not guaranteed).
            If False, yield full circle from 0 to 2*pi.

        Yields
        ------
        (x, y) : tuple[float, float]
        """
        if theta_resolution <= 0:
            raise ValueError("theta_resolution must be > 0.")

        limit = math.pi if semi else 2.0 * math.pi
        theta = 0.0
        # Use <= with small epsilon to avoid missing last step due to float accumulation
        eps = theta_resolution * 0.5

        while theta < limit + eps:
            x = self.center.x + self.radius * math.cos(theta)
            y = self.center.y + self.radius * math.sin(theta)
            yield (float(x), float(y))
            theta += theta_resolution

    # -------------------------------------------------------------------------
    # Sectors
    # -------------------------------------------------------------------------
    def sector_length(self, theta: float) -> float:
        """Return the arc length of a sector subtending theta radians at the center."""
        return self.radius * float(theta)

    def sector_area(self, theta: float) -> float:
        """Return the area of a sector subtending theta radians at the center."""
        return (self.radius ** 2) * float(theta) / 2.0

    # -------------------------------------------------------------------------
    # Circle-circle intersection (correct, complete handling)
    # -------------------------------------------------------------------------
    def intersection(
        self,
        other: "Circle",
        *,
        tol: float = 1e-9
    ) -> Union[None, Point, Tuple[Point, Point], str]:
        """
        Return the intersection(s) of this circle with another circle.

        Returns
        -------
        None
            No intersection
        Point
            Exactly one intersection point (tangent: external or internal)
        (Point, Point)
            Two intersection points
        str
            "infinite" for coincident circles (same center and radius)

        Notes
        -----
        Handles all standard circle-circle intersection cases:
        - Separate circles
        - One inside another (no intersection)
        - External tangency (1 point)
        - Internal tangency (1 point)
        - Two intersections
        - Coincident circles (infinite intersections)
        """
        if not isinstance(other, Circle):
            raise TypeError("intersection() requires another Circle.")

        c1 = self.center
        c2 = other.center
        r1 = self.radius
        r2 = other.radius

        dx = c2.x - c1.x
        dy = c2.y - c1.y
        d = math.hypot(dx, dy)

        # Coincident centers
        if _is_close(d, 0.0, rel_tol=tol, abs_tol=tol):
            if _is_close(r1, r2, rel_tol=tol, abs_tol=tol):
                return "infinite"
            return None

        # Non-intersecting cases
        if d > (r1 + r2) + tol:
            return None  # too far apart
        if d < abs(r1 - r2) - tol:
            return None  # one inside the other, no intersection

        # Compute intersection points using standard chord formula
        # a = distance from c1 to midpoint along line c1->c2
        a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)

        # h = half-chord length
        h2 = r1 * r1 - a * a
        if h2 < 0 and h2 > -tol:
            h2 = 0.0  # clamp tiny negatives due to numeric error
        if h2 < -tol:
            return None  # numeric safety; should not happen if cases above handled
        h = math.sqrt(max(0.0, h2))

        # Midpoint of chord
        xm = c1.x + a * dx / d
        ym = c1.y + a * dy / d

        # Offset vector perpendicular to (dx,dy)
        rx = -dy * (h / d)
        ry = dx * (h / d)

        p1 = Point(xm + rx, ym + ry)
        p2 = Point(xm - rx, ym - ry)

        # Tangency (one point) when h ~ 0
        if _is_close(h, 0.0, rel_tol=tol, abs_tol=tol):
            return p1

        return (p1, p2)

    # Backward-compatible misspelling (kept on purpose)
    def intersetion(self, other) -> Union[None, Point, Tuple[Point, Point], str]:
        """Backward compatible alias for intersection()."""
        return self.intersection(other)

    # -------------------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"

    def __str__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Circle):
            return self.center == other.center and self.radius == other.radius
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((self.center, self.radius))


def construct(p0: Point, p1: Point, p2: Point) -> Circle:
    """
    Return the unique circle passing through three non-collinear points.

    Raises
    ------
    AssertionError
        If points are collinear.
    """
    if geometry.colinear(p0, p1, p2):
        raise AssertionError("Circle cannot be constructed from three collinear points.")

    l1 = geometry.perpendicular_bisector(p0, p1)
    l2 = geometry.perpendicular_bisector(p1, p2)

    center = l1.intersection(l2)
    radius = geometry.distance(center, p0)

    return Circle(center, radius)
