################################################################################
"""Test Phasor Class."""
################################################################################

# Import Required Packages
import sys as _sys
import numpy as _np
import cmath as _c

class Phasor(complex):
    """
    Phasor Class - An extension of the Python complex type for scientific work.

    This class can be used in place of, or alongside the standard `complex` type
    in Python to represent complex numbers as phasors (magnitude and angle) for
    scientific computation and electrical engineering analysis and work.

    Examples
    --------
    >>> from electricpy._phasor import Phasor
    >>> Phasor(67, 120) # 67 volts at angle 120 degrees
    Phasor(magnitude=67, angle=120)
    >>> volt = Phasor(67, 120)
    >>> print(volt)
    67 ∠ 120°
        
    Parameters
    ----------
    magnitude:  float
                The phasor magnitude to express.
    angle:      float
                The phasor angle to express, in degrees.

    Properties
    ----------
    mag:        float
                The phasor magnitude, also commonly known as its absolute value.
    ang:        float
                The phasor angle, expressed in degrees.
    real:       float
                The real component of the complex value.
    imag:       float
                The imaginary component of the complex value.
    """

    def __new__(self, magnitude, angle=None):
        """
        Phasor Constructor.
        """
        # Handle Passing a Complex Type Directly to Phasor
        if isinstance(magnitude, complex):
            magnitude, ang_r = _c.polar(magnitude)
            angle = _np.degrees(ang_r)
        return complex.__new__(
            self,
            real=(magnitude * _np.cos(_np.radians(angle))),
            imag=(magnitude * _np.sin(_np.radians(angle)))
        )

    @property
    def mag(self):
        """Phasor magnitude evaluation."""
        return _np.absolute(self)

    @property
    def ang(self):
        """Phasor angle evaluation in degrees."""
        return _np.degrees(_np.angle(self))
    
    def __repr__(self):
        """Represent the Phasor."""
        return f"Phasor(magnitude={self.mag}, angle={self.ang})"
    
    def __str__(self):
        """Stringlify the Phasor."""
        angle_denotion = "∠"
        if _sys.stdout.encoding != "utf-8":
            angle_denotion = "/_"
        return f"{self.mag} {angle_denotion} {self.ang}°"
    
    def __round__(self, ndigits=0):
        """Round the Phasor."""
        return Phasor(
            round(self.mag, ndigits=ndigits),
            round(self.ang, ndigits=ndigits)
        )
    
    def from_arraylike(arraylike):
        """
        Phasor Constructor for Casting from Array Like Object.
        
        Use this method to create a new Phasor object from a two-item (2) long
        arraylike object, (e.g., a `tuple`, `list`, or NumPy array).
        """
        return Phasor(*arraylike)
