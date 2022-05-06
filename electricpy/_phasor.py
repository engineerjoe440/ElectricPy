################################################################################
"""Test Phasor Class."""
################################################################################

# Import Required Packages
import numpy as _np
import cmath as _c

class Phasor(complex):
    """
    Phasor Class - An extension of the Python complex type for scientific work.

    .. warn::
       This module is still under development and should be used with caution.

       Its interfaces may change at any time without notice, its primary goal is
       exploritory work, and may be experimented with, used (carefully), and may
       be used to solicit feedback by way of project issues in Github:
       https://github.com/engineerjoe440/ElectricPy/issues

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
    complex:    complex
                The complex type-cast of the phasor.
    """

    __magnitude: float
    __angle: float

    def __init__(self, magnitude, angle=None):
        """
        Phasor Constructor.
        
        Parameters
        ----------
        magnitude:  float
                    The phasor magnitude to express.
        angle:      float
                    The phasor angle to express, in degrees.
        """
        # Handle Passing a Complex Type Directly to Phasor
        if isinstance(magnitude, complex):
            magnitude, ang_r = _c.polar(magnitude)
            angle = _np.degrees(ang_r)
        # Load the Internal Values
        self.__magnitude = magnitude
        self.__angle = angle

    @property
    def real(self):
        """Real ( RE{} ) evaluation."""
        return self.__magnitude * _np.cos(_np.radians(self.__angle))

    @property
    def imag(self):
        """Imaginary ( IM{} ) evaluation."""
        return self.__magnitude * _np.sin(_np.radians(self.__angle))

    @property
    def mag(self):
        """Phasor magnitude evaluation."""
        return self.__magnitude

    @property
    def ang(self):
        """Phasor angle evaluation."""
        return self.__angle
    
    @property
    def complex(self):
        """Phasor representation as complex."""
        return complex(self.real, self.imag)
    
    def __repr__(self):
        """Represent the Phasor."""
        return "Phasor(magnitude={mag}, angle={ang})".format(
            mag=self.__magnitude,
            ang=self.__angle
        )
    
    def __gt__(self, __x):
        """Evaluate whether __x is greater than this."""
        # Compare Magnitudes for Phasor Types
        if isinstance(__x, Phasor):
            return self.__magnitude.__gt__(__x.__magnitude)
        # Compare Magnitudes after first Casting `complex` to Phasor
        elif isinstance(__x, complex):
            return self.__gt__(Phasor(__x))
        else:
            return self.__magnitude.__gt__(__x)
    
    def __lt__(self, __x):
        """Evaluate whether __x is less than this."""
        # Compare Magnitudes for Phasor Types
        if isinstance(__x, Phasor):
            return self.__magnitude.__lt__(__x.__magnitude)
        # Compare Magnitudes after first Casting `complex` to Phasor
        elif isinstance(__x, complex):
            return self.__lt__(Phasor(__x))
        else:
            return self.__magnitude.__lt__(__x)
    
    def __ge__(self, __x):
        """Evaluate whether __x is greater than or equal to this."""
        # Compare Magnitudes for Phasor Types
        if isinstance(__x, Phasor):
            return self.__magnitude.__ge__(__x.__magnitude)
        # Compare Magnitudes after first Casting `complex` to Phasor
        elif isinstance(__x, complex):
            return self.__ge__(Phasor(__x))
        else:
            return self.__magnitude.__ge__(__x)
    
    def __le__(self, __x):
        """Evaluate whether __x is less than or equal to this."""
        # Compare Magnitudes for Phasor Types
        if isinstance(__x, Phasor):
            return self.__magnitude.__le__(__x.__magnitude)
        # Compare Magnitudes after first Casting `complex` to Phasor
        elif isinstance(__x, complex):
            return self.__le__(Phasor(__x))
        else:
            return self.__magnitude.__le__(__x)
    
    def __str__(self):
        """Stringlify the Phasor."""
        return "{} ∠ {}°".format(self.__magnitude, self.__angle)
    
    def __round__(self, ndigits=0):
        """Round the Phasor."""
        mag = self.__magnitude
        ang = self.__angle
        mag = round(mag, ndigits=ndigits)
        ang = round(ang, ndigits=ndigits)
        return Phasor(mag, ang)
    
    def __mul__(self, __x):
        """Return self*__x."""
        if isinstance(__x, Phasor):
            return self.complex.__mul__(__x.complex)
        else:
            return self.complex.__mul__(__x)
        
    def __truediv__(self, __x):
        """Return self/__x."""
        if isinstance(__x, Phasor):
            return self.complex.__mul__(__x.complex)
        else:
            return self.complex.__mul__(__x)
    
    def __abs__(self):
        """Return the absolute magnitude."""
        return self.__magnitude
    
    def from_arraylike(arraylike):
        """
        Phasor Constructor for Casting from Array Like Object.
        
        Use this method to create a new Phasor object from a two-item (2) long
        arraylike object, (e.g., a `tuple`, `list`, or NumPy array).
        """
        return Phasor(*arraylike)


if __name__ == "__main__":
    a = Phasor(67, 120)
    print(a)
    print(a.real)
    print(a.imag)
    print(a.__repr__())
    x = Phasor.from_arraylike((67, 120))
    print(x)
    print(x.real)
    print(x.__repr__())

    y = complex(1+1j)
    print(y)
    z = Phasor(y)
    print(z)

    print("mult", Phasor(y*z))
    print("mult", Phasor(a.complex * z.complex))
    print("mult", Phasor(a * z))

    print(round(Phasor(y+z)))

    print(x > z)
    print(z > x)
    print(a <= z)
    print(abs(a))

    print(complex(y))
