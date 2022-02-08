################################################################################
"""
`electricpy` Package - `math` Module.

>>> from electricpy import math as epmath

Focussed on simplifying common mathematic formulas for electrical engineering,
this module exposes a few common functions like convolution, step-functions,
etc.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

# Import Required Packages
import numpy as _np
import scipy.signal as _sig


# Define convolution function
def convolve(tuple):
    """
    Filter Convolution Function.

    Given a tuple of terms, convolves all terms in tuple to
    return one tuple as a numpy array.

    Parameters
    ----------
    tuple:      tuple of numpy.ndarray
                Tuple of terms to be convolved.

    Returns
    -------
    c:          The convolved set of the individual terms.
                i.e. numpy.ndarray([ x1, x2, x3, ..., xn ])
    """
    c = _sig.convolve(tuple[0], tuple[1])
    if (len(tuple) > 2):
        # Iterate starting with second element and continuing
        for i in range(2, len(tuple)):
            c = _sig.convolve(c, tuple[i])
    return (c)


# Define Step function
def step(t):
    """
    Step Function [ u(t) ].

    Simple implimentation of numpy.heaviside function
    to provide standard step-function as specified to
    be zero at x<0, and one at x>=0.
    """
    return (_np.heaviside(t, 1))