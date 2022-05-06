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
from scipy.integrate import quad as integrate

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
    r"""
    Step Function [ u(t) ].

    Simple implimentation of numpy.heaviside function to provide standard
    step-function as specified to be zero at :math:`x < 0`, and one at
    :math:`x \geq 0`.

    Examples
    --------
    >>> import numpy as np
    >>> from electricpy.math import step
    >>> t = np.array([-10, -8, -5, -3, 0, 1, 2, 5, 7, 15])
    >>> x = step(t)
    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    Parameters
    ----------
    t:  arraylike
        Time samples for which the step response should be generated.
    """
    return (_np.heaviside(t, 1))

# Arbitrary Waveform RMS Calculating Function
def funcrms(func, T):
    """
    Root-Mean-Square (RMS) Evaluator for Callable Functions.

    Integral-based RMS calculator, evaluates the RMS value
    of a repetative signal (f) given the signal's specific
    period (T)

    Parameters
    ----------
    func:   float
            The periodic function, a callable like f(t)
    T:      float
            The period of the function f, so that f(0)==f(T)

    Returns
    -------
    RMS:    The RMS value of the function (f) over the interval ( 0, T )
    """
    fn = lambda x: func(x) ** 2
    integral, _ = integrate(fn, 0, T)
    return _np.sqrt(1 / T * integral)

# Define Gaussian Function
def gaussian(x, mu=0, sigma=1):
    """
    Gaussian Function.

    This function is designed to generate the gaussian
    distribution curve with configuration mu and sigma.

    Parameters
    ----------
    x:      float
            The input (array) x.
    mu:     float, optional
            Optional control argument, default=0
    sigma:  float, optional
            Optional control argument, default=1

    Returns
    -------
    Computed gaussian (numpy.ndarray) of the input x
    """
    return (1 / (sigma * _np.sqrt(2 * _np.pi)) *
            _np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

# Define Gaussian Distribution Function
def gausdist(x, mu=0, sigma=1):
    """
    Gaussian Distribution Function.

    This function is designed to calculate the generic
    distribution of a gaussian function with controls
    for mu and sigma.

    Parameters
    ----------
    x:      numpy.ndarray
            The input (array) x
    mu:     float, optional
            Optional control argument, default=0
    sigma:  float, optional
            Optional control argument, default=1

    Returns
    -------
    F:      numpy.ndarray
            Computed distribution of the gausian function at the
            points specified by (array) x
    """
    # Define Integrand
    def integrand(sq):
        return (_np.exp(-sq ** 2 / 2))
    try:
        lx = len(x)  # Find length of Input
    except:
        lx = 1  # Length 1
        x = [x]  # Pack into list
    F = _np.zeros(lx, dtype=_np.float64)
    for i in range(lx):
        x_tmp = x[i]
        # Evaluate X (altered by mu and sigma)
        X = (x_tmp - mu) / sigma
        integral = integrate(integrand, _np.NINF, X)  # Integrate
        result = 1 / _np.sqrt(2 * _np.pi) * integral[0]  # Evaluate Result
        F[i] = result
    # Return only the 0-th value if there's only 1 value available
    if (len(F) == 1):
        F = F[0]
    return (F)

# Define Probability Density Function
def probdensity(func, x, x0=0, scale=True):
    """
    Probability Density Function.

    This function uses an integral to compute the probability
    density of a given function.

    Parameters
    ----------
    func:   function
            The function for which to calculate the PDF
    x:      numpy.ndarray
            The (array of) value(s) at which to calculate
            the PDF
    x0:     float, optional
            The lower-bound of the integral, starting point
            for the PDF to be calculated over, default=0
    scale:  bool, optional
            The scaling to be applied to the output,
            default=True

    Returns
    -------
    sumx:   numpy.ndarray
            The (array of) value(s) computed as the PDF at
            point(s) x
    """
    sumx = _np.array([])
    try:
        lx = len(x)  # Find length of Input
    except:
        lx = 1  # Length 1
        x = [x]  # Pack into list
    # Recursively Find Probability Density
    for i in range(lx):
        sumx = _np.append(sumx, integrate(func, x0, x[i])[0])
    # Return only the 0-th value if there's only 1 value available
    if (len(sumx) == 1):
        sumx = sumx[0]
    else:
        if (scale == True):
            mx = sumx.max()
            sumx /= mx
        elif (scale != False):
            sumx /= scale
    return (sumx)

# Define Real FFT Evaluation Function
def rfft(arr, dt=0.01, absolute=True, resample=True):
    """
    RFFT Function.

    This function is designed to evaluat the real FFT
    of a input signal in the form of an array or list.

    Parameters
    ----------
    arr:        numpy.ndarray
                The input array representing the signal
    dt:         float, optional
                The time-step used for the array,
                default=0.01
    absolute:   bool, optional
                Control argument to force absolute
                values, default=True
    resample:   bool, optional
                Control argument specifying whether
                the FFT output should be resampled,
                or if it should have a specific
                resampling rate, default=True

    Returns
    -------
    FFT Array
    """
    # Calculate with Absolute Values
    if absolute:
        fourier = abs(_np.fft.rfft(arr))
    else:
        fourier = _np.fft.rfft(arr)
    if resample == True:
        # Evaluate the Downsampling Ratio
        dn = int(dt * len(arr))
        # Downsample to remove unnecessary points
        fixedfft = filter.dnsample(fourier, dn)
        return (fixedfft)
    elif resample == False:
        return (fourier)
    else:
        # Condition Resample Value
        resample = int(resample)
        # Downsample to remove unnecessary points
        fixedfft = filter.dnsample(fourier, resample)
        return (fixedfft)