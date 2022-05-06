################################################################################
"""
`electricpy` Package - `phasors` Module.

>>> from electricpy import phasors

Filled with calculators, evaluators, and plotting functions related to
electrical phasors, this package will provide a wide array of capabilities to
any electrical engineer.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

# Import Required Packages
import numpy as _np
import cmath as _c
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt


# Define Phase Angle Generator
def phs(ang):
    """
    Complex Phase Angle Generator.

    Generate a complex value given the phase angle
    for the complex value.

    Same as `phase`.

    Parameters
    ----------
    ang:        float
                The angle (in degrees) for which
                the value should be calculated.

    See Also
    --------
    electricpy.cprint:              Complex Variable Printing Function
    electricpy.phasor.phasorlist:   Phasor Generator for List or Array
    electricpy.phasor.phasorz:      Impedance Phasor Generator
    electricpy.phasor.phasor:       Phasor Generating Function
    """
    # Return the Complex Angle Modulator
    return (_np.exp(1j * _np.radians(ang)))


phase = phs  # Create Duplicate Name


# Define Phasor Generator
def phasor(mag, ang=0):
    """
    Complex Phasor Generator.

    Generates the standard Pythonic complex representation
    of a phasor voltage or current when given the magnitude
    and angle of the specific voltage or current.

    Parameters
    ----------
    mag:        float
                The Magnitude of the Voltage/Current
    ang:        float
                The Angle (in degrees) of the Voltage/Current

    Returns
    -------
    phasor:     complex
                Standard Pythonic Complex Representation of
                the specified voltage or current.

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.phasor(67, 120) # 67 volts at angle 120 degrees
    (-33.499999999999986+58.02370205355739j)

    See Also
    --------
    electricpy.cprint:              Complex Variable Printing Function
    electricpy.phasor.phasorlist:   Phasor Generator for List or Array
    electricpy.phasor.phasorz:      Impedance Phasor Generator
    electricpy.phasor.phs:          Complex Phase Angle Generator
    """
    # Test for Tuple/List Arg
    if isinstance(mag, (tuple, list, _np.ndarray)):
        ang = mag[1]
        mag = mag[0]
    return (_c.rect(mag, _np.radians(ang)))


# Define Impedance Conversion function
def phasorz(C=None, L=None, freq=60, complex=True):
    r"""
    Phasor Impedance Generator.

    This function's purpose is to generate the phasor-based
    impedance of the specified input given as either the
    capacitance (in Farads) or the inductance (in Henreys).
    The function will return the phasor value (in Ohms).

    .. math:: Z = \frac{-j}{\omega*C}

    .. math:: Z = j*\omega*L

    where:

    .. math:: \omega = 2*\pi*freq

    Parameters
    ----------
    C:          float, optional
                The capacitance value (specified in Farads),
                default=None
    L:          float, optional
                The inductance value (specified in Henreys),
                default=None
    freq:       float, optional
                The system frequency to be calculated upon, default=60
    complex:    bool, optional
                Control argument to specify whether the returned
                value should be returned as a complex value.
                default=True

    Returns
    -------
    Z:      complex
            The ohmic impedance of either C or L (respectively).
    """
    w = 2 * _np.pi * freq
    # C Given in ohms, return as Z
    if (C != None):
        Z = -1 / (w * C)
    # L Given in ohms, return as Z
    if (L != None):
        Z = w * L
    # If asked for imaginary number
    if (complex):
        Z *= 1j
    return Z


# Define Phasor Array Generator
def phasorlist(arr):
    """
    Complex Phasor Generator for 2-D Array or 2-D List.

    Generates the standard Pythonic complex representation
    of a phasor voltage or current when given the magnitude
    and angle of the specific voltage or current for a list
    or array of values.

    Parameters
    ----------
    arr:        array-like
                2-D array or list of magnitudes and angles.
                Each item must be set of magnitude and angle
                in form of: [mag, ang].

    Returns
    -------
    phasor:     complex
                Standard Pythonic Complex Representation of
                the specified voltage or current.

    Examples
    --------
    >>> import numpy as np
    >>> from electricpy import phasor
    >>> voltages = np.array([
    ...     [67,0],
    ...     [67,-120],
    ...     [67,120]
    ... ])
    >>> Vset = phasor.phasorlist( voltages )
    >>> print(Vset)

    See Also
    --------
    electricpy.cprint:              Complex Variable Printing Function
    electricpy.phasor.phasor:       Phasor Generating Function
    electricpy.phasor.vectarray:    Magnitude/Angle Array Pairing Function
    electricpy.phasor.phasorz:      Impedance Phasor Generator
    """
    # Use List Comprehension to Process

    # Return Array
    return _np.array([phasor(i) for i in arr])

# Define Vector Array Generator
def vectarray(arr, degrees=True, flatarray=False):
    """
    Format Complex as Array of Magnitude/Angle Pairs.

    Consume an iterable (list/tuple/ndarray/etc.) of
    complex numbers and generate an ndarray of magnitude
    and angle pairs, formatted as a 2-dimension (or
    optionally 1-dimension) array.

    Parameters
    ----------
    arr:        array-like
                Array or list of complex numbers to be
                cast to magnitude/angle pairs.
    degrees:    bool, optional
                Control option to set the angles in
                degrees. Defaults to True.
    flatarray:  bool, optional
                Control option to set the array return
                to work as a 1-dimension list. Defaults
                to False, formatting as a 2-dimension
                list.

    Returns
    -------
    polararr:   ndarray
                Array of magnitude/angle pairs as a
                2-dimension array (or optionally
                1-dimension array).

    See Also
    --------
    electricpy.phasor.phasor:       Phasor Generating Function
    electricpy.phasor.phasorlist:   Phasor Generator for List or Array
    """
    # Iteratively Append Arrays to the Base

    def vector_cast(num):
        mag, ang = _c.polar(num)

        if degrees:
            ang = _np.degrees(ang)

        return [mag, ang]

    polararr = _np.array([vector_cast(num) for num in arr])
    # Reshape Array if Needed
    if not flatarray:
        polararr = _np.reshape(polararr, (-1, 2))
    return polararr

# Define Phasor Data Generator
def phasordata(mn, mx=None, npts=1000, mag=1, ang=0, freq=60,
               retstep=False, rettime=False, sine=False):
    """
    Complex Phasor Data Generator.

    Generates a sinusoidal data set with minimum, maximum,
    frequency, magnitude, and phase angle arguments.

    Parameters
    ----------
    mn:         float, optional
                Minimum time (in seconds) to generate data for.
                default=0
    mx:         float
                Maximum time (in seconds) to generate data for.
    npts:       float, optional
                Number of data samples. default=1000
    mag:        float, optional
                Sinusoid magnitude, default=1
    ang:        float, optional
                Sinusoid angle in degrees, default=0
    freq:       float, optional
                Sinusoid frequency in Hz
    retstep:    bool, optional
                Control argument to request return of time
                step size (dt) in seconds.
    sine:       bool, optional
                Control argument to require data be generated
                with a sinusoidal function instead of cosine.

    Returns
    -------
    data:       numpy.ndarray
                The resultant data array.
    """
    # Test Inputs for Min/Max
    if mx == None:
        # No Minimum provided, use Value as Maximum
        mx = mn
        mn = 0
    # Generate Omega
    w = 2 * _np.pi * freq
    # Generate Time Array
    t, dt = _np.linspace(mn, mx, npts, retstep=True)
    # Generate Data Array
    if not sine:
        data = mag * _np.cos(w * t + _np.radians(ang))
    else:
        data = mag * _np.sin(w * t + _np.radians(ang))
    # Generate Return Data Set
    dataset = [data]
    if retstep:
        dataset.append(dt)
    if rettime:
        dataset.append(t)
    # Return Dataset
    if len(dataset) == 1:
        return (dataset[0])
    return (dataset)

# Define Complex Composition Function
def compose(*arr):
    """
    Complex Composition Function.

    Accepts a set of real values and generates an array
    of complex values. Input must be array-like, but can
    appear in various forms:

    - [ real, imag]
    - [ [ real1, ..., realn ], [ imag1, ..., imagn ] ]
    - [ [ real1, imag1 ], ..., [ realn, imagn ] ]

    Will always return values in form:

    [ complex1, ... complexn ]

    Parameters
    ----------
    arr:        array_like
                The input of real and imaginary term(s)
    """
    # Condition Input
    if len(arr) == 1:
        arr = arr[0]  # Extract 0-th term
    # Input comes in various forms, we must first detect shape
    arr = _np.asarray(arr)  # Format as Numpy Array
    # Gather Shape to Detect Format
    try:
        row, col = arr.shape
        # Passed Test, Valid Shape
        retarr = _np.array([])  # Empty Return Array
        # Now, Determine whether is type 2 or 3
        if col == 2:  # Type 3
            for i in range(row):  # Iterate over each row
                item = arr[i][0] + 1j * arr[i][1]
                retarr = _np.append(retarr, item)
        elif row == 2:  # Type 2
            for i in range(col):  # Iterate over each column
                item = arr[0][i] + 1j * arr[1][i]
                retarr = _np.append(retarr, item)
        else:
            raise ValueError("Invalid Array Shape, must be 2xN or Nx2.")
        # Successfully Generated Array, Return
        return (retarr)
    except:  # 1-Dimension Array
        length = arr.size
        # Test for invalid Array Size
        if length != 2:
            raise ValueError("Invalid Array Size, Saw Length of " + str(length))
        # Valid Size, Calculate and Return
        return (arr[0] + 1j * arr[1])


# Define Parallel Impedance Adder
def parallelz(*args):
    r"""
    Parallel Impedance Calculator.

    This function is designed to generate the total parallel
    impedance of a set (tuple) of impedances specified as real
    or complex values.

    .. math::
       Z_{eq}=(\frac{1}{Z_1}+\frac{1}{Z_2}+\dots+\frac{1}{Z_n})^{-1}

    Parameters
    ----------
    Z:      tuple of complex
            The tupled input set of impedances, may be a tuple
            of any size greater than 2. May be real, complex, or
            a combination of the two.

    Returns
    -------
    Zp:     complex
            The calculated parallel impedance of the input tuple.
    """
    # Gather length (number of elements in tuple)
    L = len(args)
    if L == 1:
        Z = args[0]  # Only One Tuple Provided
        try:
            L = len(Z)
            if (L == 1):
                Zp = Z[0]  # Only one impedance, burried in tuple
            else:
                # Inversely add the first two elements in tuple
                Zp = (1 / Z[0] + 1 / Z[1]) ** (-1)
                # If there are more than two elements, add them all inversely
                if (L > 2):
                    for i in range(2, L):
                        Zp = (1 / Zp + 1 / Z[i]) ** (-1)
        except:
            Zp = Z  # Only one impedance
    else:
        Z = args  # Set of Args acts as Tuple
        # Inversely add the first two elements in tuple
        Zp = (1 / Z[0] + 1 / Z[1]) ** (-1)
        # If there are more than two elements, add them all inversely
        if (L > 2):
            for i in range(2, L):
                Zp = (1 / Zp + 1 / Z[i]) ** (-1)
    return (Zp)

# Define Phasor Plot Generator
def phasorplot(phasor, title="Phasor Diagram", legend=False, bg=None,
            colors=None, radius=None, linewidth=None, size=None,
            label=False, labels=False, tolerance=None):
    """
    Phasor Plotting Function.

    This function is designed to plot a phasor-diagram with angles in degrees
    for up to 12 phasor sets (more may be used if additional colors are set).
    Phasors must be passed as a complex number set, (e.g.
    [ m+ja, m+ja, m+ja, ... , m+ja ] ).

    Examples
    --------
    >>> import numpy as np
    >>> from electricpy import phasor
    >>> voltages = np.array([
    ...     [67,0],
    ...     [45,-120],
    ...     [52,120]
    ... ])
    >>> plt = phasor.phasorlist( voltages, colors=["red", "green", "blue"] )
    >>> plt.show()

    .. image:: /static/PhasorPlot.png

    Parameters
    ----------
    phasor:     list of complex
                The set of phasors to be plotted.
    title:      string, optional
                The Plot Title, default="Phasor Diagram"
    legend:     bool, optional
                Control argument to enable displaying the legend, must be passed
                as an array or list of strings. `label` and `labels` are mimic-
                arguments and will perform similar operation, default=False
    bg:         string, optional
                Background-Color control, default="#d5de9c"
    radius:     float, optional
                The diagram radius, unless specified, automatically scales
    colors:     list of str, optional
                List of hexidecimal color strings denoting the line colors to
                use.
    size:       float, optional
                Control argument for figure size. default=None
    linewidth:  float, optional
                Control argument to declare the line thickness. default=None
    tolerance:  float, optional
                Minimum magnitude to plot, anything less than tolerance will be
                plotted as a single point at the origin, by default, the
                tolerance is scaled to be 1/25-th the maximum radius. To disable
                the tolerance, simply provide either False or -1.
    
    Returns
    -------
    matplotlib.pyplot:  Plotting object to be used for additional configuration
                        or plotting.
    """
    # Load Complex Values if Necessary
    try:
        len(phasor)
    except TypeError:
        phasor = [phasor]
    # Manage Colors
    if colors == None:
        colors = [
            "#FF0000", "#800000", "#FFFF00", "#808000", "#00ff00","#008000",
            "#00ffff", "#008080", "#0000ff", "#000080", "#ff00ff", "#800080"
        ]
    # Scale Radius
    if radius == None:
        radius = _np.abs(phasor).max()
    # Set Tolerance
    if tolerance == None:
        tolerance = radius / 25
    elif tolerance == False:
        tolerance = -1
    # Set Background Color
    if bg == None:
        bg = "#FFFFFF"
    # Load labels if handled in other argument
    if label != False:
        legend = label
    if labels != False:
        legend = labels
    # Check for more phasors than colors
    numphs = len(phasor)
    numclr = len(colors)
    if numphs > numclr:
        raise ValueError(
            "ERROR: Too many phasors provided. Specify more line colors."
        )

    if size == None:
        # Force square figure and square axes
        width, height = _matplotlib.rcParams['figure.figsize']
        size = min(width, height)
    # Make a square figure
    fig = _plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, facecolor=bg)
    _plt.grid(True)

    # Plot the diagram
    _plt.title(title + "\n")
    handles = _np.array([])  # Empty array for plot handles
    for i in range(numphs):
        mag, ang_r = _c.polar(phasor[i])
        # Plot with labels
        if legend != False:
            if mag > tolerance:
                hand = _plt.arrow(0, 0, ang_r, mag, color=colors[i],
                                label=legend[i], linewidth=linewidth)
            else:
                hand = _plt.plot(0, 0, 'o', markersize=linewidth * 3,
                                label=legend[i], color=colors[i])
            handles = _np.append(handles, [hand])
        # Plot without labels
        else:
            _plt.arrow(0, 0, ang_r, mag, color=colors[i], linewidth=linewidth)
    if legend != False: _plt.legend((handles), legend)
    # Set Minimum and Maximum Radius Terms
    ax.set_rmax(radius)
    ax.set_rmin(0)
    return _plt

# END
