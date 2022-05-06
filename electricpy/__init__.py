################################################################################
"""
`electricpy` Package - Main Module.

>>> import electricpy as ep

Filled with calculators, evaluators, and plotting functions, this package will
provide a wide array of capabilities to any electrical engineer.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

# Define Module Specific Variables
_name_ = "electricpy"
_version_ = "0.2.2"
__version__ = _version_  # Alias Version for User Ease

# Version Breakdown:
# MAJOR CHANGE . MINOR CHANGE . MICRO CHANGE

# Import Submodules
from .constants import *
from .phasor import compose, parallelz
from .phasor import phasorz as impedance

# Import Supporting Modules
import numpy as _np
import matplotlib.pyplot as _plt
import cmath as _c
from scipy.optimize import fsolve as _fsolve
from warnings import showwarning as _showwarning
from inspect import getframeinfo as _getframeinfo
from inspect import stack as _stack
from scipy.integrate import quad as integrate


# Define Cycle Time Function
def tcycle(ncycles=1, freq=60):
    r"""
    Time of Electrical Cycles.

    Evaluates the time for a number of n
    cycles given the system frequency.

    .. math:: t = \frac{n_{cycles}}{freq}

    Parameters
    ----------
    ncycles:    float, optional
                Number (n) of cycles to evaluate, default=1
    freq:       float, optional
                System frequency in Hz, default=60

    Returns
    -------
    t:          float
                Total time for *ncycles*
    """
    # Condition Inputs
    ncycles = _np.asarray(ncycles)
    freq = _np.asarray(freq)
    # Evaluate the time for ncycles
    time = ncycles / freq
    # Return
    if len(time) == 1:
        return (time[0])
    else:
        return (time)


# Define Reactance Calculator
def reactance(z, freq=60, sensetivity=1e-12):
    r"""
    Capacitance/Inductance from Impedance.

    Calculates the Capacitance or Inductance in Farads or Henreys
    (respectively) provided the impedance of an element.
    Will return capacitance (in Farads) if ohmic impedance is
    negative :eq:`cap`, or inductance (in Henrys) if ohmic impedance is
    positive :eq:`ind`. If imaginary: calculate with j factor
    (imaginary number).

    .. math:: C = \frac{1}{\omega*Z}
       :label: cap

    .. math:: L = \frac{Z}{\omega}
       :label: ind

    This requires that the radian frequency is found as follows:

    .. math:: \omega = 2*\pi*freq

    where `freq` is the frequency in Hertz.

    .. note::
       It's worth noting here, that the resistance will be found by
       extracting the real part of a complex value. That is:

       .. math:: R = Real( R + jX )


    Parameters
    ----------
    z:              complex
                    The Impedance Provided, may be complex (R+jI)
    freq:           float, optional
                    The Frequency Base for Provided Impedance, default=60
    sensetivity:    float, optional
                    The sensetivity used to check if a resistance was
                    provided, default=1e-12

    Returns
    -------
    out:            float
                    Capacitance or Inductance of Impedance
    """
    # Evaluate Omega
    w = 2 * _np.pi * freq
    # Input is Complex
    if isinstance(z, complex):
        # Test for Resistance
        if (abs(z.real) > sensetivity):
            R = z.real
        else:
            R = 0
        if (z.imag > 0):
            out = z / (w * 1j)
        else:
            out = 1 / (w * 1j * z)
        out = abs(out)
        # Combine with resistance if present
        if (R != 0): out = (R, out)
    else:
        if (z > 0):
            out = z / (w)
        else:
            out = 1 / (w * z)
        out = abs(out)
    # Return Output
    return (out)


# Define display function
def cprint(val, unit=None, label=None, title=None,
           pretty=True, printval=True, ret=False, decimals=3, round=3):
    """
    Phasor (Complex) Printing Function.

    This function is designed to accept a complex value (val) and print
    the value in the standard electrical engineering notation:

    **magnitude ∠ angle °**

    This function will print the magnitude in degrees, and can print
    a unit and label in addition to the value itself.

    Parameters
    ----------
    val:        complex
                The Complex Value to be Printed, may be singular value,
                tuple of values, or list/array.
    unit:       string, optional
                The string to be printed corresponding to the unit mark.
    label:      str, optional
                The pre-pended string used as a descriptive labeling string.
    title:      str, optional
                The pre-pended string describing a set of complex values.
    pretty:     bool, optional
                Control argument to force printed result to a *pretty*
                format without array braces. default=True
    printval:   bool, optional
                Control argument enabling/disabling printing of the string.
                default=True
    ret:        bool, optional
                Control argument allowing the evaluated value to be returned.
                default=False
    decimals:   int, optional
                Replaces `round` argument. Control argument specifying how
                many decimals of the complex value to be printed. May be
                negative to round to spaces to the left of the decimal place
                (follows standard round() functionality). default=3
    round:      int, optional, DEPRECATED
                Control argument specifying how many decimals of the complex
                value to be printed. May be negative to round to spaces
                to the left of the decimal place (follows standard round()
                functionality). default=3

    Returns
    -------
    numarr:     numpy.ndarray
                The array of values corresponding to the magnitude and angle,
                values are returned in the form: [[ mag, ang ],...,[ mag, ang ]]
                where the angles are evaluated in degrees.

    Examples
    --------
    >>> import electricpy as ep
    >>> v = ep.phasor(67, 120)
    >>> ep.cprint(v)
    67.0 ∠ 120.0°
    >>> voltages = np.array([[67,0],
                             [67,-120],
                             [67,120]])
    >>> Vset = ep.phasorlist( voltages )
    >>> ep.cprint(Vset)
    [['67.0 ∠ 0.0°']
    ['67.0 ∠ -120.0°']
    ['67.0 ∠ 120.0°']]


    See Also
    --------
    electricpy.phasor.phasor:       Phasor Generating Function
    electricpy.phasor.phasorlist:   Phasor Generating Function for Lists/Arrays
    electricpy.phasor.phasorz:      Impedance Phasor Generator
    """
    # Use depricated `round`
    if round != 3:
        decimals = round
        caller = _getframeinfo(_stack()[1][0])
        # Demonstrate Deprecation Warning
        _showwarning('`round` argument will be deprecated in favor of `decimals`',
                     DeprecationWarning, caller.filename, caller.lineno)
    # Interpret as numpy array if simple list
    if isinstance(val, list):
        val = _np.asarray(val)  # Ensure that input is array
    # Find length of the input array
    if isinstance(val, _np.ndarray):
        shp = val.shape
        try:
            row, col = shp  # Interpret Shape of Object
        except:
            row = shp[0]
            col = 1
        sz = val.size
        # Handle Label as a List or Array
        if isinstance(label, (list, _np.ndarray)):
            if len(label) == 1:
                tmp = label
                for _ in range(sz):
                    label = _np.append(label, [tmp])
            elif sz != len(label):
                raise ValueError("Too Few Label Arguments")
        # Handle Label as String
        elif isinstance(label, str):
            tmp = label
            for _ in range(sz):
                label = _np.append(label, [tmp])
        # Handle Lack of Label
        elif label == None:
            label = _np.array([])
            for _ in range(sz):
                label = _np.append(label, None)
        # Handle all Other Cases
        else:
            raise ValueError("Invalid Label")
        # Handle Unit as a List or Array
        if isinstance(unit, (list, _np.ndarray)):
            if len(unit) == 1:
                tmp = unit
                for _ in range(sz):
                    unit = _np.append(unit, [tmp])
            elif sz != len(unit):
                raise ValueError("Too Few Unit Arguments")
        # Handle Unit as String
        elif isinstance(unit, str):
            tmp = unit
            for _ in range(sz):
                unit = _np.append(unit, [tmp])
        # Handle Lack of Unit
        elif unit == None:
            unit = _np.array([])
            for _ in range(sz):
                unit = _np.append(unit, None)
        # Handle all Other Cases
        else:
            raise ValueError("Invalid Unit")
        # Generate Default Arrays
        printarr = _np.array([])  # Empty array
        numarr = _np.array([])  # Empty array
        # Operate on List/Array
        for i in range(row):
            _val = val[i]
            _label = label[i]
            _unit = unit[i]
            mag, ang_r = _c.polar(_val)  # Convert to polar form
            ang = _np.degrees(ang_r)  # Convert to degrees
            mag = _np.around(mag, decimals)  # Round
            ang = _np.around(ang, decimals)  # Round
            strg = ""
            if _label != None:
                strg += _label + " "
            strg += str(mag) + " ∠ " + str(ang) + "°"
            if _unit != None:
                strg += " " + _unit
            printarr = _np.append(printarr, strg)
            numarr = _np.append(numarr, [mag, ang])
        # Reshape Arrays
        printarr = _np.reshape(printarr, (row, col))
        numarr = _np.reshape(numarr, (sz, 2))
        # Print
        if printval and row == 1:
            if title != None:
                print(title)
            print(strg)
        elif printval and pretty:
            strg = ''
            start = True
            for i in printarr:
                if not start:
                    strg += '\n'
                strg += str(i[0])
                start = False
            if title != None:
                print(title)
            print(strg)
        elif printval:
            if title != None:
                print(title)
            print(printarr)
        # Return if Necessary
        if ret:
            return (numarr)
    elif isinstance(val, (int, float, complex)):
        # Handle Invalid Unit/Label
        if unit != None and not isinstance(unit, str):
            raise ValueError("Invalid Unit Type for Value")
        if label != None and not isinstance(label, str):
            raise ValueError("Invalid Label Type for Value")
        mag, ang_r = _c.polar(val)  # Convert to polar form
        ang = _np.degrees(ang_r)  # Convert to degrees
        mag = _np.around(mag, decimals)  # Round
        ang = _np.around(ang, decimals)  # Round
        strg = ""
        if label != None:
            strg += label + " "
        strg += str(mag) + " ∠ " + str(ang) + "°"
        if unit != None:
            strg += " " + unit
        # Print values (by default)
        if printval:
            if title != None:
                print(title)
            print(strg)
        # Return values when requested
        if ret:
            return ([mag, ang])
    else:
        raise ValueError("Invalid Input Type")


# Define Phase/Line Converter
def phaseline(VLL=None, VLN=None, Iline=None, Iphase=None, realonly=None,
              **kwargs):
    r"""
    Line-Line to Line-Neutral Converter.

    This function is designed to return the phase- or line-equivalent
    of the voltage/current provided. It is designed to be used when
    converting delta- to wye-connections and vice-versa.
    Given a voltage of one type, this function will return the
    voltage of the opposite type. The same is true for current.

    .. math:: V_{LL} = \sqrt{3}∠30° * V_{LN}
       :label: voltages

    Typical American (United States) standard is to note voltages in
    Line-to-Line values (VLL), and often, the Line-to-Neutral voltage
    is of value, this function uses the voltage :eq:`voltages` relation
    to evaluate either voltage given the other.

    .. math:: I_{Φ} = \frac{I_{line}}{\sqrt{3}∠30°}
       :label: currents

    Often, the phase current in a delta-connected device is of
    particular interest, and the line-current is provided. This
    function uses the current :eq:`currents` formula to evaluate
    phase- and line-current given the opposing term.

    Parameters
    ----------
    VLL:        float, optional
                The Line-to-Line Voltage; default=None
    VLN:        float, optional
                The Line-to-Neutral Voltage; default=None
    Iline:      float, optional
                The Line-Current; default=None
    Iphase:     float, optional
                The Phase-Current; default=None
    realonly:   bool, optional
                Replacement of `complex` argument. Control to return
                value in complex form; default=None
    complex:    bool, optional, DEPRECATED
                Control to return value in complex form, refer to
                `realonly` instead. default=None
    """
    # Monitor for deprecated input
    if 'complex' in kwargs.keys():
        if realonly == None:
            realonly = not kwargs['complex']
        caller = _getframeinfo(_stack()[1][0])
        # Demonstrate Deprecation Warning
        _showwarning('`complex` argument will be deprecated in favor of `realonly`',
                     DeprecationWarning, caller.filename, caller.lineno)
    output = 0
    # Given VLL, convert to VLN
    if VLL is not None:
        VLN = VLL / (VLLcVLN)
        output = VLN
    # Given VLN, convert to VLL
    elif VLN is not None:
        VLL = VLN * VLLcVLN
        output = VLL
    # Given Iphase, convert to Iline
    elif Iphase is not None:
        Iline = Iphase * ILcIP
        output = Iline
    # Given Iline, convert to Iphase
    elif Iline is not None:
        Iphase = Iline / ILcIP
        output = Iphase
    # None given, error encountered
    else:
        print("ERROR: No value given" +
              "or innapropriate value" +
              "given.")
        return (0)
    # Auto-detect Complex Values
    if isinstance(output, complex) and realonly == None:
        realonly = False
    # Return as complex only when requested
    if realonly == True:
        return abs(output)
    return output


# Define Power Set Function
def powerset(P=None, Q=None, S=None, PF=None, find=''):
    """
    Power Triangle Conversion Function.

    This function is designed to calculate all values
    in the set { P, Q, S, PF } when two (2) of the
    values are provided. The equations in this
    function are prepared for AC values, that is:
    real and reactive power, apparent power, and power
    factor.

    Parameters
    ----------
    P:      float, optional
            Real Power, unitless; default=None
    Q:      float, optional
            Reactive Power, unitless; default=None
    S:      float, optional
            Apparent Power, unitless; default=None
    PF:     float, optional
            Power Factor, unitless, provided as a
            decimal value, lagging is positive,
            leading is negative; default=None
    find:   str, optional
            Control argument to specify which value
            should be returned.

    Returns
    -------
    P:      float
            Calculated Real Power Magnitude
    Q:      float
            Calculated Reactive Power Magnitude
    S:      float
            Calculated Apparent Power Magnitude
    PF:     float
            Calculated Power Factor
    """
    # Given P and Q
    if (P != None) and (Q != None):
        S = _np.sqrt(P ** 2 + Q ** 2)
        PF = P / S
        if Q < 0:
            PF = -PF
    # Given S and PF
    elif (S != None) and (PF != None):
        P = abs(S * PF)
        Q = _np.sqrt(S ** 2 - P ** 2)
        if PF < 0:
            Q = -Q
    # Given P and PF
    elif (P != None) and (PF != None):
        S = P / PF
        Q = _np.sqrt(S ** 2 - P ** 2)
        if PF < 0:
            Q = -Q
    # Given P and S
    elif (P != None) and (S != None):
        Q = _np.sqrt(S ** 2 - P ** 2)
        PF = P / S
    # Given Q and S
    elif (Q != None) and (S != None):
        P = _np.sqrt(S ** 2 - Q ** 2)
        PF = P / S
    else:
        raise ValueError("ERROR: Invalid Parameters or too few" +
                         " parameters given to calculate.")
    # Return
    find = find.upper()
    if find == 'P':
        return (P)
    elif find == 'Q':
        return (Q)
    elif find == 'S':
        return (S)
    elif find == 'PF':
        return (PF)
    else:
        return (P, Q, S, PF)

def slew_rate(V=None, freq=None, SR=None, find=''):
    """
    Slew Rate Calculator.

    This function is designed to calculate slew rate
    i.e the change of voltage per unit of time`

    Parameters
    ----------
    V:      float, optional
            Voltage, Volts; default=None
    freq:      float, optional
            Frequency, Hz; default=None
    SR:      float, optional
            Slew Rate, Volts/sec; default=None
    find:   str, optional
            Control argument to specify which value
            should be returned.

    Returns
    -------
    V:      float
            Calculated Volatage
    freq:   float
            Calculated frequency
    SR:     float
            Calculated slew rate
    """
    if V!=None and freq!=None:
        SR = 2*_np.pi*V*freq
    elif freq!=None and SR!=None:
        V = SR/(2*_np.pi*freq)
    elif V!=None and SR!=None:
        freq = SR/(2*_np.pi*V)
    else:
        raise ValueError("ERROR: Invalid Parameters or too few" +
                        " parameters given to calculate.")
    if find == 'V':
        return (V)
    elif find == 'freq':
        return (freq)
    elif find == 'SR':
        return (SR)
    else:
        return (V, freq, SR)

# Define Power Triangle Function
def powertriangle(P=None, Q=None, S=None, PF=None, color="red",
                  text="Power Triangle", printval=False):
    """
    Power Triangle Plotting Function.

    This function is designed to draw a power triangle given
    values for the complex power system.

    .. image:: /static/PowerTriangle.png

    Parameters
    ----------
    P:          float
                Real Power, unitless, default=None
    Q:          float
                Reactive Power, unitless, default=None
    S:          float
                Apparent Power, unitless, default=None
    PF:         float
                Power Factor, unitless, provided as a decimal value, lagging is
                positive, leading is negative; default=None
    color:      string, optional
                The color of the power triangle lines, default="red"
    text:       string, optional
                The title of the power triangle plot, default="Power Triangle"
    printval:   bool, optional
                Control argument to allow the numeric values to be printed on
                the plot, default="False"
    
    Returns
    -------
    matplotlib.pyplot:  Plotting object to be used for additional configuration
                        or plotting.
    """
    # Calculate all values if not all are provided
    if (P == None or Q == None or S == None or PF == None):
        P, Q, S, PF = powerset(P, Q, S, PF)

    # Generate Lines
    Plnx = [0, P]
    Plny = [0, 0]
    Qlnx = [P, P]
    Qlny = [0, Q]
    Slnx = [0, P]
    Slny = [0, Q]

    # Plot Power Triangle
    _plt.figure(1)
    _plt.title(text)
    _plt.plot(Plnx, Plny, color=color)
    _plt.plot(Qlnx, Qlny, color=color)
    _plt.plot(Slnx, Slny, color=color)
    _plt.xlabel("Real Power (W)")
    _plt.ylabel("Reactive Power (VAR)")
    mx = max(abs(P), abs(Q))

    if P > 0:
        _plt.xlim(0, mx * 1.1)
        x = mx
    else:
        _plt.xlim(-mx * 1.1, 0)
        x = -mx
    if Q > 0:
        _plt.ylim(0, mx * 1.1)
        y = mx
    else:
        _plt.ylim(-mx * 1.1, 0)
        y = -mx
    if PF > 0:
        PFtext = " Lagging"
    else:
        PFtext = " Leading"
    text = "P:   " + str(P) + " W\n"
    text = text + "Q:   " + str(Q) + " VAR\n"
    text = text + "S:   " + str(S) + " VA\n"
    text = text + "PF:  " + str(abs(PF)) + PFtext + "\n"
    text = text + "ΘPF: " + str(_np.degrees(_np.arccos(PF))) + "°" + PFtext
    # Print all values if asked to
    if printval:
        _plt.text(x / 20, y * 4 / 5, text, color=color)
    return _plt


# Define Transformer Short-Circuit/Open-Circuit Function
def transformertest(Poc=False, Voc=False, Ioc=False, Psc=False, Vsc=False,
                    Isc=False):
    """
    Electrical Transformer Rated Test Evaluator.

    This function will determine the non-ideal circuit components of
    a transformer (Req and Xeq, or Rc and Xm) given the test-case
    parameters for the open-circuit test and/or the closed-circuit
    test. Requires one or both of two sets: { Poc, Voc, Ioc }, or
    { Psc, Vsc, Isc }.
    All values given must be given as absolute value, not complex.
    All values returned are given with respect to primary.

    Parameters
    ----------
    Poc:    float, optional
            The open-circuit measured power (real power), default=None
    Voc:    float, optional
            The open-circuit measured voltage (measured on X),
            default=None
    Ioc:    float, optional
            The open-circuit measured current (measured on primary),
            default=None
    Psc:    float, optional
            The short-circuit measured power (real power), default=None
    Vsc:    float, optional
            The short-circuit measured voltage (measured on X),
            default=None
    Isc:    float, optional
            The short-circuit measured current (measured on X),
            default=None

    Returns
    -------
    {Req,Xeq,Rc,Xm}:    Given all optional args
    {Rc, Xm}:           Given open-circuit parameters
    {Req, Xeq}:         Given short-circuit parameters
    """
    SC = False
    OC = False
    # Given Open-Circuit Values
    if (Poc != None) and (Voc != None) and (Ioc != None):
        PF = Poc / (Voc * Ioc)
        Y = _c.rect(Ioc / Voc, -_np.arccos(PF))
        Rc = 1 / Y.real
        Xm = -1 / Y.imag
        OC = True
    # Given Short-Circuit Values
    if (Psc != None) and (Vsc != None) and (Isc != None):
        PF = Psc / (Vsc * Isc)
        Zeq = _c.rect(Vsc / Isc, _np.arccos(PF))
        Req = Zeq.real
        Xeq = Zeq.imag
        SC = True
    # Return All if Found
    if OC and SC:
        return (Req, Xeq, Rc, Xm)
    elif OC:
        return (Rc, Xm)
    elif SC:
        return (Req, Xeq)
    else:
        print("An Error Was Encountered.\n" +
              "Not enough arguments were provided.")


# Define Non-Linear Power Factor Calculator
def nlinpf(PFtrue=False, PFdist=False, PFdisp=False):
    """
    Non-Linear Power Factor Evaluator.

    This function is designed to evaluate one of three unknowns
    given the other two. These particular unknowns are the arguments
    and as such, they are described in the representative sections
    below.

    Parameters
    ----------
    PFtrue:     float, exclusive
                The "True" power-factor, default=None
    PFdist:     float, exclusive
                The "Distorted" power-factor, default=None
    PFdisp:     float, exclusive
                The "Displacement" power-factor, default=None

    Returns
    -------
    float:  This function will return the unknown variable from the previously
            described set of variables.
    """
    if (PFtrue != None and PFdist != None and PFdisp != None):
        raise ValueError("ERROR: Too many constraints, no solution.")
    elif (PFdist != None and PFdisp != None):
        return (PFdist * PFdisp)
    elif (PFtrue != None and PFdisp != None):
        return (PFtrue / PFdisp)
    elif (PFtrue != None and PFdist != None):
        return (PFtrue / PFdist)
    else:
        raise ValueError("ERROR: Function requires at least two arguments.")


# Define Short-Circuit RL Current Calculator
def iscrl(V, Z, t=None, f=None, mxcurrent=True, alpha=None):
    """
    Short-Circuit-Current (ISC) Calculator.

    The Isc-RL function (Short Circuit Current for RL Circuit)
    is designed to calculate the short-circuit current for an
    RL circuit.

    Parameters
    ----------
    V:          float
                The absolute magnitude of the voltage.
    Z:          float
                The complex value of the impedance. (R + jX)
    t:          float, optional
                The time at which the value should be calculated,
                should be specified in seconds, default=None
    f:          float, optional
                The system frequency, specified in Hz, default=None
    mxcurrent:  bool, optional
                Control variable to enable calculating the value at
                maximum current, default=True
    alpha:      float, optional
                Angle specification, default=None

    Returns
    -------
    Opt 1 - (Irms, IAC, K):     The RMS current with maximum DC
                                offset, the AC current magnitude,
                                and the asymmetry factor.
    Opt 2 - (i, iAC, iDC, T):   The Instantaneous current with
                                maximum DC offset, the instantaneous
                                AC current, the instantaneous DC
                                current, and the time-constant T.
    Opt 3 - (Iac):              The RMS current without DC offset.
    """
    # Calculate omega, theta, R, and X
    if (f != None):
        omega = 2 * _np.pi * f
    else:
        omega = None
    R = abs(Z.real)
    X = abs(Z.imag)
    theta = _np.arctan(X / R)

    # If Maximum Current is Desired and No alpha provided
    if (mxcurrent and alpha == None):
        alpha = theta - _np.pi / 2
    elif (mxcurrent and alpha != None):
        raise ValueError("ERROR: Inappropriate Arguments Provided.\n" +
                         "Not both mxcurrent and alpha can be provided.")

    # Calculate Asymmetrical (total) Current if t != None
    if (t != None and f != None):
        # Calculate RMS if none of the angular values are provided
        if (alpha == None and omega == None):
            # Calculate tau
            tau = t / (1 / 60)
            K = _np.sqrt(1 + 2 * _np.exp(-4 * _np.pi * tau / (X / R)))
            IAC = abs(V / Z)
            Irms = K * IAC
            # Return Values
            return (Irms, IAC, K)
        elif (alpha == None or omega == None):
            raise ValueError("ERROR: Inappropriate Arguments Provided.")
        # Calculate Instantaneous if all angular values provided
        else:
            # Convert Degrees to Radians
            omega = _np.radians(omega)
            alpha = _np.radians(alpha)
            theta = _np.radians(theta)
            # Calculate T
            T = X / (2 * _np.pi * f * R)  # seconds
            # Calculate iAC and iDC
            iAC = _np.sqrt(2) * V / Z * _np.sin(omega * t + alpha - theta)
            iDC = -_np.sqrt(2) * V / Z * _np.sin(alpha - theta) * _np.exp(-t / T)
            i = iAC + iDC
            # Return Values
            return (i, iAC, iDC, T)
    elif ((t != None and f == None) or (t == None and f != None)):
        raise ValueError("ERROR: Inappropriate Arguments Provided.\n" +
                         "Must provide both t and f or neither.")
    else:
        Iac = abs(V / Z)
        return (Iac)


# Define Voltage Divider Calculator
def voltdiv(Vin, R1, R2, Rload=None):
    r"""
    Electrical Voltage Divider Function.

    This function is designed to calculate the output
    voltage of a voltage divider given the input voltage,
    the resistances (or impedances) and the load resistance
    (or impedance) if present.

    .. math:: V_{out} = V_{in} * \frac{R_2}{R_1+R_2}

    .. math:: V_{out}=V_{in}*\frac{R_2||R_{load}}{R_1+(R_2||R_{load})}

    Parameters
    ----------
    Vin:    float
            The Input Voltage, may be real or complex
    R1:     float
            The top resistor of the divider (real or complex)
    R2:     float
            The bottom resistor of the divider, the one which
            the output voltage is measured across, may be
            either real or complex
    Rload:  float, optional
            The Load Resistor (or impedance), default=None

    Returns
    -------
    Vout:   float
            The Output voltage as measured across R2 and/or Rload
    """
    # Determine whether Rload is given
    if (Rload == None):  # No Load Given
        Vout = Vin * R2 / (R1 + R2)
    else:  # Load was given
        Rp = R2*Rload/(R2 + Rload)
        Vout = Vin * Rp / (R1 + Rp)
    return (Vout)


# Define Current Divider Calculator
def curdiv(Ri, Rset, Vin=None, Iin=None, Vout=False, combine=True):
    r"""
    Electrical Current Divider Function.

    This function is disigned to accept the input current, or input
    voltage to a resistor (or impedance) network of parallel resistors
    (impedances) and calculate the current through a particular element.

    Parameters
    ----------
    Ri:         float
                The Particular Resistor of Interest, should not be included in
                the tuple passed to Rset.
    Rset:       float
                Tuple of remaining resistances (impedances) in network.
    Vin:        float, optional
                The input voltage for the system, default=None
    Iin:        float, optional
                The input current for the system, default=None
    Vout:       bool, optional
                Control argument to enable return of the voltage across the
                resistor (impedance) of interest (Ri)
    combine:    bool, optional
                Control argument to force resistance combination. default=True

    Returns
    -------
    Opt1 - Ii:          The Current through the resistor (impedance) of interest
    Opt2 - (Ii,Vi):     The afore mentioned current, and voltage across the
                        resistor (impedance) of interest
    """
    # Validate Tuple
    if not isinstance(Rset, tuple):
        Rset = (Rset,)  # Set as Tuple
    # Calculate The total impedance
    if combine:
        # Combine tuples, then calculate total resistance
        Rtot = parallelz(Rset + (Ri,))
    else:
        Rtot = parallelz(Rset)
    # Determine Whether Input was given as Voltage or Current
    if (Vin != None and Iin == None):  # Vin Provided
        Iin = Vin / Rtot  # Calculate total current
        Ii = Iin * Rtot / Ri  # Calculate the current of interest
    elif (Vin == None and Iin != None):  # Iin provided
        Ii = Iin * Rtot / Ri  # Calculate the current of interest
    else:
        raise ValueError("ERROR: Too many or too few constraints provided.")
    if (Vout):  # Asked for voltage across resistor of interest
        Vi = Ii * Ri
        return (Ii, Vi)
    else:
        return (Ii)

#Induction Machine Slip
def induction_machine_slip(Nr, freq=60, poles=4):
    r"""
    Induction Machine slip calculator.

    This function is used to calculate the slip of an induction machine.

    .. math:: slip = 1 - \frac{Nr}{120*frac{freq}{poles}}

    Parameters
    ----------
    Nr: float, Induction Machine Speed (in rpm)
    freq: int, Supply AC frequency, default=60
    poles: Number of poles inside Induction Machine, default=4

    Returns
    -------
    slip: float, Induction Machine forward Slip
    """
    Ns = (120*freq)/poles
    return (Ns - Nr)/(Ns)

# Define Function to Evaluate Resistance Needed for LED
def led_resistor(Vsrc, Vfwd = 2, Ifwd = 20):
    r"""
    LED Resistor Calculator.

    This function will evaluate the necessary resistance value for a simple LED
    circuit with a voltage source, resistor, and LED.

    .. math:: R_\text{LED} = \frac{V_\text{SRC} - V_\text{FWD}}{I_\text{FWD}}

    Parameters
    ----------
    Vsrc:   float
            Source voltage, as measured across both LED and resistor in circuit.
    Vfwd:   float, optional
            Forward voltage of LED (or series LEDs if available), default=2
    Ifwd:   float, optional
            Forward current of LEDs in milliamps, default=20 (milliamps)
    
    Returns
    -------
    R:      float
            The resistance value most appropriate for the LED circuit.
    """
    # Calculate and Return!
    R = (Vsrc - Vfwd) / (Ifwd * 1000)
    return R


# Define Instantaneous Power Calculator
def instpower(P, Q, t, freq=60):
    r"""
    Instantaneous Power Function.

    This function is designed to calculate the instantaneous power at a
    specified time t given the magnitudes of P and Q.

    .. math:: P_{inst} = P+P*cos(2*\omega*t)-Q*sin(2*\omega*t)

    Parameters
    ----------
    P:      float
            Magnitude of Real Power
    Q:      float
            Magnitude of Reactive Power
    t:      float
            Time at which to evaluate
    freq:   float, optional
            System frequency (in Hz), default=60

    Returns
    -------
    Pinst:  float
            Instantaneous Power at time t
    """
    # Evaluate omega
    w = 2 * _np.pi * freq
    # Calculate
    Pinst = P + P * _np.cos(2 * w * t) - Q * _np.sin(2 * w * t)
    return (Pinst)


# Define Delta-Wye Impedance Network Calculator
def dynetz(delta=None, wye=None, round=None):
    r"""
    Delta-Wye Impedance Converter.

    This function is designed to act as the conversion utility
    to transform delta-connected impedance values to wye-
    connected and vice-versa.

    .. math::
       Z_{sum} = Z_{1/2} + Z_{2/3} + Z_{3/1}//
       Z_1 = \frac{Z_{1/2}*Z_{3/1}}{Z_{sum}}//
       Z_2 = \frac{Z_{1/2}*Z_{2/3}}{Z_{sum}}//
       Z_3 = \frac{Z_{2/3}*Z_{3/1}}{Z_{sum}}

    .. math::
       Z_{ms} = Z_1*Z_2 + Z_2*Z_3 + Z_3*Z_1//
       Z_{2/3} = \frac{Z_{ms}}{Z_1}//
       Z_{3/1} = \frac{Z_{ms}}{Z_2}//
       Z_{1/2} = \frac{Z_{ms}}{Z_3}

    Parameters
    ----------
    delta:  tuple of float, exclusive
            Tuple of the delta-connected impedance values as:
            { Z12, Z23, Z31 }, default=None
    wye:    tuple of float, exclusive
            Tuple of the wye-connected impedance valuse as:
            { Z1, Z2, Z3 }, default=None

    Returns
    -------
    delta-set:  tuple of float
                Delta-Connected impedance values { Z12, Z23, Z31 }
    wye-set:    tuple of float
                Wye-Connected impedance values { Z1, Z2, Z3 }
    """
    # Determine which set of impedances was provided
    if (delta != None and wye == None):
        Z12, Z23, Z31 = delta  # Gather particular impedances
        Zsum = Z12 + Z23 + Z31  # Find Sum
        # Calculate Wye Impedances
        Z1 = Z12 * Z31 / Zsum
        Z2 = Z12 * Z23 / Zsum
        Z3 = Z23 * Z31 / Zsum
        Zset = (Z1, Z2, Z3)
        if round != None: Zset = _np.around(Zset, round)
        return (Zset)  # Return Wye Impedances
    elif (delta == None and wye != None):
        Z1, Z2, Z3 = wye  # Gather particular impedances
        Zmultsum = Z1 * Z2 + Z2 * Z3 + Z3 * Z1
        Z23 = Zmultsum / Z1
        Z31 = Zmultsum / Z2
        Z12 = Zmultsum / Z3
        Zset = (Z12, Z23, Z31)
        if round != None: Zset = _np.around(Zset, round)
        return (Zset)  # Return Delta Impedances
    else:
        raise ValueError(
            "ERROR: Either delta or wye impedances must be specified."
        )

#calculating impedance of bridge network
def bridge_impedance(z1, z2, z3, z4, z5):
    r"""
    Bridge Impedance Calculator.
    
    The following condition describing the Wheatstone Bridge is utilized to
    ensure that current through `z5` will be zero.

    .. math:: z1 \cdot z3 = z2 \cdot z4
    
    .. image:: /static/WheatstoneBridgeCircuit.png
    
    Parameters
    ----------
    z1:     [float, complex]
            Bridge impedance 1
    z2:     [float, complex]
            Bridge impedance 2
    z3:     [float, complex]
            Bridge impedance 3
    z4:     [float, complex]
            Bridge impedance 4
    z5:     [float, complex]
            Detector impedance or impedance between two bridge branches

    Returns
    -------
    effective bridge impedance

    """
    if z1 * z3 == z2 * z4:
        return (z1 + z2) * (z3 + z4) / (z1 + z2 + z3 + z4)
    else:
        za, zb, zc = dynetz(delta = (z1, z5, z4))
        ze1 = zb + z2
        ze2 = zc + z3
        return za + (ze1*ze2)/(ze1+ze2)


# Define Single Line Power Flow Calculator
def powerflow(Vsend, Vrec, Xline):
    r"""
    Approximated Power-Flow Calculator.

    This function is designed to calculate the ammount of real
    power transferred from the sending end to the recieving end
    of an electrical line given the sending voltage (complex),
    the receiving voltage (complex) and the line impedance.

    .. math::
       P_{flow}=\frac{|V_{send}|*|V_{rec}|}{X_{line}}*sin(\theta_{send}
       -\theta_{rec})

    Parameters
    ----------
    Vsend:      complex
                The sending-end voltage, should be complex
    Vrec:       complex
                The receiving-end voltage, should be complex
    Xline:      float
                The line admitance, should be float

    Returns
    -------
    pflow:      float
                The Real power transferred from sending-end to
                receiving-end, positive values denote power
                flow from send to receive, negative values
                denote vice-versa.
    """
    # Evaluate the Input Terms
    Vs = abs(Vsend)
    ds = _c.phase(Vsend)
    Vr = abs(Vrec)
    dr = _c.phase(Vrec)
    # Calculate Power Flow
    pflow = (Vs * Vr) / (Xline) * _np.sin(ds - dr)
    return (pflow)


# Define Impedance From Power and X/R
def zsource(S, V, XoR, Sbase=None, Vbase=None, perunit=True):
    """
    Source Impedance Calculator.

    Used to calculate the source impedance given the apparent power
    magnitude and the X/R ratio.

    Parameters
    ----------
    S:          float
                The (rated) apparent power magnitude of the source.
                This may also be refferred to as the "Short-Circuit MVA"
    V:          float
                The (rated) voltage of the source terminals, not
                specifically identified as either Line-to-Line or Line-to-
                Neutral.
    XoR:        float
                The X/R ratio rated for the source, may optionally be a list
                of floats to accomidate sequence impedances or otherwise.
    Sbase:      float, optional
                The per-unit base for the apparent power. If set to
                None, will automatically force Sbase to equal S.
                If set to True will treat S as the per-unit value.
    Vbase:      float, optional
                The per-unit base for the terminal voltage. If set to
                None, will automaticlaly force Vbase to equal V. If
                set to True, will treat V as the per-unit value.
    perunit:    boolean, optional
                Control value to enable the return of output in per-
                unit base. default=True

    Returns
    -------
    Zsource_pu: complex
                The per-unit evaluation of the source impedance.
                Will be returned in ohmic (not per-unit) value if
                *perunit* argument is specified as False.
    """
    # Force Sbase and Vbase if needed
    if Vbase == None:
        Vbase = V
    if Sbase == None:
        Sbase = S
    # Prevent scaling if per-unit already applied
    if Vbase == True:
        Vbase = 1
    if Sbase == True:
        Sbase = 1
    # Set to per-unit
    Spu = S / Sbase
    Vpu = V / Vbase
    # Evaluate Zsource Magnitude
    Zsource_pu = Vpu ** 2 / Spu
    # Evaluate the angle
    nu = _np.degrees(_np.arctan(XoR))
    # Conditionally Evaluate Phasor Impedance
    if isinstance(nu, (list, _np.ndarray)):
        Zsource_pu = []
        for angle in nu:
            Zsource_pu.append(phasor(Zsource_pu, angle))
    else:
        Zsource_pu = phasor(Zsource_pu, nu)
    if not perunit:
        Zsource = Zsource_pu * Vbase ** 2 / Sbase
        return (Zsource)
    return (Zsource_pu)


# Define Impedance Decomposer
def zdecompose(Zmag, XoR):
    """
    Impedance Decomposition Function.

    A function to decompose the impedance magnitude into its
    corresponding resistance and reactance using the X/R ratio.

    It is possible to "neglect" R, or make it a very small number;
    this is done by setting the X/R ratio to a very large number
    (X being much larger than R).

    Parameters
    ----------
    Zmag:       float
                The magnitude of the impedance.
    XoR:        float
                The X/R ratio (reactance over impedance).

    Returns
    -------
    R:          float
                The resistance (in ohms)
    X:          float
                The reactance (in ohms)
    """
    # Evaluate Resistance
    R = Zmag / _np.sqrt(XoR ** 2 + 1)
    # Evaluate Reactance
    X = R * XoR
    # Return
    return (R, X)


# Define Power Reactance Calculator
def powerimpedance(S, V, PF=None, parallel=False, terms=False):
    r"""
    Impedance from Apparent Power Formula.

    Function to determine the ohmic resistance/reactance
    (impedance) represented by the apparent power (S).

    .. math:: R = \frac{V^2}{P} \hspace{2cm} X = \frac{V^2}{Q}
       :label: series-resistance

    .. math:: Z = \left(\frac{V^2}{S}\right)^*
       :label: series-impedance

    .. math:: Z = \left(\frac{V^2}{(3*S)}\right)^*
       :label: parallel

    This function can evaluate the component values for
    both series :eq:`series-resistance`/:eq:`series-impedance`
    and parallel :eq:`parallel` connected circuits.

    Parameters
    ----------
    S:          complex, float
                The apparent power of the passive element,
                may be purely resistive or purely reactive.
    V:          float
                The operating voltage of the passive element.
                Note that this is specifically not Line-Line or
                Line-Neutral voltage, rather the voltage of the
                element.
    PF:         float, optional
                The operating Power-Factor, should be specified
                if S is given as a float (not complex). Positive
                PF correlates to lagging, negative to leading.
                default=None
    parallel:   bool, optional
                Control point to specify whether the ohmic
                impedance should be returned as series components
                (False opt.) or parallel components (True opt.).
    terms:      bool, optional
                Control point to specify whether return should
                be made as resistance and reactance, or simply
                the complex impedance. Setting of False will
                return complex impedance, setting of True will
                return individual components (R, X).

    Returns
    -------
    R:          float
                The ohmic resistance required to consume the
                specified apparent power (S) at the rated
                voltage (V).
    X:          float
                The ohmic reactance required to consume the
                specified apparent power (S) at the rated
                voltage (V).
    """
    # Condition Inputs
    V = abs(V)
    # Test for Parallel Component Option and Evaluate
    if isinstance(S, complex) or PF != None:
        if PF != None:
            # Evaluate Elements
            P, Q, S, PF = powerset(S=S, PF=PF)
        else:
            P = S.real
            Q = S.imag
        # Compute Elements
        if parallel:
            Zp = V ** 2 / (3 * (P + 1j*Q))
        else:
            Zp = V ** 2 / (P + 1j*Q)
        Z = _np.conjugate(Zp)
        R = Z.real
        X = Z.imag
        # Conditionally Return as Impedance
        if terms:
            return (R, X)
        return Z
    # Not Complex (just R)
    R = V ** 2 / S
    return R


# Define Cold-Junction-Voltage Calculator
def coldjunction(Tcj, coupletype="K", To=None, Vo=None, P1=None, P2=None,
                 P3=None, P4=None, Q1=None, Q2=None, round=None):
    """
    Thermocouple Cold-Junction Formula.

    Function to calculate the expected cold-junction-voltage given
    the temperature at the cold-junction.

    Parameters
    ----------
    Tcj:        float
                The temperature (in degrees C) that the junction is
                currently subjected to.
    coupletype: string, optional
                Thermocouple Type, may be one of (B,E,J,K,N,R,S,T), default="K"
    To:         float, optional
                Temperature Constant used in Polynomial.
    Vo:         float, optional
                Voltage Constant used in Polynomial.
    P1:         float, optional
                Polynomial constant.
    P2:         float, optional
                Polynomial constant.
    P3:         float, optional
                Polynomial constant.
    P4:         float, optional
                Polynomial constant.
    Q1:         float, optional
                Polynomial constant.
    Q2:         float, optional
                Polynomial constant.
    round:      int, optional
                Control input to specify how many decimal places the result
                should be rounded to, default=1.

    Returns
    -------
    Vcj:        float
                The calculated cold-junction-voltage in volts.
    """
    # Condition Inputs
    coupletype = coupletype.upper()
    # Validate Temperature Range
    if coupletype == "B":
        if not (0 < Tcj and Tcj < 70):
            raise ValueError("Temperature out of range.")
    else:
        if not (-20 < Tcj and Tcj < 70):
            raise ValueError("Temperature out of range.")
    # Define Constant Lookup System
    lookup = ["B", "E", "J", "K", "N", "R", "S", "T"]
    if not (coupletype in lookup):
        raise ValueError("Invalid Thermocouple Type")
    index = lookup.index(coupletype)
    # Define Constant Dictionary
    # Load Data Into Terms
    parameters = {}
    for var in COLD_JUNCTION_DATA.keys():
        parameters[var] = parameters.get(var, None) or COLD_JUNCTION_DATA[var][index]
    To, Vo, P1, P2, P3, P4, Q1, Q2 = [parameters[key] for key in COLD_JUNCTION_KEYS]
    # Define Formula Terms
    tx = (Tcj - To)
    num = tx * (P1 + tx * (P2 + tx * (P3 + P4 * tx)))
    den = 1 + tx * (Q1 + Q2 * tx)
    Vcj = Vo + num / den
    # Round Value if Allowed
    if round != None:
        Vcj = _np.around(Vcj, round)
    # Return in milivolts
    return (Vcj * m)


# Define Thermocouple Temperature Calculation
def thermocouple(V, coupletype="K", fahrenheit=False, cjt=None, To=None,
                 Vo=None, P1=None, P2=None, P3=None, P4=None, Q1=None, Q2=None,
                 Q3=None, round=1):
    """
    Thermocouple Temperature Calculator.

    Utilizes polynomial formula to calculate the temperature being monitored
    by a thermocouple. Allows for various thermocouple types (B,E,J,K,N,R,S,T)
    and various cold-junction-temperatures.

    Parameters
    ----------
    V:          float
                Measured voltage (in Volts)
    coupletype: string, optional
                Thermocouple Type, may be one of (B,E,J,K,N,R,S,T), default="K"
    fahrenheit: bool, optional
                Control to enable return value as Fahrenheit instead of Celsius,
                default=False
    cjt:        float, optional
                Cold-Junction-Temperature
    To:         float, optional
                Temperature Constant used in Polynomial.
    Vo:         float, optional
                Voltage Constant used in Polynomial.
    P1:         float, optional
                Polynomial constant.
    P2:         float, optional
                Polynomial constant.
    P3:         float, optional
                Polynomial constant.
    P4:         float, optional
                Polynomial constant.
    Q1:         float, optional
                Polynomial constant.
    Q2:         float, optional
                Polynomial constant.
    Q3:         float, optional
                Polynomial constant.
    round:      int, optional
                Control input to specify how many decimal places the result
                should be rounded to, default=1.

    Returns
    -------
    T:          float
                The temperature (by default in degrees C, but optionally in
                degrees F) as computed by the function.
    """
    # Condition Inputs
    coupletype = coupletype.upper()
    V = V / m  # Scale volts to milivolts
    # Determine Cold-Junction-Voltage
    if cjt != None:
        Vcj = coldjunction(cjt, coupletype, To, Vo, P1, P2, P3, P4, Q1, Q2, round)
        V += Vcj / m
    # Define Constant Lookup System
    lookup = ["B", "E", "J", "K", "N", "R", "S", "T"]
    if not (coupletype in lookup):
        raise ValueError("Invalid Thermocouple Type")
    # Determine Array Selection
    vset = THERMO_COUPLE_VOLTAGES[coupletype]
    if V < vset[0] * m:
        raise ValueError("Voltage Below Lower Bound")
    elif vset[0] <= V < vset[1]:
        select = 0
    elif vset[1] <= V < vset[2]:
        select = 1
    elif vset[2] <= V < vset[3]:
        select = 2
    elif vset[3] <= V < vset[4]:
        select = 3
    elif vset[4] <= V <= vset[5]:
        select = 4
    elif vset[5] < V:
        raise ValueError("Voltage Above Upper Bound")
    else:
        raise ValueError("Internal Error!")
    # Load Data Into Terms
    parameters = {}
    for i, key in enumerate(THERMO_COUPLE_KEYS):
        parameters[key] = parameters.get(key, None) or THERMO_COUPLE_DATA[coupletype][i][select]
    Vo, To, P1, P2, P3, P4, Q1, Q2, Q3 = [parameters[key] for key in THERMO_COUPLE_KEYS]
    # Calculate Temperature in Degrees C
    num = (V - Vo) * (P1 + (V - Vo) * (P2 + (V - Vo) * (P3 + P4 * (V - Vo))))
    den = 1 + (V - Vo) * (Q1 + (V - Vo) * (Q2 + Q3 * (V - Vo)))
    temp = To + num / den
    # Return Temperature
    if fahrenheit:
        temp = (temp * 9 / 5) + 32
    temp = _np.around(temp, round)
    return (temp)


# Define RTD Calculator
def rtdtemp(RT, rtdtype="PT100", fahrenheit=False, Rref=None, Tref=None,
            a=None, round=1):
    """
    RTD Temperature Calculator.

    Evaluates the measured temperature based on the measured resistance
    and the RTD type.

    Parameters
    ----------
    RT:         float
                The measured resistance (in ohms).
    rtdtype:    string
                RTD Type string, may be one of: (PT100, PT1000,
                CU100, NI100, NI120, NIFE), default=PT100
    fahrenheit: bool, optional
                Control parameter to force return into degrees
                fahrenheit, default=False
    Rref:       float, optional
                Resistance reference, commonly used if non-standard
                RTD type being used. Specified in ohms.
    Tref:       float, optional
                Temperature reference, commonly used if non-standard
                RTD type being used. Specified in degrees Celsius.
    a:          float, optional
                Scaling value, commonly used if non-standard
                RTD type being used.
    round:      int, optional
                Control argument to specify number of decimal points
                in returned value.

    Returns
    -------
    temp:       float
                Calculated temperature, defaults to degrees Celsius.
    """
    # Load Variables
    if Rref == None:
        Rref = RTD_TYPES[rtdtype][0]
    if Tref == None:
        Tref = 0
    if a == None:
        a = RTD_TYPES[rtdtype][1]
    # Define Terms
    num = RT - Rref + Rref * a * Tref
    den = Rref * a
    temp = num / den
    # Return Temperature
    if fahrenheit:
        temp = (temp * 9 / 5) + 32
    temp = _np.around(temp, round)
    return (temp)


# Define function to find VDC setpoint
def vscdcbus(VLL, Zs, P, Q=0, mmax=0.8, debug=False):
    """
    Voltage Sourced Converter DC Bus Voltage Function.

    The purpose of this function is to calculate the
    required DC-bus voltage for a Voltage-Sourced-
    Converter (VSC) given the desired P/Q parameters
    and the known source impedance (Vs) of the VSC.

    Parameters
    ----------
    VLL:    complex
            Line-to-Line voltage on the line-side of
            the source impedance.
    Zs:     complex
            The source impedance of the VSC
    P:      float
            The desired real-power output
    Q:      float, optional
            The desired reactive-power output, default=0
    mmax:   float, optional
            The maximum of the m value for the converter
            default=0.8
    debug:  bool, optional
            Control value to enable printing stages of
            the calculation, default=False

    Returns
    -------
    VDC:    float
            The DC bus voltage.
    """
    # Determine the Load Current
    Iload = _np.conj((P + 1j * Q) / (VLL * _np.sqrt(3)))
    # Evaluate the Terminal Voltage
    Vtln = abs(VLL / _np.sqrt(3) + Iload * Zs)
    # Find the Peak Terminal Voltage
    Vtpk = _np.sqrt(2) * Vtln
    # Calculate the VDC value
    VDC = 2 * Vtpk / mmax
    if debug:
        print("Iload", Iload)
        print("Vtln", Vtln)
        print("Vtpk", Vtpk)
        print("VDC", VDC)
    return (VDC)


# Define kp/ki/w0L calculating function
def vscgains(Rs, Ls, tau=0.005, freq=60):
    """
    Voltage Sourced Converter Gains Calculator.

    This function is designed to calculate the kp, ki,
    and omega-not-L values for a Phase-Lock-Loop based VSC.

    Parameters
    ----------
    Rs:     float
            The equiv-resistance (in ohms) of the VSC
    Ls:     float
            The equiv-inductance (in Henrys) of the VSC
    tau:    float, optional
            The desired time-constant, default=0.005
    freq:   float, optional
            The system frequency (in Hz), default=60

    Returns
    -------
    kp:     float
            The Kp-Gain Value
    ki:     float
            The Ki-Gain Value
    w0L:    float
            The omega-not-L gain value
    """
    # Calculate kp
    kp = Ls / tau
    # Calculate ki
    ki = kp * Rs / Ls
    # Calculate w0L
    w0L = 2 * _np.pi * freq * Ls
    return (kp, ki, w0L)


# Define Convolution Bar-Graph Function:
def convbar(h, x, outline=True):
    """
    Convolution Bar-Graph Plotter Function.

    Generates plots of each of two input arrays as bar-graphs, then
    generates a convolved bar-graph of the two inputs to demonstrate
    and illustrate convolution, typically for an educational purpose.

    Examples
    --------
    >>> import numpy as np
    >>> import electricpy as ep
    >>> h = np.array([0, 1, 1, 1, 0])
    >>> x = np.array([0, 1, 1, 1, 0])
    >>> ep.convbar(h, x)
    
    .. image:: /static/convbar-example.png

    Parameters
    ----------
    h:      numpy.ndarray
            Impulse Response - Given as Array (Prefferably Numpy Array)
    x:      numpy.ndarray
            Input Function - Given as Array (Prefferably Numpy Array)
    """
    # The impulse response
    M = len(h)
    t = _np.arange(M)
    # Plot
    _plt.subplot(121)
    if (outline): _plt.plot(t, h, color='red')
    _plt.bar(t, h, color='black')
    _plt.xticks([0, 5, 9])
    _plt.ylabel('h')
    _plt.title('Impulse Response')
    _plt.grid()

    # The input function
    N = len(x)
    s = _np.arange(N)
    # Plot
    _plt.subplot(122)
    if (outline): _plt.plot(s, x, color='red')
    _plt.bar(s, x, color='black')
    _plt.xticks([0, 10, 19])
    _plt.title('Input Function')
    _plt.grid()
    _plt.ylabel('x')

    # The output
    L = M + N - 1
    w = _np.arange(L)
    _plt.figure(3)
    y = _np.convolve(h, x)
    if (outline): _plt.plot(w, y, color='red')
    _plt.bar(w, y, color='black')
    _plt.ylabel('y')
    _plt.grid()
    _plt.title('Convolved Output')
    return _plt


# Define Peak Calculator
def peak(val):
    r"""
    Sinusoid RMS to Peak Converter.

    Provides a readable format to convert an RMS (Root-Mean-Square) value to its
    peak representation. Performs a simple multiplication with the square-root
    of two.

    .. math:: V_{\text{peak}} = \sqrt{2} \cdot V_{\text{RMS}}

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.peak(120)
    169.7056274847714
    """
    return (_np.sqrt(2) * val)


# Define RMS Calculator
def rms(val):
    r"""
    Sinusoid Peak to RMS Converter.

    Provides a readable format to convert a peak value to its RMS
    (Root-Mean-Square) representation. Performs a simple division by the
    square-root of two.

    .. math:: V_{\text{RMS}} = \frac{V_{\text{peak}}}{\sqrt{2}}

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.rms(169.7)
    119.99602076735711
    """
    return (val * _np.sqrt(0.5))


# Define Normalized Power Spectrum Function
def wrms(func, dw=0.1, NN=100, quad=False, plot=True,
         title="Power Density Spectrum", round=3):
    """
    WRMS Function.

    This function is designed to calculate the RMS bandwidth (Wrms) using a
    numerical process.

    Parameters
    ----------
    func:       function
                The callable function to use for evaluation
    dw:         float, optional
                The delta-omega to be used, default=0.1
    NN:         int, optional
                The total number of points, default=100
    quad:       bool, optional
                Control value to enable use of integrals
                default=False
    plot:       bool, optional
                Control to enable plotting, default=True
    title:      string, optional
                Title displayed with plot,
                default="Power Density Spectrum"
    round:      int, optional
                Control to round the Wrms on plot,
                default=3

    Returns
    -------
    W:          float
                Calculated RMS Bandwidth (rad/sec)
    """
    # Define omega
    omega = _np.linspace(0, (NN - 1) * dw, NN)
    # Initialize Fraction Terms
    Stot = Sw2 = 0
    # Power Density Spectrum
    Sxx = _np.array([])
    for n in range(NN):
        # Calculate Power Density Spectrum
        Sxx = _np.append(Sxx, func(omega[n]))
        Stot = Stot + Sxx[n]
        Sw2 = Sw2 + (omega[n] ** 2) * Sxx[n]
    if (quad):
        def intf(w):
            return (w ** 2 * func(w))

        num = integrate(intf, 0, _np.inf)[0]
        den = integrate(func, 0, _np.inf)[0]
        # Calculate W
        W = _np.sqrt(num / den)
    else:
        # Calculate W
        W = _np.sqrt(Sw2 / Stot)
    Wr = _np.around(W, round)
    # Plot Upon Request
    if (plot):
        _plt.plot(omega, Sxx)
        _plt.title(title)
        # Evaluate Text Location
        x = 0.65 * max(omega)
        y = 0.80 * max(Sxx)
        _plt.text(x, y, "Wrms: " + str(Wr))
        _plt.show()
    # Return Calculated RMS Bandwidth
    return (W)


# Define Hartley's Equation for Data Capacity
def hartleydata(BW, M):
    """
    Hartley Data Function.

    Function to calculate Hartley's Law, the maximum data rate achievable for
    a given noiseless channel.

    Parameters
    ----------
    BW:         float
                Bandwidth of the data channel.
    M:          float
                Number of signal levels.

    Returns
    -------
    C:          float
                Capacity of channel (in bits per second)
    """
    C = 2 * BW * _np.log2(M)
    return (C)


# Define Shannon's Equation For Data Capacity
def shannondata(BW, S, N):
    """
    Shannon Data Function.

    Function to calculate the maximum data rate that may be achieved given a
    data channel and signal/noise characteristics using Shannon's equation.

    Parameters
    ----------
    BW:         float
                Bandwidth of the data channel.
    S:          float
                Signal strength (in Watts).
    N:          float
                Noise strength (in Watts).

    Returns
    -------
    C:          float
                Capacity of channel (in bits per second)
    """
    C = BW * _np.log2(1 + S / N)
    return (C)


# Define CRC Generator (Sender Side)
def crcsender(data, key):
    """
    CRC Sender Function.

    Function to generate a CRC-embedded message ready for transmission.

    Contributing Author Credit:
    Shaurya Uppal
    Available from: geeksforgeeks.org

    Parameters
    ----------
    data:       string of bits
                The bit-string to be encoded.
    key:        string of bits
                Bit-string representing key.

    Returns
    -------
    codeword:   string of bits
                Bit-string representation of
                encoded message.
    """
    # Define Sub-Functions
    def xor(a, b):
        # initialize result
        result = []

        # Traverse all bits, if bits are
        # same, then XOR is 0, else 1
        for i in range(1, len(b)):
            if a[i] == b[i]:
                result.append('0')
            else:
                result.append('1')

        return (''.join(result))

    # Performs Modulo-2 division
    def mod2div(divident, divisor):
        # Number of bits to be XORed at a time.
        pick = len(divisor)

        # Slicing the divident to appropriate
        # length for particular step
        tmp = divident[0: pick]

        while pick < len(divident):

            if tmp[0] == '1':

                # replace the divident by the result
                # of XOR and pull 1 bit down
                tmp = xor(divisor, tmp) + divident[pick]

            else:  # If leftmost bit is '0'

                # If the leftmost bit of the dividend (or the
                # part used in each step) is 0, the step cannot
                # use the regular divisor; we need to use an
                # all-0s divisor.
                tmp = xor('0' * pick, tmp) + divident[pick]

                # increment pick to move further
            pick += 1

        # For the last n bits, we have to carry it out
        # normally as increased value of pick will cause
        # Index Out of Bounds.
        if tmp[0] == '1':
            tmp = xor(divisor, tmp)
        else:
            tmp = xor('0' * pick, tmp)

        checkword = tmp
        return (checkword)

    # Condition data
    data = str(data)
    # Condition Key
    key = str(key)
    l_key = len(key)

    # Appends n-1 zeroes at end of data
    appended_data = data + '0' * (l_key - 1)
    remainder = mod2div(appended_data, key)

    # Append remainder in the original data
    codeword = data + remainder
    return (codeword)


# Define CRC Generator (Sender Side)
def crcremainder(data, key):
    """
    CRC Remainder Function.

    Function to calculate the CRC remainder of a CRC message.

    Contributing Author Credit:
    Shaurya Uppal
    Available from: geeksforgeeks.org

    Parameters
    ----------
    data:       string of bits
                The bit-string to be decoded.
    key:        string of bits
                Bit-string representing key.

    Returns
    -------
    remainder: string of bits
                Bit-string representation of
                encoded message.
    """
    # Define Sub-Functions
    def xor(a, b):
        # initialize result
        result = []

        # Traverse all bits, if bits are
        # same, then XOR is 0, else 1
        for i in range(1, len(b)):
            if a[i] == b[i]:
                result.append('0')
            else:
                result.append('1')

        return (''.join(result))

    # Performs Modulo-2 division
    def mod2div(divident, divisor):
        # Number of bits to be XORed at a time.
        pick = len(divisor)

        # Slicing the divident to appropriate
        # length for particular step
        tmp = divident[0: pick]

        while pick < len(divident):

            if tmp[0] == '1':

                # replace the divident by the result
                # of XOR and pull 1 bit down
                tmp = xor(divisor, tmp) + divident[pick]

            else:  # If leftmost bit is '0'

                # If the leftmost bit of the dividend (or the
                # part used in each step) is 0, the step cannot
                # use the regular divisor; we need to use an
                # all-0s divisor.
                tmp = xor('0' * pick, tmp) + divident[pick]

                # increment pick to move further
            pick += 1

        # For the last n bits, we have to carry it out
        # normally as increased value of pick will cause
        # Index Out of Bounds.
        if tmp[0] == '1':
            tmp = xor(divisor, tmp)
        else:
            tmp = xor('0' * pick, tmp)

        checkword = tmp
        return (checkword)

    # Condition data
    data = str(data)
    # Condition Key
    key = str(key)
    l_key = len(key)

    # Appends n-1 zeroes at end of data
    appended_data = data + '0' * (l_key - 1)
    remainder = mod2div(appended_data, key)

    return (remainder)


# Define String to Bits Function
def string_to_bits(str):
    # noqa: D401   "String" is an intended leading word.
    """
    String to Bits Converter.

    Converts a Pythonic string to the string's binary representation.

    Parameters
    ----------
    str:        string
                The string to be converted.

    Returns
    -------
    data:       string
                The binary representation of the
                input string.
    """
    data = (''.join(format(ord(x), 'b') for x in str))
    return (data)


# Define Per-Unit Impedance Formula
def zpu(S, VLL=None, VLN=None):
    r"""
    Per-Unit Impedance Evaluator.

    Evaluates the per-unit impedance value given the per-unit power and voltage
    bases.

    .. math:: Z_{pu}=\frac{V_{LL}^2}{S}

    .. math:: Z_{pu}=\frac{(\sqrt{3}*V_{LN})^2}{S}

    Parameters
    ----------
    S:          float
                The per-unit power base.
    VLL:        float, optional
                The Line-to-Line Voltage; default=None
    VLN:        float, optional
                The Line-to-Neutral Voltage; default=None

    Returns
    -------
    Zbase:      float
                The per-unit impedance base.
    """
    if (VLL == None and VLN == None):
        raise ValueError("ERROR: One voltage must be provided.")
    if VLL != None:
        return (VLL ** 2 / S)
    else:
        return ((_np.sqrt(3) * VLN) ** 2 / S)


# Define Per-Unit Current Formula
def ipu(S, VLL=None, VLN=None, V1phs=None):
    r"""
    Per-Unit Current Evaluator.

    Evaluates the per-unit current value given the per-unit
    power and voltage bases.

    .. math:: I_{pu}=\frac{S}{\sqrt{3}*V_{LL}}

    .. math:: I_{pu}=\frac{S}{3*V_{LN}}

    Parameters
    ----------
    S:          float
                The per-unit power base.
    VLL:        float, optional
                The Line-to-Line Voltage; default=None
    VLN:        float, optional
                The Line-to-Neutral Voltage; default=None
    V1phs:      float, optional
                The voltage base of the single phase system.

    Returns
    -------
    Ibase:      float
                The per-unit current base.
    """
    if (VLL == None and VLN == None):
        raise ValueError("ERROR: One voltage must be provided.")
    if VLL != None:
        return (S / (_np.sqrt(3) * VLL))
    elif VLN != None:
        return (S / (3 * VLN))
    else:
        return (S / V1phs)


# Define Per-Unit Change of Base Function
def puchgbase(quantity, puB_old, puB_new):
    r"""
    Per-Unit Change of Base Function.

    Performs a per-unit change of base operation for the given
    value constrained by the old base and new base.

    .. math:: Z_{pu-new}=Z_{pu-old}*\frac{BASE_{OLD}}{BASE_{NEW}}

    Parameters
    ----------
    quantity:   complex
                Current per-unit value in old base.
    puB_old:    float
                Old per-unit base.
    puB_new:    float
                New per-unit base.

    Returns
    -------
    pu_new:     complex
                New per-unit value.
    """
    pu_new = quantity * puB_old / puB_new
    return (pu_new)


# Define Recomposition Function
def zrecompose(z_pu, S3phs, VLL=None, VLN=None):
    """
    Impedance from Per-Unit System Evaluator.

    Function to reverse per-unit conversion and return the ohmic value
    of an impedance given its per-unit parameters of R and X (as Z).

    Parameters
    ----------
    z_pu:       complex
                The per-unit, complex value corresponding to the
                impedance
    S3phs:      float
                The total three-phase power rating of the system.
    VLL:        float, optional
                The Line-to-Line Voltage; default=None
    VLN:        float, optional
                The Line-to-Neutral Voltage; default=None

    Returns
    -------
    z:          complex
                The ohmic impedance evaluated from the per-unit base.
    """
    # Evaluate the per-unit impedance
    zbase = zpu(S3phs, VLL, VLN)
    # Evaluate the impedance
    z = z_pu * zbase
    return (z)


# Define X/R Recomposition Function
def rxrecompose(x_pu, XoR, S3phs=None, VLL=None, VLN=None):
    """
    Resistance/Reactance from Per-Unit System Evaluator.

    Function to reverse per-unit conversion and return the ohmic value
    of an impedance given its per-unit parameters of X.

    Parameters
    ----------
    x_pu:       float
                The per-unit, complex value corresponding to the
                impedance
    XoR:        float
                The X/R ratio (reactance over impedance).
    S3phs:      float, optional
                The total three-phase power rating of the system.
                If left as None, the per-unit values will be set
                to 1, resulting in an unscaled impedance
    VLL:        float, optional
                The Line-to-Line Voltage; default=None
    VLN:        float, optional
                The Line-to-Neutral Voltage; default=None

    Returns
    -------
    z:          complex
                The ohmic impedance evaluated from the per-unit base.
    """
    # Ensure Absolute Value
    x_pu = abs(x_pu)
    # Find R from X/R
    r_pu = x_pu / XoR
    # Compose into z
    z_pu = r_pu + 1j * x_pu
    # Recompose
    if S3phs == None:
        return (z_pu)
    else:
        z = zrecompose(z_pu, S3phs, VLL, VLN)
        return (z)


# Define Generator Internal Voltage Calculator
def geninternalv(I, Zs, Vt, Vgn=None,Zm=None, Zmp=None, Zmpp=None, Ip=None, Ipp=None):
    """
    Electric Generator Internal Voltage Evaluator.

    Evaluates the internal voltage for a generator given the
    generator's internal impedance and internal mutual coupling
    impedance values.

    Parameters
    ----------
    I:          complex
                The current on the phase of interest.
    Zs:         complex
                The internal impedance of the phase of
                interest in ohms.
    Vt:         complex
                The generator's terminal voltage.
    Vgn:        complex, optional
                The ground-to-neutral connection voltage.
    Zmp:        complex, optional
                The mutual coupling with the first additional
                phase impedance in ohms.
    Zmpp:       complex, optional
                The mutual coupling with the second additional
                phase impedance in ohms.
    Ip:         complex, optional
                The first mutual phase current in amps.
    Ipp:        complex, optional
                The second mutual phase current in amps.

    Returns
    -------
    Ea:         complex
                The internal voltage of the generator.
    """
    # All Parameters Provided
    if Zmp == Zmpp == Ip == Ipp != None:
        if Vgn == None:
            Vgn = 0
        Ea = Zs * I + Zmp * Ip + Zmpp * Ipp + Vt + Vgn
    # Select Parameters Provided
    elif Vgn == Zm == Ip == Ipp == None:
        Ea = Zs * I + Vt
    # Invalid Parameter Set
    else:
        raise ValueError("Invalid Parameter Set")
    return (Ea)


# FFT Coefficient Calculator Function
def funcfft(func, minfreq=60, maxmult=15, complex=False):
    """
    FFT Evaluator for Callable Functions.

    Given the callable function handle for a periodic function,
    evaluates the harmonic components of the function.

    Parameters
    ----------
    func:       function
                Callable function from which to evaluate values.
    minfreq:    float, optional
                Minimum frequency (in Hz) at which to evaluate FFT.
                default=60
    maxmult:    int, optional
                Maximum harmonic (multiple of minfreq) which to
                evaluate. default=15
    complex:    bool, optional
                Control argument to force returned values into
                complex format.

    Returns
    -------
    DC:         float
                The DC offset of the FFT result.
    A:          list of float
                The real components from the FFT.
    B:          list of float
                The imaginary components from the FFT.
    """
    # Apply Nyquist scaling
    NN = 2 * maxmult + 2
    # Determine Time from Fundamental Frequency
    T = 1 / minfreq
    # Generate time range to apply for FFT
    t, dt = _np.linspace(0, T, NN, endpoint=False, retstep=True)
    # Evaluate FFT
    y = _np.fft.rfft(func(t)) / t.size
    # Return Complex Values
    if complex:
        return (y)
    # Split out useful values
    else:
        y *= 2
        return (y[0].real, y[1:-1].real, -y[1:-1].imag)


def sampfft(data, dt, minfreq=60.0, complex=False):
    """
    Sample Dataset FFT Evaluator.

    Given a data array and the delta-t for the data array, evaluates
    the harmonic composition of the data.

    Parameters
    ----------
    data:       numpy.ndarray
                Numpy data array containing 1-D values.
    dt:         float
                Time-difference (delta-t) between data samples.
    minfreq:    float, optional
                Minimum frequency (in Hz) at which to evaluate FFT.
                default=60
    complex:    bool, optional
                Control argument to force returned values into
                complex format.

    Returns
    -------
    DC:         float
                The DC offset of the FFT result.
    A:          list of float
                The real components from the FFT.
    B:          list of float
                The imaginary components from the FFT.
    """
    # Calculate Terms
    FR = 1 / (dt * len(data))
    NN = 1 // (dt * minfreq)
    # Test for Invalid System
    if FR > minfreq:
        raise ValueError(
            "Too few data samples to evaluate FFT at specified minimum "
            "frequency."
        )
    elif FR == minfreq:
        # Evaluate FFT
        y = _np.fft.rfft(data) / len(data)
    else:
        # Slice data array to appropriate fundamental frequency
        cut_data = data[:int(NN)]
        # Evaluate FFT
        y = _np.fft.rfft(cut_data) / len(cut_data)
    # Return Complex Values
    if complex:
        return (y)
    # Split out useful values
    else:
        y *= 2
        return (y[0].real, y[1:-1].real, -y[1:-1].imag)


# Define FFT Plotting Function
def fftplot(dc, real, imag=None, title="Fourier Coefficients"):
    """
    FFT System Plotter.

    Plotting function for FFT (harmonic) values, plots the DC, Real, and
    Imaginary components.

    Parameters
    ----------
    dc:         float
                The DC offset term
    real:       list of float
                Real terms of FFT (cosine terms)
    imag:       list of float, optional
                Imaginary terms of FFT (sine terms)
    title:      str, optional
                String appended to plot title,
                default="Fourier Coefficients"
    
    Returns
    -------
    matplotlib.pyplot:  Plotting object to be used for additional configuration
                        or plotting.
    """
    # Define Range values for plots
    rng = range(1, len(real) + 1, 1)
    xtic = range(0, len(real) + 1, 1)
    # Set up Arguments
    a0x = [0, 0]
    a0y = [0, dc / 2]
    # Plot
    _plt.title(title)
    _plt.plot(a0x, a0y, 'g', label="DC-Term")
    _plt.stem(
        rng,
        real,
        linefmt='r',
        markerfmt='ro',
        label="Real-Terms",
        use_line_collection=True
    )
    if imag != None:
        _plt.stem(
            rng,
            imag,
            linefmt='b',
            markerfmt='bo',
            label="Imaginary-Terms",
            use_line_collection=True
        )
    _plt.xlabel("Harmonics (Multiple of Fundamental)")
    _plt.ylabel("Harmonic Magnitude")
    _plt.axhline(0.0, color='k')
    _plt.legend()
    if (len(xtic) < 50):
        _plt.xticks(xtic)
    return _plt


# Define FFT Composition Plotting Function
def fftsumplot(dc, real, imag=None, freq=60, xrange=None, npts=1000,
               plotall=False, title="Fourier Series Summation"):
    """
    FFT Summation Plotter.

    Function to generate the plot of the sumed FFT results.

    Parameters
    ----------
    dc:         float
                The DC offset term
    real:       list of float
                Real terms of FFT (cosine terms)
    imag:       list of float
                Imaginary terms of FFT (sine terms)
    freq:       float, optional
                Fundamental (minimum nonzero) frequency in Hz,
                default=60
    xrange:     list of float, optional
                List of two floats containing the minimum
                time and the maximum time.
    npts:       int, optional
                Number of time step points, default=1000
    title:      str, optional
                String appended to plot title,
                default="Fourier Series Summation"
    
    Returns
    -------
    matplotlib.pyplot:  Plotting object to be used for additional configuration
                        or plotting.
    """
    # Determine the number (N) of terms
    N = len(real)
    # Determine the system period (T)
    T = 1 / freq
    # Generate Domain Array
    if xrange == None:
        x = _np.linspace(0, T, npts)
    else:
        x = _np.linspace(xrange[0], xrange[1], npts)
    # Initialize output with DC term
    yout = _np.ones(len(x)) * dc
    # Plot each iteration of the Fourier Series
    for k in range(1, N):
        if plotall:
            _plt.plot(x, yout)
        yout += real[k - 1] * _np.cos(k * 2 * _np.pi * x / T)
        if imag != None:
            yout += imag[k - 1] * _np.sin(k * 2 * _np.pi * x / T)
    _plt.plot(x, yout)
    _plt.title(title)
    _plt.xlabel("Time (seconds)")
    _plt.ylabel("Magnitude")
    return _plt


# Define harmonic system generation function
def harmonics(real, imag=None, dc=0, freq=60, domain=None):
    """
    Harmonic Function Generator.

    Generate a function or dataset for a harmonic system
    given the real (cosine), imaginary (sine), and DC
    components of the system.

    Parameters
    ----------
    real:       list of float
                The real (cosine) component coefficients
                for the harmonic system.
    imag:       list of float, optional
                The imaginary (sine) component coefficients
                for the harmonic system.
    dc:         float, optional
                The DC offset for the harmonic system,
                default=0
    freq:       float, optional
                The fundamental frequency of the system in
                Hz, default=60
    domain:     list of float, optional
                Domain of time samples at which to calculate
                the harmonic system, must be array-like, will
                cause function to return numpy array instead
                of function object.

    Returns
    -------
    system:     function
                Function object handle which can be used to
                call the function to evaluate the harmonic
                system at specified times.
    """
    # Validate Inputs
    if not isinstance(real, (list, _np.ndarray)):
        raise ValueError("Argument *real* must be array-like.")
    if imag != None and not isinstance(imag, (list, _np.ndarray)):
        raise ValueError("Argument *imag* must be array-like.")
    # Calculate Omega
    w = 2 * _np.pi * freq

    def _harmonic_(t):
        out = dc
        for k in range(len(real)):
            # Evaluate Current Coefficient
            A = real[k]
            if imag != None:
                B = imag[k]
            else:
                B = 0
            m = k + 1
            # Calculate Output
            out += A * _np.cos(m * w * t) + B * _np.sin(m * w * t)
        # Return Value
        return (out)

    if domain is None:
        return _harmonic_ # Return as callable for external use
    else:
        return _harmonic_(domain)


# Define Single Phase Motor Startup Capacitor Formula
def motorstartcap(V, I, freq=60):
    """
    Single Phase Motor Starting Capacitor Function.

    Function to evaluate a reccomended value for the
    startup capacitor associated with a single phase
    motor.

    Parameters
    ----------
    V:          float
                Magnitude of motor terminal voltage in volts.
    I:          float
                Magnitude of motor no-load current in amps.
    freq:       float, optional
                Motor/System frequency, default=60.

    Returns
    -------
    C:          float
                Suggested capacitance in Farads.
    """
    # Condition Inputs
    I = abs(I)
    V = abs(V)
    # Calculate Capacitance
    return I / (2 * _np.pi * freq * V)


# Define Power Factor Correction Function
def pfcorrection(S, PFold, PFnew, VLL=None, VLN=None, V=None, freq=60):
    """
    Power Factor Correction Function.

    Function to evaluate the additional reactive power and
    capacitance required to achieve the desired power factor
    given the old power factor and new power factor.

    Parameters
    ----------
    S:          float
                Apparent power consumed by the load.
    PFold:      float
                The current (old) power factor, should be a decimal
                value.
    PFnew:      float
                The desired (new) power factor, should be a decimal
                value.
    VLL:        float, optional
                The Line-to-Line Voltage; default=None
    VLN:        float, optional
                The Line-to-Neutral Voltage; default=None
    V:          float, optional
                Voltage across the capacitor, ignores line-to-line
                or line-to-neutral constraints. default=None
    freq:       float, optional
                System frequency, default=60

    Returns
    -------
    C:          float
                Required capacitance in Farads.
    Qc:         float
                Difference of reactive power, (Qc = Qnew - Qold)
    """
    # Condition Inputs
    S = abs(S)
    # Calculate Initial Terms
    Pold = S * PFold
    Qold = _np.sqrt(S ** 2 - Pold ** 2)
    # Evaluate Reactive Power Requirements
    Scorrected = Pold / PFnew
    Qcorrected = _np.sqrt(Scorrected ** 2 - Pold ** 2)
    Qc = Qold - Qcorrected
    # Evaluate Capacitance Based on Voltage Input
    if VLL == VLN == V == None:
        raise ValueError("One voltage must be specified.")
    elif VLN != None:
        C = Qc / (2 * _np.pi * freq * 3 * VLN ** 2)
    else:
        if VLL != None:
            V = VLL
        C = Qc / (2 * _np.pi * freq * V ** 2)
    # Return Value
    return (C, Qc)


# Define Apparent Power / Voltage / Current Relation Function
def acpiv(S=None, I=None, VLL=None, VLN=None, V=None, PF=None):
    """
    AC Power-Voltage-Current Relation Function.

    Relationship function to return apparent power, voltage, or
    current in one of various forms.

    Parameters
    ----------
    S:          complex, optional
                Apparent power, may be single or three-phase,
                specified in volt-amps (VAs)
    I:          complex, optional
                Phase current in amps
    VLL:        complex, optional
                Line-to-Line voltage in volts
    VLN:        complex, optional
                Line-to-Neutral voltage in volts
    V:          complex, optional
                Single-phase voltage in volts

    Returns
    -------
    S:          complex
                Apparent power, returned only if one voltage
                and current is specified
    I:          complex
                Phase current, returned only if one voltage
                and apparent power is specified
    VLL:        complex
                Line-to-Line voltage, returned only if current
                and apparent power specified, returned as set
                with other voltages in form: (VLL, VLN, V)
    VLN:        complex
                Line-to-Neutral voltage, returned only if
                current and apparent power specified, returned
                as set with other voltages in form: (VLL, VLN, V)
    V:          complex
                Single-phase voltage, returned only if current
                and apparent power specified, returned as set
                with other voltages in form: (VLL, VLN, V)
    PF:         float, optional
                Supporting argument to convert floating-point
                apparent power to complex representation.
    """
    # Validate Inputs
    if S == I == None:
        raise ValueError("To few arguments.")
    # Convert Apparent Power to Complex
    if PF != None:
        S = S * PF + 1j * _np.sqrt(S ** 2 - (S * PF) ** 2)
    # Solve Single-Phase
    if V != None:
        if S == None:  # Solve for Apparent Power
            S = V * _np.conj(I)
            return (S)
        else:  # Solve for Current
            I = _np.conj(S / V)
            return (I)
    # Solve Line-to-Line
    elif VLL != None:
        if S == None:  # Solve for Apparent Power
            S = _np.sqrt(3) * VLL * _np.conj(I)
            return (S)
        else:  # Solve for Current
            I = _np.conj(S / (_np.sqrt(3) * VLL))
            return (I)
    # Solve Line-to-Neutral
    elif VLN != None:
        if S == None:  # Solve for Apparent Power
            S = 3 * VLN * _np.conj(I)
            return (S)
        else:  # Solve for Current
            I = _np.conj(S / (3 * VLN))
            return (I)
    # Solve for Voltages
    else:
        V = S / _np.conj(I)
        VLL = S / (_np.sqrt(3) * _np.conj(I))
        VLN = S / (3 * _np.conj(I))
        return (VLL, VLN, V)


# Define Primary Ratio Function
def primary(val, Np, Ns=1, invert=False):
    """
    Electrical Transformer Primary Evaluator.

    Returns a current or voltage value reflected across
    a transformer with a specified turns ratio Np/Ns.
    Converts to the primary side.

    Parameters
    ----------
    val:        complex
                Value to be reflected across transformer.
    Np:         float
                Number of turns on primary side.
    Ns:         float, optional
                Number of turns on secondary side.
    invert:     bool, optional
                Control argument to invert the turns ratio,
                used when reflecting current across a
                voltage transformer, or voltage across a
                current transformer.

    Returns
    -------
    reflection: complex
                The reflected value referred to the primary
                side according to Np and Ns.
    """
    if invert:
        return (val * Ns / Np)
    return (val * Np / Ns)


# Define Secondary Ratio Function
def secondary(val, Np, Ns=1, invert=False):
    """
    Electrical Transformer Secondary Evaluator.

    Returns a current or voltage value reflected across
    a transformer with a specified turns ratio Np/Ns.
    Converts to the secondary side.

    Parameters
    ----------
    val:        complex
                Value to be reflected across transformer.
    Np:         float
                Number of turns on primary side.
    Ns:         float, optional
                Number of turns on secondary side.
    invert:     bool, optional
                Control argument to invert the turns ratio,
                used when reflecting current across a
                voltage transformer, or voltage across a
                current transformer.

    Returns
    -------
    reflection: complex
                The reflected value referred to the secondary
                side according to Np and Ns.
    """
    if invert:
        return (val * Np / Ns)
    return (val * Ns / Np)


def tap_changing_transformer(Vgen, Vdis, Pload, Qload, R, X):
    r"""
    Calculate Turn Ratio of Load Tap Changing Transformer.

    The purpose of a tap changer is to regulate the output voltage of a
    transformer. It does this by altering the number of turns in one winding and
    thereby changing the turns ratio of the transformer
    
    .. math:: \sqrt{\frac{Vgen^2}{Vgen \cdot Vdis - R \cdot P - X \cdot Q}}

    Parameters
    ----------
    Vgen:   float
            Generating station voltage
    Vdis:   float
            Distribution network voltage
    Pload:  float
            Transmission line load active power in Watt
    Qload:  float
            Transmission line load reactive power in VAR
    R:      float
            Resistance of transmission line
    X:      float
            Reactance of transmission line
    
    Returns
    -------
    ts:     float
            Turns ration of transformer
    """
    # Evaluate the turns ratio
    ts = (Vgen*Vgen) / (Vgen*Vdis - (R * Pload + X * Qload) )
    return pow(ts, 0.5)

def suspension_insulators(number_capacitors, capacitance_ratio, Voltage):
    r"""
    Discrete Capacitors Voltage in a Suspension Insulator Strain.

    To perform the calculations described here, the following formulas are
    satisfied, and used to construct a matrix used to solve for
    :math:`V_i \text{i in range(1,n)}`.

    .. math:: \sum_{i=1}^{n-2} V_{i} + V_{n-1} \cdot (1+m) - V_{n} \cdot m=0

    .. math:: \sum_{i=1}^{n} V_{i} = V_{\text{transmission line}}
    
    .. image:: /static/SuspensionInuslator.png
    
    `Additional Information
    <https://electrical-engineering-portal.com/download-center/books-and-guides/power-substations/insulator-pollution>`_
    
    Parameters
    ----------
    number_capacitors:  int
                        Number of disk capacitors hung to transmission line
    capacitance_ratio:  float
                        Ratio of disk capacitance and pin to pole air capacitance
    Voltage:            float
                        Voltage difference between the transmission line and ground
    
    Returns
    -------
    string_efficiency:          float
                                String efficiency of capacitive disks
    capacitor_disk_voltages:    float
                                Voltage across each capacitive disk starting
                                from top to bottom
    """
    m = _np.zeros((number_capacitors, number_capacitors))
    # Iterate over capacitors
    for i in range(number_capacitors - 1):
        # Iterate over capacitors
        for j in range(number_capacitors - 1):
            # If inner iteration is less than outer iteration
            if i >= j:
                m[i, j] = 1 / capacitance_ratio

    for i in range(number_capacitors - 1):
        m[i, i] = (1 + 1 / capacitance_ratio)

        m[i, i + 1] = -1

    m[number_capacitors - 1, :] = 1

    v = _np.zeros((number_capacitors, 1))

    v[number_capacitors - 1, 0] = Voltage

    capacitor_disk_voltages = _np.matmul(_np.linalg.inv(m), v)

    string_efficiency = (Voltage * 100) / (number_capacitors * capacitor_disk_voltages[-1, 0])

    return capacitor_disk_voltages, string_efficiency

# Define Natural Frequency/Resonant Frequency Calculator
def natfreq(C, L, Hz=True):
    r"""
    Natural Frequency Evaluator.

    Evaluates the natural frequency (resonant frequency) of a circuit given the
    circuit's C and L values. Defaults to returning values in Hz, but may also
    return in rad/sec.

    .. math:: freq=\frac{1}{\sqrt{L*C}*(2*\pi)}

    Parameters
    ----------
    C:          float
                Capacitance Value in Farads.
    L:          float
                Inductance in Henries.
    Hz:         bool, optional
                Control argument to set return value in either
                Hz or rad/sec; default=True.

    Returns
    -------
    freq:       float
                Natural (Resonant) frequency, will be in Hz if
                argument *Hz* is set True (default), or rad/sec
                if argument is set False.
    """
    # Evaluate Natural Frequency in rad/sec
    freq = 1 / _np.sqrt(L * C)
    # Convert to Hz as requested
    if Hz:
        freq = freq / (2 * _np.pi)
    return (freq)


# Define Voltage/Current Unbalance Equation
def unbalance(A, B, C, all=False):
    """
    Voltage/Current Unbalance Function.

    Performs a voltage/current unbalance calculation
    to determine the maximum current/voltage
    unbalance. Returns result as a decimal percentage.

    Parameters
    ----------
    A:          float
                Phase-A value
    B:          float
                Phase-B value
    C:          float
                Phase-C value
    all:        bool, optional
                Control argument to require function
                to return all voltage/current unbalances.

    Returns
    -------
    unbalance:  float
                The unbalance as a percentage of the
                average. (i.e. 80% = 0.8)
    """
    # Condition Inputs
    A = abs(A)
    B = abs(B)
    C = abs(C)
    # Gather Average
    avg = (A + B + C) / 3
    # Determine Variance
    dA = abs(A - avg)
    dB = abs(B - avg)
    dC = abs(C - avg)
    # Gather Maximum Variation
    mx = max(dA, dB, dC)
    # Calculate Maximum Variation
    unbalance = mx / avg
    # Return Results
    if all:
        return (dA / avg, dB / avg, dC / avg)
    else:
        return (unbalance)


# Define Cosine Filter Function
def cosfilt(arr, Srate, domain=False):
    """
    Cosine Filter Function.

    Cosine Filter function for filtering a dataset
    representing a sinusoidal function with or without
    harmonics to evaluate the fundamental value.

    Parameters
    ----------
    arr:        numpy.ndarray
                The input data array.
    Srate:      int
                Sampling rate for dataset, specified in
                number of values per fundamental cycle.
    domain:     bool, optional
                Control argument to force return of
                x-axis array for the filtered data.

    Returns
    -------
    cosf:       numpy.ndarray
                Cosine-filtered data
    xarray:     numpy.ndarray
                X-axis array for the filtered data.
    """
    # Evaluate index set
    ind = _np.arange(Srate - 1, len(arr) - 1)

    # Define Cosine Coefficient Function
    def cos(k, Srate):
        return (_np.cos(2 * _np.pi * k / Srate))

    # Calculate Constant
    const = 2 / Srate
    # Iteratively Calculate
    cosf = 0
    for k in range(0, Srate - 1):
        slc = (ind - (Srate - 1)) + k
        cosf += cos(k, Srate) * arr[slc]
    # Scale
    cosf = const * cosf
    # Return Cosine-Filtered Array
    if domain:
        xarray = _np.linspace(Srate + Srate / 4 - 1, len(arr) - 1, len(cosf))
        xarray = xarray / Srate
        return (cosf, xarray)
    return (cosf)


# Define Sine Filter Function
def sinfilt(arr, Srate, domain=False):
    """
    Sine Filter Function.

    Sine Filter function for filtering a dataset
    representing a sinusoidal function with or without
    harmonics to evaluate the fundamental value.

    Parameters
    ----------
    arr:        numpy.ndarray
                The input data array.
    Srate:      int
                Sampling rate for dataset, specified in
                number of values per fundamental cycle.
    domain:     bool, optional
                Control argument to force return of
                x-axis array for the filtered data.

    Returns
    -------
    sinf:       numpy.ndarray
                Sine-filtered data
    xarray:     numpy.ndarray
                X-axis array for the filtered data.
    """
    # Evaluate index set
    ind = _np.arange(Srate - 1, len(arr) - 1)

    # Define Cosine Coefficient Function
    def sin(k, Srate):
        return (_np.sin(2 * _np.pi * k / Srate))

    # Calculate Constant
    const = 2 / Srate
    # Iteratively Calculate
    sinf = 0
    for k in range(0, Srate - 1):
        slc = (ind - (Srate - 1)) + k
        sinf += sin(k, Srate) * arr[slc]
    # Scale
    sinf = const * sinf
    # Return Cosine-Filtered Array
    if domain:
        xarray = _np.linspace(Srate + Srate / 4 - 1, len(arr) - 1, len(sinf))
        xarray = xarray / Srate
        return (sinf, xarray)
    return (sinf)


# Define Characteristic Impedance Calculator
def characterz(R, G, L, C, freq=60):
    r"""
    Characteristic Impedance Calculator.

    Function to evaluate the characteristic
    impedance of a system with specefied
    line parameters as defined. System uses
    the standard characteristic impedance
    equation :eq:`Zc`.

    .. math:: Z_c = \sqrt{\frac{R+j\omega L}{G+j\omega C}}
       :label: Zc

    Parameters
    ----------
    R:          float
                Resistance in ohms.
    G:          float
                Conductance in mhos (siemens).
    L:          float
                Inductance in Henries.
    C:          float
                Capacitance in Farads.
    freq:       float, optional
                System frequency in Hz, default=60

    Returns
    -------
    Zc:         complex
                Charcteristic Impedance of specified line.
    """
    # Evaluate omega
    w = 2 * _np.pi * freq
    # Evaluate Zc
    Zc = _np.sqrt((R + 1j * w * L) / (G + 1j * w * C))
    return (Zc)

# Define propagation_constants for long transmission line
def propagation_constants(z, y, length):
    r"""
    Transaction Line Propagation Constant Calculator.

    This function will evaluate the propagation constants for a long transmission
    line whose properties are governed by the differential equation:
    
    .. math:: \frac{d^2V}{dx^2} = \gamma V
    
    From the above equation, the following formulas are derived to evaluate the
    desired constants.
    
    .. math:: \gamma = \sqrt( z * y )
    
    .. math:: Z_{\text{surge}} = \sqrt( z / y )
    
    .. math:: \alpha = \Re{ \gamma }
    
    .. math:: \beta = \Im{ \gamma }
    
    Parameters
    ----------
    z:              complex
                    Impedence of the transmission line: R+j*2*pi*f*L
    y:              complex
                    Admitance of the transmission line g+j*2*pi*f*C
        
    Returns
    -------
    params:    dict
               Dictionary of propagation constants including:
                    
                         gamma:   Propagation constant
                         zc:            Surge impedance
                         alpha:      Attenuation constant
                         beta:        Imaginary portion of gamma
    """
    # Validate the line length is substantial enough for calculation
    if not (length > 500):
        raise ValueError(
            "Long transmission line length should be grater than 500km"
        )
    gamma = _np.sqrt(z * y)
    alpha = gamma.real
    beta = gamma.imag
    zc = _np.sqrt(z / y)
    params = {
        'gamma': gamma,
        'alpha': alpha,
        'beta': beta,
        'Surge_impedance': zc
    }

    return params


# Define Simple Transformer Phase Shift Function
def xfmphs(style="DY", shift=30):
    """
    Electrical Transformer Phase-Shift Calculator.

    Use with transformer orientation to evaluate the phase-shift across a
    transformer. For example, find the phase shift for a Delta-Wye transformer
    as seen from the delta side.

    Parameters
    ----------
    style:      {'DY','YD','DD','YY'}, optional
                String that denotes the transformer
                orientation. default='DY'
    shift:      float, optional
                Transformer angle shift, default=30

    Returns
    -------
    phase:      complex
                Phasor including the phase shift and
                positive or negative characteristic.

    Examples
    --------
    >>> import electricpy as ep
    >>> # Find shift of Delta-Wye Transformer w/ 30° shift
    >>> shift = ep.xfmphs(style="DY",shift=30)
    >>> ep.cprint(shift)
    1.0 ∠ 30.0°
    """
    # Define Direction Dictionary
    orientation = {
        "DY": 1,
        "YD": -1,
        "DD": 0,
        "YY": 0,
    }
    # Find Direction
    v = orientation[style.upper()]
    # Calculate Shift
    phase = _np.exp(1j * _np.radians(v * abs(shift)))
    # Return
    return (phase)


# Define Induction Machine Thevenin Voltage Calculator
def indmachvth(Vas, Rs, Lm, Lls=0, Ls=None, freq=60, calcX=True):
    r"""
    Induction Machine Thevenin Voltage Calculator.

    Function to calculate the Thevenin equivalent voltage of an
    induction machine given a specific set of parameters.

    .. math:: V_{th}=\frac{j\omega L_m}{R_s+j\omega(L_{ls}+L_m)}V_{as}

    where:

    .. math:: \omega = \omega_{es} = 2\pi\cdot f_{\text{electric}}

    Parameters
    ----------
    Vas:        complex
                Terminal Stator Voltage in Volts
    Rs:         float
                Stator resistance in ohms
    Lm:         float
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Vth:        complex
                Thevenin-Equivalent voltage (in volts) of induction
                machine described.

    See Also
    --------
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if calcX:  # Convert Inductances to Reactances
        w = 2 * _np.pi * freq
        Lm *= w
        Lls *= w
    # Calculate Thevenin Voltage, Return
    Vth = 1j * Lm / (Rs + 1j * (Lls + Lm)) * Vas
    return (Vth)


# Define Induction Machine Thevenin Impedance Calculator
def indmachzth(Rs, Lm, Lls=0, Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Thevenin Impedance Calculator.

    Function to calculate the Thevenin equivalent impedance of an
    induction machine given a specific set of parameters.

    .. math::
       Z_{th} = \frac{(R_s+j\omega L_{ls})j\omega L_m}
       {R_s+j\omega(L_{ls}+L_m)}+j\omega L_{lr}

    where:

    .. math:: \omega = \omega_{es} = 2\pi\cdot f_{\text{electric}}

    Parameters
    ----------
    Rs:         float
                Stator resistance in ohms
    Lm:         float
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Zth:        complex
                Thevenin-Equivalent impedance (in ohms) of induction
                machine described.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if calcX:  # Convert Inductances to Reactances
        w = 2 * _np.pi * freq
        Lm *= w
        Lls *= w
        Llr *= w
    # Calculate Thevenin Impedance
    Zth = (Rs + 1j * Lls) * (1j * Lm) / (Rs + 1j * (Lls + Lm)) + 1j * Llr
    return (Zth)


# Define Induction Machine Mechancal Power Calculator
def indmachpem(slip, Rr, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0,
               Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Mechanical Power Calculator for Induction Machines.

    Function to calculate the mechanical power using the thevenin
    equivalent circuit terms.

    .. math::
       P_{em}=\frac{|V_{th_{\text{stator}}}|^2\cdot\frac{R_r}{slip}}
       {\left[\left(\frac{R_r}{slip}+R_{th_{\text{stator}}}\right)^2
       +X_{th_{\text{stator}}}^2\right]\cdot\omega_{es}}\cdot(1-slip)

    Parameters
    ----------
    slip:       float
                The mechanical/electrical slip factor of the
                induction machine.
    Rr:         float
                Rotor resistance in ohms
    Vth:        complex, optional
                Thevenin-equivalent stator voltage of the
                induction machine, may be calculated internally
                if given stator voltage and machine parameters.
    Zth:        complex, optional
                Thevenin-equivalent inductance (in ohms) of the
                induction machine, may be calculated internally
                if given machine parameters.
    Vas:        complex, optional
                Terminal Stator Voltage in Volts
    Rs:         float, optional
                Stator resistance in ohms
    Lm:         float, optional
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Pem:        float
                Power (in watts) that is produced or consumed
                by the mechanical portion of the induction machine.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    w = 2 * _np.pi * freq
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Vth == None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth == None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Use Terms to Calculate Pem
    Rth = Zth.real
    Xth = Zth.imag
    Pem = (abs(Vth) ** 2 * Rr / slip) / (((Rr / slip + Rth) ** 2 + Xth ** 2) * w) * (1 - slip)
    return (Pem)


# Define Induction Machine Torque Calculator
def indmachtem(slip, Rr, p=0, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0,
               Llr=0, Ls=None, Lr=None, wsyn=None, freq=60, calcX=True):
    r"""
    Induction Machine Torque Calculator.

    Calculate the torque generated or consumed by an induction
    machine given the machine parameters of Vth and Zth by use
    of the equation below.

    .. math::
       T_{em}=\frac{3|V_{th_{\text{stator}}}|^2}
       {\left[\left(\frac{R_r}{slip}+R_{th_{\text{stator}}}\right)^2
       +X_{th_{\text{stator}}}\right]}\frac{R_r}{slip*\omega_{sync}}

    where:

    .. math::
       \omega_{sync}=\frac{\omega_{es}}{\left(\frac{poles}{2}\right)}

    Parameters
    ----------
    slip:       float
                The mechanical/electrical slip factor of the
                induction machine.
    Rr:         float
                Rotor resistance in ohms
    p:          int, optional
                Number of poles in the induction machine
    Vth:        complex, optional
                Thevenin-equivalent stator voltage of the
                induction machine, may be calculated internally
                if given stator voltage and machine parameters.
    Zth:        complex, optional
                Thevenin-equivalent inductance (in ohms) of the
                induction machine, may be calculated internally
                if given machine parameters.
    Vas:        complex, optional
                Terminal Stator Voltage in Volts
    Rs:         float, optional
                Stator resistance in ohms
    Lm:         float, optional
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    wsync:       float, optional
                Synchronous speed in rad/sec, may be specified
                directly as a replacement of p (number of poles).
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Tem:        float
                Torque (in Newton-meters) that is produced or consumed
                by the mechanical portion of the induction machine.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    w = 2 * _np.pi * freq
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if p != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (p / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if not any((p, wsyn)):
        raise ValueError("Poles or Synchronous Speed must be specified.")
    if Vth == None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth == None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Use Terms to Calculate Pem
    Rth = Zth.real
    Xth = Zth.imag
    Tem = 3 * abs(Vth) ** 2 / ((Rr / slip + Rth) ** 2 + Xth) * Rr / (slip * wsyn)
    return (Tem)


# Define Induction Machine Peak Slip Calculator
def indmachpkslip(Rr, Zth=None, Rs=0, Lm=0, Lls=0, Llr=0, Ls=None,
                  Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Slip at Peak Torque Calculator.

    Function to calculate the slip encountered by an induction machine
    with the parameters specified when the machine is generating peak
    torque. Uses formula as shown below.

    .. math:: \text{slip} = \frac{R_r}{|Z_{th}|}

    where:

    .. math::
       Z_{th} = \frac{(R_s+j\omega L_{ls})j\omega L_m}
       {R_s+j\omega(L_{ls}+L_m)}+j\omega L_{lr}

    Parameters
    ----------
    Rr:         float
                Rotor resistance in ohms
    Zth:        complex, optional
                Thevenin-equivalent inductance (in ohms) of the
                induction machine, may be calculated internally
                if given machine parameters.
    Rs:         float, optional
                Stator resistance in ohms
    Lm:         float, optional
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    s_peak:     float
                The peak slip for the induction machine described.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    w = 2 * _np.pi * freq
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Zth == None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Calculate Peak Slip
    s_peak = Rr / abs(Zth)
    return (s_peak)


# Define Induction Machine Phase-A, Rotor Current Calculator
def indmachiar(Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0,
               Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Rotor Current Calculator.

    Calculation function to find the phase-A, rotor current for an
    induction machine given the thevenin voltage and impedance.

    This current is calculated using the following formulas:

    .. math:: I_{a_{\text{rotor}}} = \frac{V_{th}}{|Z_{th}|+Z_{th}}

    where:

    .. math:: V_{th}=\frac{j\omega L_m}{R_s+j\omega(L_{ls}+L_m)}V_{as}

    .. math::
       Z_{th} = \frac{(R_s+j\omega L_{ls})j\omega L_m}
       {R_s+j\omega(L_{ls}+L_m)}+j\omega L_{lr}

    .. math:: \omega = \omega_{es} = 2\pi\cdot f_{\text{electric}}

    Parameters
    ----------
    Vth:        complex, optional
                Thevenin-equivalent stator voltage of the
                induction machine, may be calculated internally
                if given stator voltage and machine parameters.
    Zth:        complex, optional
                Thevenin-equivalent inductance (in ohms) of the
                induction machine, may be calculated internally
                if given machine parameters.
    Vas:        complex, optional
                Terminal Stator Voltage in Volts
    Rs:         float, optional
                Stator resistance in ohms
    Lm:         float, optional
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Iar:        complex
                The rotor, phase-A current in amps.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    w = 2 * _np.pi * freq
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if p != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (p / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Vth == None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth == None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Calculate Rotor Current
    Iar = Vth / (Zth.real + Zth)
    return (Iar)


# Define Induction Machine Peak Torque Calculator
def indmachpktorq(Rr, s_pk=None, Iar=None, Vth=None, Zth=None, Vas=0, Rs=0,
                  Lm=0, Lls=0, Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Peak Torque Calculator.

    Calculation function to find the peak torque for an
    induction machine given the thevenin voltage and impedance.

    This current is calculated using the following formulas:

    .. math::
       T_{em}=(|I_{a_{\text{rotor}}}|)^2\cdot\frac{R_r}
       {\text{slip}_{\text{peak}}}

    where:

    .. math:: I_{a_{\text{rotor}}} = \frac{V_{th}}{|Z_{th}|+Z_{th}}

    .. math:: V_{th}=\frac{j\omega L_m}{R_s+j\omega(L_{ls}+L_m)}V_{as}

    .. math::
       Z_{th} = \frac{(R_s+j\omega L_{ls})j\omega L_m}
       {R_s+j\omega(L_{ls}+L_m)}+j\omega L_{lr}

    .. math:: \omega = \omega_{es} = 2\pi\cdot f_{\text{electric}}

    Parameters
    ----------
    Rr:         float
                Rotor resistance in Ohms
    s_pk:       float, optional
                Peak induction machine slip, may be calculated
                internally if remaining machine characteristics are
                provided.
    Iar:        complex, optional
                Phase-A, Rotor Current in Amps, may be calculated
                internally if remaining machine characteristics are
                provided.
    Vth:        complex, optional
                Thevenin-equivalent stator voltage of the
                induction machine, may be calculated internally
                if given stator voltage and machine parameters.
    Zth:        complex, optional
                Thevenin-equivalent inductance (in ohms) of the
                induction machine, may be calculated internally
                if given machine parameters.
    Vas:        complex, optional
                Terminal Stator Voltage in Volts
    Rs:         float, optional
                Stator resistance in ohms
    Lm:         float, optional
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Tpk:        float
                Peak torque of specified induction machine in
                newton-meters.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    indmachstarttorq:   Induction Machine Starting Torque Calculator
    """
    # Condition Inputs
    w = 2 * _np.pi * freq
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if p != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (p / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Vth == None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth == None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    if Iar == None:
        if not all((Vth, Zth)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Ias
        Iar = indmachiar(Vth=Vth, Zth=Zth)
    if s_pk == None:
        if not all((Rr, Zth)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Peak Slip
        s_pk = indmachpkslip(Rr=Rr, Zth=Zth)
    # Use Terms to Calculate Peak Torque
    Tpk = abs(Iar) ** 2 * Rr / s_pk
    return (Tpk)


# Define Induction Machine Starting Torque Calculator
def indmachstarttorq(Rr, Iar=None, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0,
                     Lls=0, Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Starting Torque Calculator.

    Calculation function to find the starting torque for an
    induction machine given the thevenin voltage and impedance.

    This current is calculated using the following formulas:

    .. math::
       T_{em}=(|I_{a_{\text{rotor}}}|)^2\cdot\frac{R_r}
       {\text{slip}_{\text{peak}}}

    where:

    .. math:: \text{slip} = 1

    .. math::
       I_{a_{\text{rotor}}} = \frac{V_{th}}{\frac{R_r}{\text{slip}}+Z_{th}}

    .. math:: V_{th}=\frac{j\omega L_m}{R_s+j\omega(L_{ls}+L_m)}V_{as}

    .. math::
       Z_{th} = \frac{(R_s+j\omega L_{ls})j\omega L_m}
       {R_s+j\omega(L_{ls}+L_m)}+j\omega L_{lr}

    .. math:: \omega = \omega_{es} = 2\pi\cdot f_{\text{electric}}

    Parameters
    ----------
    Rr:         float
                Rotor resistance in Ohms
    Iar:        complex, optional
                Phase-A, Rotor Current in Amps, may be calculated
                internally if remaining machine characteristics are
                provided.
    Vth:        complex, optional
                Thevenin-equivalent stator voltage of the
                induction machine, may be calculated internally
                if given stator voltage and machine parameters.
    Zth:        complex, optional
                Thevenin-equivalent inductance (in ohms) of the
                induction machine, may be calculated internally
                if given machine parameters.
    Vas:        complex, optional
                Terminal Stator Voltage in Volts
    Rs:         float, optional
                Stator resistance in ohms
    Lm:         float, optional
                Magnetizing inductance in Henrys
    Lls:        float, optional
                Stator leakage inductance in Henrys, default=0
    Llr:        float, optional
                Rotor leakage inductance in Henrys, default=0
    Ls:         float, optional
                Stator inductance in Henrys
    Lr:         float, optional
                Rotor inductance in Henrys
    freq:       float, optional
                System (electrical) frequency in Hz, default=60
    calcX:      bool, optional
                Control argument to force system to calculate
                system reactances with system frequency, or to
                treat them as previously-calculated reactances.
                default=True

    Returns
    -------
    Tstart:     float
                Peak torque of specified induction machine in
                newton-meters.

    See Also
    --------
    indmachvth:         Induction Machine Thevenin Voltage Calculator
    indmachzth:         Induction Machine Thevenin Impedance Calculator
    indmachpem:         Induction Machine Electro-Mechanical Power Calculator
    indmachtem:         Induction Machine Electro-Mechanical Torque Calculator
    indmachpkslip:      Induction Machine Peak Slip Calculator
    indmachpktorq:      Induction Machine Peak Torque Calculator
    indmachiar:         Induction Machine Phase-A Rotor Current Calculator
    """
    # Condition Inputs
    w = 2 * _np.pi * freq
    if Ls != None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr != None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if p != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (p / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Slip is 1 (one) for starting
    slip = 1
    # Test for Valid Input Set
    if Vth == None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth == None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    if Iar == None:
        if not all((Vth, Zth)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Ias
        Iar = Vth / (Rr / slip + Zth)
    # Use Terms to Calculate Peak Torque
    Tstart = abs(Iar) ** 2 * Rr / slip
    return (Tstart)


# Define Induction Machine Stator Torque Calculator
def pstator(Pem, slip):
    r"""
    Stator Power Calculator for Induction Machine.

    Given the electromechanical power and the slip,
    this function will calculate the power related to the
    stator (provided or consumed).

    .. math:: P_s=\frac{P_{em}}{1-\text{slip}}

    Parameters
    ----------
    Pem:        float
                Electromechanical power in watts.
    slip:       float
                Slip factor in rad/sec.

    Returns
    -------
    Ps:         float
                Power related to the stator in watts.

    See Also
    --------
    protor:         Rotor Power Calculator for Induction Machines
    """
    # Calculate and Return
    Ps = Pem / (1 - slip)
    return (Ps)


# Define Induction Machine Rotor Torque Calculator
def protor(Pem, slip):
    r"""
    Rotor Power Calculator for Induction Machine.

    Given the electromechanical power and the slip,
    this function will calculate the power related to the
    rotor (provided or consumed).

    .. math:: P_r=-\text{slip}\cdot\frac{P_{em}}{1-\text{slip}}

    Parameters
    ----------
    Pem:        float
                Electromechanical power in watts.
    slip:       float
                Slip factor in rad/sec.

    Returns
    -------
    Pr:         float
                Power related to the rotor in watts.

    See Also
    --------
    pstator:         Stator Power Calculator for Induction Machines
    """
    # Calculate and Return
    Pr = -slip * (Pem / (1 - slip))
    return (Pr)


# Define De Calculator for Transmission Lines
def de_calc(rho, freq=60):
    r"""
    De Transmission Line Value Calculator.

    Simple calculator to find the De value for a line
    with particular earth resistivity (rho).

    .. math:: D_e=D_{e_{\text{constant}}}\sqrt{\frac{\rho}{freq}}

    Parameters
    ----------
    rho:        float
                Earth resistivity (in ohm-meters), may also
                be passed a string in the set: {SEA, SWAMP,
                AVG,AVERAGE,DAMP,DRY,SAND,SANDSTONE}
    freq:       float, optional
                System frequency in Hertz, default=60
    """
    # If Descriptive String Provided, Use to Determine Rho
    if isinstance(rho, str):
        rho = rho.upper()
        try:
            rho = RHO_VALUES[rho]
        except KeyError:
            raise ValueError("Invalid Earth Resistivity string try to select \
            from set of (SEA, SWAMP, AVG, AVERAGE, DAMP, DRY, SAND, SANDSTONE")
    # Calculate De
    De = De0 * _np.sqrt(rho / freq)
    return (De)


# Define Impedance Per Length Calculator
def zperlength(Rd=None, Rself=None, Rac=None, Rgwac=None, De=None,
               rho="AVG", Ds=None, Dsgw=None, dia_gw=None, Dab=None,
               Dbc=None, Dca=None, Dagw=None, Dbgw=None, Dcgw=None,
               resolve=True, freq=60):
    """
    Transmission Line Impedance (RL) Calculator.

    Simple impedance matrix generator to provide the full
    impedance per length matrix.

    Parameters
    ----------
    Rd:         float, optional
                Resistance Rd term in ohms, will be generated
                automatically if set to None, default=None
    Rself:      float, optional
                Self Resistance term in ohms.
    Rac:        float, optional
                AC resistance in ohms.
    Rgwac:      float, optional
                Ground-Wire AC resistance in ohms.
    De:         float, optional
                De term, in feet, if None provided, and `rho`
                parameter is specified, will interpretively be
                calculated.
    rho:        float, optional
                Earth resistivity in ohm-meters. default="AVG"
    Ds:         float, optional
                Distance (self) for each phase conductor in feet,
                commonly known as GMD.
    Dsgw:       float, optional
                Distance (self) for the ground wire conductor in
                feet, commonly known as GMD.
    dia_gw:     float, optional
                Ground-Wire diameter in feet, may be used to
                calculate an approximate Dsgw if no Dsgw is provided.
    Dab:        float, optional
                Distance between phases A and B, in feet.
    Dbc:        float, optional
                Distance between phases B and C, in feet.
    Dca:        float, optional
                Distance between phases C and A, in feet.
    Dagw:       float, optional
                Distance between phase A and ground conductor, in feet.
    Dbgw:       float, optional
                Distance between phase B and ground conductor, in feet.
    Dcgw:       float, optional
                Distance between phase C and ground conductor, in feet.
    resolve:    bool, optional
                Control argument to specify whether the resultant
                ground-wire inclusive per-length impedance matrix
                should be reduced to a 3x3 equivalent matrix.
                default=True
    freq:       float, optional
                System frequency in Hertz.
    """
    # Start with Empty Arrays
    Rperlen = 0
    Lperlen = 0
    # Generate Rd
    if Rd == None:
        Rd = freq * carson_r
    # Generate Dsgw if Not Provided
    if Dsgw == None and dia_gw != None:
        Dsgw = _np.exp(-1 / 4) * dia_gw / 2
    # Generate Real Part
    if Rd > 0:
        # Generate Rself if not Provided
        if Rself == None:
            # Validate Inputs
            if not all((Rd, Rac)):
                raise ValueError("Too few arguments")
            Rself = Rac + Rd
        # Generate RperLength Matrix
        Rperlen = _np.array([
            [Rself, Rd, Rd],
            [Rd, Rself, Rd],
            [Rd, Rd, Rself]
        ])
        # Add GW effects If Necessary
        if all((Rgwac, Dsgw, Dagw, Dbgw, Dcgw)):
            # Calculate Rselfgw
            Rselfgw = Rgwac + Rd
            # Append Right-Most Column
            Rperlen = _np.append(Rperlen,
                                 [[Rd], [Rd], [Rd]], axis=1)
            # Append New Row
            Rperlen = _np.append(Rperlen,
                                 [[Rd, Rd, Rd, Rselfgw]], axis=0)
    # Generate Imaginary Part
    if any((De, Ds, rho)):
        # Validate Inputs
        if not all((Dab, Dbc, Dca)):
            raise ValueError("Distance Terms [Dab,Dbc,Dca] Required")
        if Ds == None:
            raise ValueError("Distance Self (Ds) Required")
        # De must be generated
        if De == None:
            if rho == None:
                raise ValueError("Too few arguments")
            De = de_calc(rho, freq)
        # Generate LperLength Matrix
        Lperlen = _np.array([
            [_np.log(De / Ds), _np.log(De / Dab), _np.log(De / Dca)],
            [_np.log(De / Dab), _np.log(De / Ds), _np.log(De / Dbc)],
            [_np.log(De / Dca), _np.log(De / Dbc), _np.log(De / Ds)]
        ])
        # Add GW effects If Necessary
        if all((Rgwac, Dsgw, Dagw, Dbgw, Dcgw)):
            # Append Right-Most Column
            Lperlen = _np.append(Lperlen,
                                 [[_np.log(De / Dagw)], [_np.log(De / Dbgw)], [_np.log(De / Dcgw)]],
                                 axis=1)
            # Append New Row
            Lperlen = _np.append(Lperlen,
                                 [[_np.log(De / Dagw), _np.log(De / Dbgw),
                                   _np.log(De / Dcgw), _np.log(De / Dsgw)]], axis=0)
        Lperlen = Lperlen * (1j * u0 * freq)
    # Add Real and Imaginary Parts
    Zperlen = Rperlen + Lperlen
    # Resolve to 3x3 Matrix if Needed
    if resolve and all((Rgwac, Dsgw, Dagw, Dbgw, Dcgw)):
        # Perform Slicing to Retrieve Useful Arrays
        Za = Zperlen[:3, :3]
        Zb = Zperlen[:3, 3:4]
        Zc = Zperlen[3:4, :3]
        Zd = Zperlen[3:4, 3:4]
        # Calculate New (3x3) Equivalent Zperlen
        Zperlen = Za - _np.dot(Zb, _np.dot(_np.linalg.inv(Zd), Zc))
    return (Zperlen)


# Define Transposition Matrix Formula
def transposez(Zeq, fabc=1 / 3, fcab=1 / 3, fbca=1 / 3, linelen=1):
    r"""
    Transmission Matrix Equivalent Transposition Calculator.

    Given the impedance matrix and the percent of the line spent
    in each transposition relation (ABC, CAB, and BCA).

    .. math::
       f_{abc}Z_{eq}+f_{cab}R_p^{-1}\cdot Z_{eq}\cdot R_p+
       f_{bca}Z_{eq}R_p\cdot Z_{eq}\cdot R_p^{-1}

    where:

    .. math:
       R_p=\begin{bmatrix}\\
       0 & 0 & 1 \\
       1 & 0 & 0 \\
       0 & 1 & 0 \\
       \end{bmatrix}

    Parameters
    ----------
    Zeq:        array_like
                Per-Length (or total length) line impedance in ohms.
    fabc:       float, optional
                Percentage of line set with phase relation ABC,
                default=1/3
    fcab:       float, optional
                Percentage of line set with phase relation CAB,
                default=1/3
    fbca:       float, optional
                Percentage of line set with phase relation BCA,
                default=1/3
    linelen:    Length of line (unitless), default=1
    """
    # Condition Input
    Zeq = _np.asarray(Zeq)
    # Define Rp Array
    Rp = _np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    # Define Inverse Rp Array
    _Rp = _np.linalg.inv(Rp)
    Zeq = fabc * Zeq + fcab * (_Rp.dot(Zeq.dot(Rp))) + fbca * (Rp.dot(Zeq.dot(_Rp)))
    Zeq = Zeq * linelen
    return (Zeq)


# Define GMD Calculator
def gmd(Ds, *args):
    r"""
    GMD (Geometric Mean Distance) Calculator.

    Calculates the GMD (Geometric Mean Distance) for a system
    with the parameters of a list of arguments.

    .. math:: GMD=(D_s*D_1*\ddot*D_n)^{\frac{1}{1+n}}

    Parameters
    ----------
    Ds:         float
                Self distance (unitless), normally provided from
                datasheet/reference
    *args:      floats, optional
                Remaining set of distance values (unitless)
    """
    # Find the Root from Number of Arguments
    root = len(args) + 1
    # Calculate the Root Term
    gmdx = Ds
    for dist in args:
        gmdx *= dist
    # Apply Root Calculation
    GMD = gmdx ** (1 / root)
    return (GMD)


# Define FOC IM Rated Value Calculator
def indmachfocratings(Rr, Rs, Lm, Llr=0, Lls=0, Lr=None,
                      Ls=None, Vdqs=1, Tem=1, wes=1):
    r"""
    FOC Ind. Machine Rated Operation Calculator.

    Determines the parameters and characteristics of a Field-
    Oriented-Controlled Induction Machine operating at its
    rated limits.

    Parameters
    ----------
    Rr:         float
                Rotor resistance in per-unit-ohms
    Rs:         float
                Stator resistance in per-unit-ohms
    Lm:         float
                Magnetizing inductance in per-unit-Henrys
    Llr:        float, optional
                Rotor leakage inductance in per-unit-Henrys,
                default=0
    Lls:        float, optional
                Stator leakage inductance in per-unit-Henrys,
                default=0
    Lr:         float, optional
                Rotor inductance in per-unit-Henrys
    Ls:         float, optional
                Stator inductance in per-unit-Henrys
    Vdqs:       complex, optional
                The combined DQ-axis voltage required for rated
                operation, in per-unit-volts, default=1+j0
    Tem:        float, optional
                The mechanical torque required for rated operation,
                in per-unit-newton-meters, default=1
    wes:        float, optional
                The per-unit electrical system frequency, default=1

    Returns
    -------
    Idqr:       complex
                Combined DQ-axis Rotor Current in per-unit-amps
    Idqs:       complex
                Combined DQ-axis Stator Current in per-unit-amps
    LAMdqr:     complex
                Combined DQ-axis Rotor Flux in per-unit
    LAMdqs:     complex
                Combined DQ-axis Stator Flux in per-unit
    slip_rat:   float
                Rated Slip as percent of rotational and system frequencies
    w_rat:      float
                Rated System frequency in per-unit-rad/sec
    lamdr_rat:  float
                Rated D-axis rotor flux in per-unit
    """
    # Condition Inputs:
    if Ls == None:  # Use Lls instead of Ls
        Ls = Lls + Lm
    if Lr == None:  # Use Llr instead of Lr
        Lr = Llr + Lm

    # Define Equations Function as Solver
    def equations(val):
        Idr, Iqr, Ids, Iqs, LAMdr, LAMqr, LAMds, LAMqs, wr = val
        A = (Rs * Ids - wes * LAMqs) - Vdqs
        B = Rs * Iqs - wes * LAMds
        C = Rr * Idr - (wes - wr) * LAMqr
        D = Rr * Iqr + (wes - wr) * LAMdr
        E = (Ls * Ids + Lm * Idr) - LAMds
        F = (Ls * Iqs + Lm * Iqr) - LAMqs
        G = (Lm * Ids + Lr * Idr) - LAMdr
        H = (Lm * Iqs + Lr * Iqr) - LAMqr
        I = (Lm / Lr * (LAMdr * Iqs - LAMqr * Ids)) - Tem
        return (A, B, C, D, E, F, G, H, I)

    # Define Initial Guesses
    Idr0 = -1
    Iqr0 = -1
    Ids0 = 1
    Iqs0 = 1
    LAMdr0 = Lm * Ids0 + Lr * Idr0
    LAMqr0 = Lm * Iqs0 + Lr * Iqr0
    LAMds0 = Ls * Ids0 + Lm * Idr0
    LAMqs0 = Ls * Iqs0 + Lm * Iqr0
    wr = 1
    # Use Iterative Solver to Find Results
    Idr, Iqr, Ids, Iqs, LAMdr, LAMqr, LAMds, LAMqs, wr = _fsolve(equations, (
        Idr0, Iqr0, Ids0, Iqs0, LAMdr0, LAMqr0, LAMds0, LAMqs0, wr))
    # Calculate Remaining Rating Terms
    slip_rated = (wes - wr) / wes
    w_rated = wr
    lamdr_rated = abs(LAMdr + 1j * LAMqr)
    return (
        compose(Idr, Iqr),
        compose(Ids, Iqs),
        compose(LAMdr, LAMqr),
        compose(LAMds, LAMqs),
        slip_rated,
        w_rated,
        lamdr_rated
    )


# Define FOC IM Control Equation Evaluation Function
def imfoc_control(Tem_cmd, LAMdr_cmd, wr_cmd, Rr, Rs, Lm,
                  Llr=0, Lls=0, Lr=None, Ls=None, s_err=0):
    """
    FOC Ind. Machine Rated Operation Calculator.

    Determines the parameters and characteristics of a Field-
    Oriented-Controlled Induction Machine operating at its
    rated limits.

    Parameters
    ----------
    Tem_cmd:    float
                Mechanical torque setpoint in per-unit-newton-meters
    LAMdr_cmd:  float
                D-axis flux setpoint in per-unit
    wr_cmd:     float
                Mechanical (rotor) speed in per-unit-rad/sec
    Rr:         float
                Rotor resistance in per-unit-ohms
    Rs:         float
                Stator resistance in per-unit-ohms
    Lm:         float
                Magnetizing inductance in per-unit-Henrys
    Llr:        float, optional
                Rotor leakage inductance in per-unit-Henrys,
                default=0
    Lls:        float, optional
                Stator leakage inductance in per-unit-Henrys,
                default=0
    Lr:         float, optional
                Rotor inductance in per-unit-Henrys
    Ls:         float, optional
                Stator inductance in per-unit-Henrys
    s_err:      float, optional
                Error in slip calculation as a percent (e.g. 0.25),
                default=0

    Returns
    -------
    Vdqs:       complex
                Combined DQ-axis Stator Voltage in per-unit volts
    Idqr:       complex
                Combined DQ-axis Rotor Current in per-unit-amps
    Idqs:       complex
                Combined DQ-axis Stator Current in per-unit-amps
    LAMdqr:     complex
                Combined DQ-axis Rotor Flux in per-unit
    LAMdqs:     complex
                Combined DQ-axis Stator Flux in per-unit
    wslip:      float
                Machine Slip frequency in per-unit-rad/sec
    wes:        float
                The electrical system frequency in per-unit-rad/sec
    """
    # Condition Inputs:
    if Ls == None:  # Use Lls instead of Ls
        Ls = Lls + Lm
    if Lr == None:  # Use Llr instead of Lr
        Lr = Llr + Lm
    # Calculate Additional Constraints
    sigma = (1 - Lm ** 2 / (Ls * Lr))
    accuracy = 1 + s_err
    # Command Values (Transient and Steady State)
    Ids = LAMdr_cmd / Lm
    Iqs = Tem_cmd / ((Lm / Lr) * LAMdr_cmd)
    wslip = Rr / (Lr * accuracy) * (Lm * Iqs) / LAMdr_cmd
    wes = wslip + wr_cmd
    # Stator dq Voltages (Steady State)
    Vds = Rs * Ids - wes * sigma * Ls * Iqs
    Vqs = Rs * Iqs - wes * Ls * Ids
    # Remaining Steady State
    Iqr = -Lm / Lr * Iqs
    Idr = 0
    LAMqr = 0
    LAMqs = sigma * Ls * Iqs
    LAMds = Ls * Ids
    return (
        compose(Vds, Vqs),
        compose(Idr, Iqr),
        compose(Ids, Iqs),
        compose(LAMdr_cmd, LAMqr),
        compose(LAMds, LAMqs),
        wslip,
        wes
    )


# Define Synch. Machine Eq Calculator
def synmach_Eq(Vt_pu, Itmag, PF, Ra, Xd, Xq):
    # noqa: D401   "Synchronous" is an intentional descriptor
    r"""
    Synchronous Machine Eq Calculator.

    Given specified parameter set, will calculate
    the internal voltage on the q-axis (Eq).

    .. math:: E_q=V_{t_{pu}}-\left[R_a\cdot I_{t_{pu}}+
       j\cdot X_q\cdot I_{t_{pu}}+j(X_d-X_q)\cdot I_{ad}\right]

    where:

    .. math:: I_{t_{pu}}=I_{t_{mag}}\cdot e^{-j(
       \angle{V_{t_{pu}}}-\cos^{-1}(PF))}

    .. math:: \theta_q=\angle{V_{t_{pu}}-\left(R_a
       I_{t_{pu}}+j\cdot X_qI_{t_{pu}}\right)

    .. math:: I_{ad}=\left|I_{t_{pu}}\cdot\sin(
       -\cos^{-1}(PF)+\theta_q)\right|e^{j(\theta_q
       -90°)}

    Parameters
    ----------
    Vt_pu:      complex
                Terminal voltage in per-unit-volts
    Itmag:      float
                Terminal current magnitude in per-
                unit-amps
    PF:         float
                Machine Power Factor, (+)ive values denote
                leading power factor, (-)ive values denote
                lagging power factor
    Ra:         float
                AC resistance in per-unit-ohms
    Xd:         float
                D-axis reactance in per-unit-ohms
    Xq:         float
                Q-axis reactance in per-unit-ohms

    Returns
    -------
    Eq:         complex
                Internal Synchronous Machine Voltage
                in per-unit-volts
    """
    # Calculate Required Terms
    phi = _np.arccos(PF)
    Itmag = abs(Itmag)
    It_pu = Itmag * _np.exp(-1j * (_np.angle(Vt_pu) + phi))
    th_q = _np.angle(Vt_pu - (Ra * It_pu + 1j * Xq * It_pu))
    Iad = (abs(It_pu) * _np.sin(phi + th_q)) * _np.exp(1j * (th_q - _np.pi / 2))
    # Calculate Eq
    Eq = Vt_pu - (Ra * It_pu + 1j * Xq * It_pu + 1j * (Xd - Xq) * Iad)
    return (Eq)


# Define Power-Factor Voltage/Current Relation
def vipf(V=None, I=None, PF=1, find=''):
    """
    Voltage / Current / Power Factor Solver.

    Given two of the three parameters, will solve for the
    third; beit voltage, current, or power factor.

    Parameters
    ----------
    V:          complex
                System voltage (in volts), default=None
    I:          complex
                System current (in amps), default=None
    PF:         float
                System power factor, (+)ive values denote
                leading power factor, (-)ive values denote
                lagging power factor; default=1
    find:       str, optional
                Control argument to specify which value
                should be returned.

    Returns
    -------
    V:          complex
                System voltage (in volts), default=None
    I:          complex
                System current (in amps), default=None
    PF:         float
                System power factor, (+)ive values denote
                leading power factor, (-)ive values denote
                lagging poer factor; default=1
    """
    # Test to find Voltage
    if isinstance(V, float) and isinstance(I, complex):
        phi = -_np.sign(PF) * _np.arccos(PF)
        V = V * _np.exp(-1j * phi)
    # Test to find Current
    elif isinstance(V, complex) and isinstance(I, float):
        phi = _np.sign(PF) * _np.arccos(PF)
        I = I * _np.exp(-1j * phi)
    # Test to find Power Factor
    elif all([V, I]):
        phi = _np.angle(V) - _np.angle(I)
        PF = _np.cos(phi)
    # Failed Mode
    else:
        raise ValueError("All values must be provided.")
    # Return
    find = find.upper()
    if find == 'V':
        return (V)
    elif find == 'I':
        return (I)
    elif find == 'PF':
        return (PF)
    else:
        return (V, I, PF)


# Define Synchronous Speed Calculator
def syncspeed(Npol, freq=60, Hz=False):
    # noqa: D401   "Synchronous" is an intentional descriptor
    r"""
    Synchronous Speed Calculator Function.

    Simple method of calculating the synchronous speed of an induction machine
    given the number of poles in the machine's construction, and
    the machine's operating electrical frequency.

    .. math:: \omega_{\text{syn}}=\frac{2\pi
       \cdot\text{freq}}{\frac{N_{\text{pol}}}{2}}

    Parameters
    ----------
    Npol:       int
                Number of electrical poles in machine's construction.
    freq:       float, optional
                Frequency of electrical system in Hertz, default=60
    Hz:         bool, optional
                Boolean control to enable return in Hertz. default=False

    Returns
    -------
    wsyn:       float
                Synchronous Speed of Induction Machine, defaults to units of
                rad/sec, but may be set to Hertz if `Hz` set to True.
    """
    wsyn = 2 * _np.pi * freq / (Npol / 2)
    if Hz:
        return (wsyn / (2 * _np.pi))
    return (wsyn)


# Define Machine Slip Calculation Function
def machslip(mech, syn=60):
    r"""
    Machine Slip Calculator.

    Given the two parameters (mechanical and synchronous speed, or frequency)
    this function will return the unitless slip of the rotating machine.

    .. math:: \text{slip}=\frac{\text{syn}-\text{mech}}
       {\text{syn}}

    Parameters
    ----------
    mech:       float
                The mechanical frequency (or speed), of the rotating machine.
    syn:        float, optional
                The synchronous frequency (or speed), defaults as a frequency
                set to 60Hz, default=60

    Returns
    -------
    slip:       float
                The rotating machine's slip constant.
    """
    slip = (syn - mech) / syn
    return (slip)


# Define 3-Phase Valpha Calculator
def phs3valpha(VA, VB=0, VC=0):
    r"""
    Three-Phase V-Alpha Calculator.

    Accepts the three-phase voltages for which the accumulated Alpha voltage
    should be calculated.

    .. math:: V_{\alpha}=V_A-\frac{V_B}{2}-\frac{V_C}{2}

    Parameters
    ----------
    VA:         [float, complex]
                A-phase voltage, (or tuple/list of voltages), unitless.
    VB:         [float, complex], optional
                B-phase voltage, unitless.
    VC:         [float, complex], optional
                C-phase voltage, unitless.

    Returns
    -------
    Valpha:     [float, complex]
                Alpha-voltage as calculated from input three-phase voltages.
                Matches type of inputs.
    """
    # Handle Combined (list/tuple) Input
    if (isinstance(VA, (tuple, list)) and VB == 0 and VC == 0):
        if len(VA) != 3:
            raise ValueError("Invalid input set, must "
                             "be list of three elements, three inputs,"
                             " or three array-like objects of equal "
                             "length.")
        Valpha = VA[0] - VA[1] / 2 - VA[2] / 2
    # Handle Separated Inputs
    else:
        Valpha = VA - VB / 2 - VC / 2
    # Return the Alpha-Voltage
    return (Valpha)

def wireresistance(length=None,diameter=None,rho=16.8*10**-9,R=None):
    r"""
    Wire Resistance Calculator.

    Enter three values to calculate the remaing one. Even though every variable
    is unitless, please use the International System of Units.

    .. math:: R = \frac{\rho*l}{A}

    Parameters
    ----------
    length:     [float], optional
                Wire length, unitless
    diameter:   [float], optional
                Wire diameter, unitless.
    rho:        [float], optional
                Material resistivity, unitless
                Default value is copper resistivity: 16.8*10-9 
    R:          [float], optional
                Wire resistance, unitless.

    Returns
    -------
    length:     [float], optional
                Wire length, unitless
    diameter:   [float], optional
                Wire diameter, unitless.
    rho:        [float], optional
                Material resistivity, unitless
                Default value is copper resistivity: 16.8*10-9 
    R:          [float], optional
                Wire resistance, unitless.
    """
    if R == length == diameter == None:
        raise ValueError("To few arguments.")
    # Given length and diameter
    if length != None and diameter != None:
        # calculating the area
        A = pi*( diameter ** 2 ) / 4
        return rho*length/A
    # Given resistance and diameter
    elif R != None and diameter != None:
        # calculating the area
        A = pi*( diameter ** 2 ) / 4
        return R*A/rho
    # Given resistance and length
    elif R != None and length != None:
        A = rho*length/R
        return _np.sqrt(4*A/pi)

def parallel_plate_capacitance(A=None, d=None, e=e0, C=None):
    r"""
    Parallel-Plate Capacitance Calculator.

    Enter three values to calculate the remaing one. Even though every variable
    is unitless, please use the International System of Units.

    .. math:: C = \frac{\varepsilon \cdot A}{d}

    Parameters
    ----------
    A:  float, optional
        Area of the plate, unitless.
    d:  float, optional
        Distance between the plates, unitless.
    e:  float, optional
        Permitivity of the dielectric, unitless.
        Default value is the permittivity of free space: 8.854E-12
    C:  float, optional
        Capacitance, unitless.

    Returns
    -------
    A:  float, optional
        Area of the plate, unitless.
    d:  float, optional
        Distance between the plates, unitless.
    e:  float, optional
        Permitivity of the dielectric, unitless.
        Default value is the permittivity of free space: 8.854E-12
    C:  float, optional
        Capacitance, unitless.
    """
    if C == A == d == None:
        raise ValueError("To few arguments.")
    # Given area and distance
    if A != None and d != None:
        return e*A/d
    # Given capacitance and distance
    elif C != None and d != None:
        return d*C/e
    # Given capacitance and area
    elif C != None and A != None:
        return e*A/C

def solenoid_inductance(A=None, l=None, N=None, u=u0, L=None):
    r"""
    Solenoid Inductance Calculator.

    Enter four values to calculate the remaing one. Even though every variable
    is unitless, please use the International System of Units.

    .. math:: L = \frac{\mu \cdot N^2 \cdot A}{l}

    Parameters
    ----------
    A:  float, optional
        Cross sectional area, unitless.
    l:  float, optional
        Length, unitless.
    N:  float, optional
        Number of turns, unitless.
    u:  float, optional
        Core permeability, unitless.
        Default value is the permeability of free space: 4πE-7
    L:  float, optional
        Inductance, unitless.

    Returns
    -------
    A:  float, optional
        Cross sectional area, unitless.
    l:  float, optional
        Length, unitless.
    N:  float, optional
        Number of turns, unitless.
    u:  float, optional
        Core permeability, unitless.
        Default value is the permeability of free space: 4πE-7
    L:  float, optional
        Inductance, unitless.
    """
    if L == A == l == N == None:
        raise ValueError("To few arguments.")
    # Given area, length and number of turns
    if A != None and l != None and N != None:
        return N**2*u*A/l
    # Given inductance, length and number of turns
    elif L != None and l != None and N != None:
        return L*l/(N**2*u)
    # Given inductance, area and number of turns
    elif L != None and A != None and N != None:
        return N**2*u*A/L
    # Given inductance, area and length
    elif L != None and A != None and l != None:
        return _np.sqrt(L*l/(u*A))
        
def ic_555_astable(R=None,C=None,freq=None,t_high=None,t_low=None):
    """
    555 Integrated Circuit Calculator.
    
    Evaluate a number of common attributes related to the common 555 integrated
    circuit including time period, frequency, duty cycle, time spent low during
    each cycle, time spent high during each cycle.

    TODO: This function should be broken into multiple smaller functions.
    
    Parameters
    ---------- 
    R:      list[float, float] or tuple(float, float), optional
            List of 2 resistor which are need in configuring IC 555.
    C:      float, optional
            Capacitance between Threshold Pin and ground
    f:      float, optional
            Electrical system frequency in Hertz. 
    t_high: float, optional
            ON time of IC 555 
    t_low:  float, optional
            OFF time of IC 555 
    
    Returns 
    ------- 
    dict:   "time_period": Time period of oscillating IC 555 
            "frequency": frequency of oscilation of IC 555 
            "duty_cycle": ration between ON time and total time
            "t_low": ON time of IC 555 
            "t_high": OFF time of IC 555 
    """
    if R!=None and C!=None:
        if len(R) != 2:
            raise ValueError(
                "Monostable 555 IC will have only 2 resitances to be fixed "
                f"but {len(R)} were given"
            )

        [R1, R2] = R

        T = _np.log(2)*C*(R1+2*R2)
        freq = 1/T
        t_low = _np.log(2)*C*R2
        t_high = _np.log(2)*C*(R1+R2)
        duty_cycle = t_high*100/T

        return {
            'time_period':T,
            'frequency':freq,
            'duty_cycle':duty_cycle,
            't_low':t_low,
            't_high':t_high
        }
    
    elif t_high!=None and t_low!=None and C!=None:

        x2 = t_low/C*_np.log(2)
        x1 = t_high/C*_np.log(2)
        T = t_high+t_low
        freq = 1/(T)
        duty_cycle = t_high/(T)

        return {
            'time_period':T,
            'frequency':freq,
            'duty_cycle':duty_cycle,
            'R1':x1-x2,
            'R2':x2
        }
    else:
        raise TypeError("Not enough parqmeters are passed")
            
def ic_555_monostable(R=None,C=None,freq=None,t_high=None,t_low=None):
    """
    555 Integrated Circuit Calculator.
    
    Evaluate a number of common attributes related to the common 555 integrated
    circuit including time period, frequency, duty cycle, time spent low during
    each cycle, time spent high during each cycle.

    TODO: This function should be broken into multiple smaller functions.
    
    Parameters
    ---------- 
    R:      list[float, float] or tuple(float, float), optional
            List of 2 resistor which are need in configuring IC 555.
    C:      float, optional
            Capacitance between Threshold Pin and ground
    f:      float, optional
            Electrical system frequency in Hertz. 
    t_high: float, optional
            ON time of IC 555 
    t_low:  float, optional
            OFF time of IC 555 
    
    Returns 
    ------- 
    dict:   "time_period": Time period of oscillating IC 555 
            "frequency": frequency of oscilation of IC 555 
            "duty_cycle": ration between ON time and total time
            "t_low": ON time of IC 555 
            "t_high": OFF time of IC 555 
    """
    T = t_high+t_low
    if R is None:
        try:
            assert C!=None and T!=None
        except AssertionError:
            raise ValueError(
                "To find Resitance, Capacitance and delay time should be "
                "provided"
            )
        return T/(_np.log(3)*C)
    elif C is None:
        try:
            assert R!=None and T!=None
        except AssertionError:
            raise ValueError(
                "To find Capacitance , Resistance and delay time should be "
                "provided"
            )
        return T/(_np.log(3)*R)

    elif T is None:

        try:
            assert R!=None and T!=None
        except AssertionError:
            raise ValueError(
                "To find Time delay , Resistance and Capacitance should be "
                "provided"
            )
        return R*C*_np.log(3)


def t_attenuator(Adb, Z0):
    r"""
    T attenuator.

    The T attenuator is a type of attenuator that looks like the letter T. 
    The T attenuator consists of three resistors. Two of these are connected in 
    series and the other one is connected from between the two other resistors
    to ground. The resistors in series often have the same resistance.

    .. math:: R1 = Z0*(\frac{10^{\frac{A_{db}}{20}}-1}{10^{\frac{A_{db}}{20}}+1});
    .. math:: R2 = Z0*(\frac{10^{\frac{A_{db}}{20}}}{10^{\frac{A_{db}}{10}}-1})

    .. image:: /static/t-attenuator-circuit.png

    Parameters
    ---------- 
    Adb: float Attenuation in db
    Z0: float Impedence

    Returns 
    ------- 
    R1: float T attenuator R1
    R2: float T attenuator R2
    """
    x = Adb/20

    R1 = Z0*(_np.power(10, x)-1)/(_np.power(10, x)+1)
    R2 = 2*Z0*_np.power(10, x)/(_np.power(10, 2*x)-1)

    return R1,R2

def pi_attenuator(Adb, Z0):
    r"""
    Pi attenuator.

    The Pi attenuator is a type of attenuator that looks like the Greek letter π.
    The Pi attenuator consists of three resistors. One of these is connected in series and 
    the other two are connected in parallel to ground. The parallel resistors often have the same resistance.

    .. math:: R1 = Z0*(\frac{10^{\frac{A_{db}}{20}}+1}{10^{\frac{A_{db}}{20}}-1})
    .. math:: R2 = \frac{Z0}{2}*(10^{\frac{A_{db}}{20}} - \frac{1}{10^{\frac{A_{db}}{20}}})
    .. image:: /static/pi-attenuator-circuit.png

    Parameters
    ---------- 
    Adb: float Attenuation in db
    Z0: float Impedence

    Returns 
    ------- 
    R1: float π attenuator R1
    R2: float π attenuator R2
    """
    x = Adb/20

    R1 = Z0*(_np.power(10, x)+1)/(_np.power(10, x)-1)
    R2 = (Z0/2)*(_np.power(10, x) - (1/(_np.power(10, x))))

    return R1,R2
# END OF FILE
