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


import cmath as _c
from inspect import getframeinfo as _getframeinfo
from inspect import stack as _stack
from warnings import showwarning as _showwarning

import matplotlib.pyplot as _plt
import numpy as _np
from scipy.integrate import quad as integrate

from .version import NAME, VERSION
from .constants import *
from .phasors import phasor, parallelz

__version__ = VERSION

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

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.tcycle(1, freq=60) #Value of ncycles=1 & freq=60
    0.016666666666666666
    >>> ep.tcycle(1, freq=50) #Value of ncycles=1 & freq=50
    0.02
    """
    # Condition Inputs
    if isinstance(ncycles, _np.ndarray) and isinstance(freq, _np.ndarray):
        if ncycles.shape != freq.shape:
            raise ValueError("ncycles and freq must be the same shape")

    elif isinstance(ncycles, list) and isinstance(freq, list):
        if len(ncycles) != len(freq):
            raise ValueError("ncycles and freq must be the same length")

    ncycles = _np.asarray(ncycles)
    freq = _np.asarray(freq)
    if 0 in freq:
        raise ZeroDivisionError("Frequency must not be 0")
    if not (freq > 0).all():
        # frequency must be postive value
        raise ValueError("Frequency must be postive value")
    # Evaluate the time for ncycles
    time = ncycles / freq
    # Return
    if isinstance(time, _np.ndarray) and len(time) == 1:
        return time[0]
    else:
        return time

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
    float:          Capacitance or Inductance of Impedance

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.reactance(z=5) # ohms - inductive impedance
    0.0132629...
    """
    # Evaluate Omega
    w = 2 * _np.pi * freq
    # Input is Complex
    if isinstance(z, complex):
        # Test for Resistance
        if abs(z.real) > sensetivity:
            R = z.real
        else:
            R = 0
        if z.imag > 0:
            out = z / (w * 1j)
        else:
            out = 1 / (w * 1j * z)
        out = abs(out)
        # Combine with resistance if present
        if R != 0:
            out = (R, out)
    else:
        if z > 0:
            out = z / w
        else:
            out = 1 / (w * z)
        out = abs(out)
    # Return Output
    return out


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
    >>> import numpy as np
    >>> import electricpy as ep
    >>> from electricpy import phasors
    >>> v = phasor(67, 120)
    >>> ep.cprint(v)
    67.0 ∠ 120.0°
    >>> voltages = np.array([[67,0],
    ...                      [67,-120],
    ...                      [67,120]])
    >>> Vset = ep.phasors.phasorlist( voltages )
    >>> ep.cprint(Vset)
    67.0 ∠ 0.0°
    67.0 ∠ -120.0°
    67.0 ∠ 120.0°


    See Also
    --------
    electricpy.phasors.phasor:       Phasor Generating Function
    electricpy.phasors.phasorlist:   Phasor Generating Function for Lists/Arrays
    electricpy.phasors.phasorz:      Impedance Phasor Generator
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
        except (ValueError, IndexError):
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
        elif label is None:
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
        elif unit is None:
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
            if _label is not None:
                strg += _label + " "
            strg += str(mag) + " ∠ " + str(ang) + "°"
            if _unit is not None:
                strg += " " + _unit
            printarr = _np.append(printarr, strg)
            numarr = _np.append(numarr, [mag, ang])
        # Reshape Arrays
        printarr = _np.reshape(printarr, (row, col))
        numarr = _np.reshape(numarr, (sz, 2))
        # Print
        if printval and row == 1:
            if title is not None:
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
            if title is not None:
                print(title)
            print(strg)
        elif printval:
            if title is not None:
                print(title)
            print(printarr)
        # Return if Necessary
        if ret:
            return (numarr)
    elif isinstance(val, (int, float, complex)):
        # Handle Invalid Unit/Label
        if unit is not None and not isinstance(unit, str):
            raise ValueError("Invalid Unit Type for Value")
        if label is not None and not isinstance(label, str):
            raise ValueError("Invalid Label Type for Value")
        mag, ang_r = _c.polar(val)  # Convert to polar form
        ang = _np.degrees(ang_r)  # Convert to degrees
        mag = _np.around(mag, decimals)  # Round
        ang = _np.around(ang, decimals)  # Round
        strg = ""
        if label is not None:
            strg += label + " "
        strg += str(mag) + " ∠ " + str(ang) + "°"
        if unit is not None:
            strg += " " + unit
        # Print values (by default)
        if printval:
            if title is not None:
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

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.cprint(ep.phaseline(VLL=(13.8*ep.k))) # 13.8kV
    7967.434 ∠ -30.0°
    """
    # Monitor for deprecated input
    if 'complex' in kwargs.keys():
        if realonly is None:
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
    if isinstance(output, complex) and realonly is None:
        realonly = False
    # Return as complex only when requested
    if realonly:
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

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.powerset(P=400, Q=300)
    (400, 300, 500.0, 0.8)
    >>> ep.powerset(P=400, Q=300, find="PF")
    0.8
    """
    # Given P and Q
    if (P is not None) and (Q is not None):
        S = _np.sqrt(P ** 2 + Q ** 2)
        PF = P / S
        if Q < 0:
            PF = -PF
    # Given S and PF
    elif (S is not None) and (PF is not None):
        P = abs(S * PF)
        Q = _np.sqrt(S ** 2 - P ** 2)
        if PF < 0:
            Q = -Q
    # Given P and PF
    elif (P is not None) and (PF is not None):
        S = P / PF
        Q = _np.sqrt(S ** 2 - P ** 2)
        if PF < 0:
            Q = -Q
    # Given P and S
    elif (P is not None) and (S is not None):
        Q = _np.sqrt(S ** 2 - P ** 2)
        PF = P / S
    # Given Q and S
    elif (Q is not None) and (S is not None):
        P = _np.sqrt(S ** 2 - Q ** 2)
        PF = P / S
    else:
        raise ValueError("ERROR: Invalid Parameters or too few" +
                         " parameters given to calculate.")
    # Return
    find = find.upper()
    if find == 'P':
        return P
    elif find == 'Q':
        return Q
    elif find == 'S':
        return S
    elif find == 'PF':
        return PF
    else:
        return P, Q, S, PF


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
    if V is not None and freq is not None:
        SR = 2 * _np.pi * V * freq
    elif freq is not None and SR is not None:
        V = SR / (2 * _np.pi * freq)
    elif V is not None and SR is not None:
        freq = SR / (2 * _np.pi * V)
    else:
        raise ValueError("ERROR: Invalid Parameters or too few" +
                         " parameters given to calculate.")
    if find == 'V':
        return V
    elif find == 'freq':
        return freq
    elif find == 'SR':
        return SR
    else:
        return V, freq, SR


# Define Non-Linear Power Factor Calculator
def non_linear_pf(PFtrue=False, PFdist=False, PFdisp=False):
    """
    Non-Linear Power Factor Evaluator.

    This function is designed to evaluate one of three unknowns
    given the other two. These particular unknowns are the arguments
    and as such, they are described in the representative sections
    below.

    .. note:: Also available as `nlinpf`.

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
    if PFtrue is not None and PFdist is not None and PFdisp is not None:
        raise ValueError("ERROR: Too many constraints, no solution.")
    if PFdist is not None and PFdisp is not None:
        return PFdist * PFdisp
    if PFtrue is not None and PFdisp is not None:
        return PFtrue / PFdisp
    if PFtrue is not None and PFdist is not None:
        return PFtrue / PFdist
    raise ValueError("ERROR: Function requires at least two arguments.")


# Alias to original Name
nlinpf = non_linear_pf


# Define Short-Circuit RL Current Calculator
def short_circuit_current(V, Z, t=None, f=None, mxcurrent=True, alpha=None):
    """
    Short-Circuit-Current (ISC) Calculator.

    The Isc-RL function (Short Circuit Current for RL Circuit)
    is designed to calculate the short-circuit current for an
    RL circuit.

    .. note:: Also available as `iscrl`.

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
    if f is not None:
        omega = 2 * _np.pi * f
    else:
        omega = None
    R = abs(Z.real)
    X = abs(Z.imag)
    theta = _np.arctan(X / R)

    # If Maximum Current is Desired and No alpha provided
    if mxcurrent and alpha is None:
        alpha = theta - _np.pi / 2
    elif mxcurrent and alpha is not None:
        raise ValueError("ERROR: Inappropriate Arguments Provided.\n" +
                         "Not both mxcurrent and alpha can be provided.")

    # Calculate Asymmetrical (total) Current if t is not None
    if t is not None and f is not None:
        # Calculate RMS if none of the angular values are provided
        if alpha is None and omega is None:
            # Calculate tau
            tau = t / (1 / 60)
            K = _np.sqrt(1 + 2 * _np.exp(-4 * _np.pi * tau / (X / R)))
            IAC = abs(V / Z)
            Irms = K * IAC
            # Return Values
            return Irms, IAC, K
        elif alpha is None or omega is None:
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
            iDC = -_np.sqrt(2) * V / Z * \
                _np.sin(alpha - theta) * _np.exp(-t / T)
            i = iAC + iDC
            # Return Values
            return i, iAC, iDC, T
    elif (t is not None and f is None) or (t is None and f is not None):
        raise ValueError("ERROR: Inappropriate Arguments Provided.\n" +
                         "Must provide both t and f or neither.")
    else:
        Iac = abs(V / Z)
        return Iac


# Alias to original Name
iscrl = short_circuit_current


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

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.voltdiv(Vin=12, R1=4, R2=8)
    8.0
    >>> ep.voltdiv(Vin=12, R1=6, R2=12, Rload=12) # R2 and Rload are parallel
    6.0
    """
    # Determine whether Rload is given
    if Rload is None:  # No Load Given
        Vout = Vin * R2 / (R1 + R2)
    else:  # Load was given
        Rp = R2 * Rload / (R2 + Rload)
        Vout = Vin * Rp / (R1 + Rp)
    return Vout


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

    Examples
    --------
    >>> from electricpy.constants import k
    >>> import electricpy as ep
    >>> ep.curdiv(Ri=1*k, Rset=(1*k, 1*k), Iin=12) # 12-amps, split three ways
    4.0
    >>> ep.curdiv(Ri=1*k, Rset=(1*k, 1*k), Iin=12, Vout=True) # Find Voltage
    (4.0, 4000.0)
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
    if Vin is not None and Iin is None:  # Vin Provided
        Iin = Vin / Rtot  # Calculate total current
        Ii = Iin * Rtot / Ri  # Calculate the current of interest
    elif Vin is None and Iin is not None:  # Iin provided
        Ii = Iin * Rtot / Ri  # Calculate the current of interest
    else:
        raise ValueError("ERROR: Too many or too few constraints provided.")
    if Vout:  # Asked for voltage across resistor of interest
        Vi = Ii * Ri
        return Ii, Vi
    else:
        return Ii


# Induction Machine Slip
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
    Ns = (120 * freq) / poles
    return (Ns - Nr) / (Ns)


# Define Function to Evaluate Resistance Needed for LED
def led_resistor(Vsrc, Vfwd=2, Ifwd=20):
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
    float:  Instantaneous Power at time t.
    """
    # Evaluate omega
    w = 2 * _np.pi * freq
    # Calculate
    Pinst = P + P * _np.cos(2 * w * t) - Q * _np.sin(2 * w * t)
    return Pinst


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
    if delta is None and wye is None:
        raise ValueError(
            "ERROR: Either delta or wye impedances must be specified."
        )
    # Determine which set of impedances was provided
    if delta is not None and wye is None:
        Z12, Z23, Z31 = delta  # Gather particular impedances
        Zsum = Z12 + Z23 + Z31  # Find Sum
        # Calculate Wye Impedances
        Z1 = Z12 * Z31 / Zsum
        Z2 = Z12 * Z23 / Zsum
        Z3 = Z23 * Z31 / Zsum
        Zset = (Z1, Z2, Z3)
        if round is not None:
            Zset = _np.around(Zset, round)
        return Zset  # Return Wye Impedances
    if delta is None and wye is not None:
        Z1, Z2, Z3 = wye  # Gather particular impedances
        Zmultsum = Z1 * Z2 + Z2 * Z3 + Z3 * Z1
        Z23 = Zmultsum / Z1
        Z31 = Zmultsum / Z2
        Z12 = Zmultsum / Z3
        Zset = (Z12, Z23, Z31)
        if round is not None:
            Zset = _np.around(Zset, round)
        return Zset  # Return Delta Impedances


# calculating impedance of bridge network
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
            Bridge impedance 1iscrl
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
    za, zb, zc = dynetz(delta=(z1, z5, z4))
    ze1 = zb + z2
    ze2 = zc + z3
    return za + (ze1 * ze2) / (ze1 + ze2)


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
    pflow = (Vs * Vr) / Xline * _np.sin(ds - dr)
    return pflow


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
    if Vbase is None:
        Vbase = V
    if Sbase is None:
        Sbase = S
    # Prevent scaling if per-unit already applied
    if Vbase:
        Vbase = 1
    if Sbase:
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
            Zsource_pu.append(phasors(Zsource_pu, angle))
    else:
        Zsource_pu = phasors(Zsource_pu, nu)
    if not perunit:
        Zsource = Zsource_pu * Vbase ** 2 / Sbase
        return Zsource
    return Zsource_pu


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
    return R, X


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
    if isinstance(S, complex) or PF is not None:
        if PF is not None:
            # Evaluate Elements
            P, Q, S, PF = powerset(S=S, PF=PF)
        else:
            P = S.real
            Q = S.imag
        # Compute Elements
        if parallel:
            Zp = V ** 2 / (3 * (P + 1j * Q))
        else:
            Zp = V ** 2 / (P + 1j * Q)
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
    return VDC


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
    return kp, ki, w0L


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
    return _np.sqrt(2) * val


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
    return val * _np.sqrt(0.5)


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
    if quad:
        def intf(w):
            return w ** 2 * func(w)

        num = integrate(intf, 0, _np.inf)[0]
        den = integrate(func, 0, _np.inf)[0]
        # Calculate W
        W = _np.sqrt(num / den)
    else:
        # Calculate W
        W = _np.sqrt(Sw2 / Stot)
    Wr = _np.around(W, round)
    # Plot Upon Request
    if plot:
        _plt.plot(omega, Sxx)
        _plt.title(title)
        # Evaluate Text Location
        x = 0.65 * max(omega)
        y = 0.80 * max(Sxx)
        _plt.text(x, y, "Wrms: " + str(Wr))
        _plt.show()
    # Return Calculated RMS Bandwidth
    return W


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
    return C


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
    return C


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
    if VLL is None and VLN is None:
        raise ValueError("ERROR: One voltage must be provided.")
    if VLL is not None:
        return VLL ** 2 / S
    else:
        return (_np.sqrt(3) * VLN) ** 2 / S


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
    if VLL is None and VLN is None:
        raise ValueError("ERROR: One voltage must be provided.")
    if VLL is not None:
        return S / (_np.sqrt(3) * VLL)
    elif VLN is not None:
        return S / (3 * VLN)
    else:
        return S / V1phs


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
    return pu_new


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
    return z


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
    if S3phs is None:
        return z_pu
    else:
        z = zrecompose(z_pu, S3phs, VLL, VLN)
        return z


# Define Generator Internal Voltage Calculator
def geninternalv(I, Zs, Vt, Vgn=None, Zm=None, Zmp=None, Zmpp=None, Ip=None, Ipp=None):
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
    if Zmp == Zmpp == Ip == Ipp is not None:
        if Vgn is None:
            Vgn = 0
        Ea = Zs * I + Zmp * Ip + Zmpp * Ipp + Vt + Vgn
    # Select Parameters Provided
    elif Vgn == Zm == Ip == Ipp is None:
        Ea = Zs * I + Vt
    # Invalid Parameter Set
    else:
        raise ValueError("Invalid Parameter Set")
    return Ea


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
        return y
    # Split out useful values
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag


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
        return y[0].real, y[1:-1].real, -y[1:-1].imag


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
    if imag is not None:
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
    if xrange is None:
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
        if imag is not None:
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
    if imag is not None and not isinstance(imag, (list, _np.ndarray)):
        raise ValueError("Argument *imag* must be array-like.")
    # Calculate Omega
    w = 2 * _np.pi * freq

    def _harmonic_(t):
        out = dc
        for k in range(len(real)):
            # Evaluate Current Coefficient
            A = real[k]
            if imag is not None:
                B = imag[k]
            else:
                B = 0
            m = k + 1
            # Calculate Output
            out += A * _np.cos(m * w * t) + B * _np.sin(m * w * t)
        # Return Value
        return (out)

    if domain is None:
        return _harmonic_  # Return as callable for external use
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
    if VLL == VLN == V is None:
        raise ValueError("One voltage must be specified.")
    elif VLN is not None:
        C = Qc / (2 * _np.pi * freq * 3 * VLN ** 2)
    else:
        if VLL is not None:
            V = VLL
        C = Qc / (2 * _np.pi * freq * V ** 2)
    # Return Value
    return C, Qc


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
    PF:         float, optional
                Power factor to condition the apparent power to an appropriate
                complex value.

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

    Examples
    --------
    >>> import electricpy as ep
    >>> ep.acpiv(S=550, V=167)
    3.2934131736526946
    >>> ep.acpiv(S=550, I=3.2934131736526946)
    (96.4174949546675, 55.66666666666667, 167.0)
    """
    # Validate Inputs
    if S == I is None:
        raise ValueError("To few arguments.")
    # Convert Apparent Power to Complex
    if PF is not None:
        S = S * PF + 1j * _np.sqrt(S ** 2 - (S * PF) ** 2)
    # Solve Single-Phase
    if V is not None:
        if S is None:  # Solve for Apparent Power
            S = V * _np.conj(I)
            return S
        else:  # Solve for Current
            I = _np.conj(S / V)
            return I
    # Solve Line-to-Line
    elif VLL is not None:
        if S is None:  # Solve for Apparent Power
            S = _np.sqrt(3) * VLL * _np.conj(I)
            return S
        else:  # Solve for Current
            I = _np.conj(S / (_np.sqrt(3) * VLL))
            return I
    # Solve Line-to-Neutral
    elif VLN is not None:
        if S is None:  # Solve for Apparent Power
            S = 3 * VLN * _np.conj(I)
            return S
        else:  # Solve for Current
            I = _np.conj(S / (3 * VLN))
            return I
    # Solve for Voltages
    else:
        V = S / _np.conj(I)
        VLL = S / (_np.sqrt(3) * _np.conj(I))
        VLN = S / (3 * _np.conj(I))
        return VLL, VLN, V


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
        return val * Ns / Np
    return val * Np / Ns


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
        return val * Np / Ns
    return val * Ns / Np


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
    ts = (Vgen * Vgen) / (Vgen * Vdis - (R * Pload + X * Qload))
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
                        Ratio of disk capacitance and pin to pole air
                        capacitance
    Voltage:            float
                        Voltage difference between the transmission line and
                        ground

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

    string_efficiency = (
        (Voltage * 100) / (number_capacitors * capacitor_disk_voltages[-1, 0])
    )

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
        return dA / avg, dB / avg, dC / avg
    else:
        return unbalance


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
        return _np.cos(2 * _np.pi * k / Srate)

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
        return cosf, xarray
    return cosf


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
        return _np.sin(2 * _np.pi * k / Srate)

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
        return sinf, xarray
    return sinf


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
    return De


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
    if Rd is None:
        Rd = freq * carson_r
    # Generate Dsgw if Not Provided
    if Dsgw is None and dia_gw is not None:
        Dsgw = _np.exp(-1 / 4) * dia_gw / 2
    # Generate Real Part
    if Rd > 0:
        # Generate Rself if not Provided
        if Rself is None:
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
        if Ds is None:
            raise ValueError("Distance Self (Ds) Required")
        # De must be generated
        if De is None:
            if rho is None:
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
                                 [[_np.log(De / Dagw)], [_np.log(De / Dbgw)],
                                  [_np.log(De / Dcgw)]],
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
    return Zperlen


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
    Zeq = fabc * Zeq + fcab * (_Rp.dot(Zeq.dot(Rp))) + \
        fbca * (Rp.dot(Zeq.dot(_Rp)))
    Zeq = Zeq * linelen
    return Zeq


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
    return GMD


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

    Examples
    --------
    >>> import electricpy as ep
    >>> # Demonstrate the generic functionality
    >>> ep.vipf(V=480, I=ep.phasors.phasor(20, 120))
    (480, (-9.999999999999996+17.320508075688775j), -0.499999...)
    >>> # Find the power factor
    >>> ep.vipf(V=480, I=ep.phasors.phasor(20, 120), find="PF")
    -0.49999...
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
        return V
    if find == 'I':
        return I
    if find == 'PF':
        return PF
    return V, I, PF


# Define Synchronous Speed Calculator
def syncspeed(Npol, freq=60, Hz=False, rpm=False):
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
    rpm:        bool, optional
                Boolean control to enable return in rpm. default=False


    Returns
    -------
    wsyn:       float
                Synchronous Speed of Induction Machine, defaults to units of
                rad/sec, but may be set to Hertz or RPM if `Hz` or `rpm` set to True.
    """
    if Npol == 0:
        raise ZeroDivisionError("Poles of an electrical machine \
        can not be zero")
    wsyn = 2 * _np.pi * freq / (Npol / 2)
    if Hz:
        return (2*freq / (Npol))
    if rpm:
        return (120 * freq)/(Npol)
    return wsyn


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
    return slip


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
    if isinstance(VA, (tuple, list)) and VB == 0 and VC == 0:
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
    return Valpha


def wireresistance(length=None, diameter=None, rho=16.8 * 10 ** -9, R=None):
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
    if R == length == diameter is None:
        raise ValueError("To few arguments.")
    # Given length and diameter
    if length is not None and diameter is not None:
        # calculating the area
        A = pi * (diameter ** 2) / 4
        return rho * length / A
    # Given resistance and diameter
    if R is not None and diameter is not None:
        # calculating the area
        A = pi * (diameter ** 2) / 4
        return R * A / rho
    # Given resistance and length
    if R is not None and length is not None:
        A = rho * length / R
        return _np.sqrt(4 * A / pi)


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
    if C == A == d is None:
        raise ValueError("To few arguments.")
    # Given area and distance
    if A is not None and d is not None:
        return e * A / d
    # Given capacitance and distance
    if C is not None and d is not None:
        return d * C / e
    # Given capacitance and area
    if C is not None and A is not None:
        return e * A / C


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
    if L == A == l == N is None:
        raise ValueError("To few arguments.")
    # Given area, length and number of turns
    if A is not None and l is not None and N is not None:
        return N ** 2 * u * A / l
    # Given inductance, length and number of turns
    if L is not None and l is not None and N is not None:
        return L * l / (N ** 2 * u)
    # Given inductance, area and number of turns
    if L is not None and A is not None and N is not None:
        return N ** 2 * u * A / L
    # Given inductance, area and length
    if L is not None and A is not None and l is not None:
        return _np.sqrt(L * l / (u * A))


def ic_555_astable(R=None, C=None, freq=None, t_high=None, t_low=None):
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
    if R is not None and C is not None:
        if len(R) != 2:
            raise ValueError(
                "Monostable 555 IC will have only 2 resitances to be fixed "
                f"but {len(R)} were given"
            )

        [R1, R2] = R

        T = _np.log(2) * C * (R1 + 2 * R2)
        freq = 1 / T
        t_low = _np.log(2) * C * R2
        t_high = _np.log(2) * C * (R1 + R2)
        duty_cycle = t_high * 100 / T

        return {
            'time_period': T,
            'frequency': freq,
            'duty_cycle': duty_cycle,
            't_low': t_low,
            't_high': t_high
        }

    if t_high is not None and t_low is not None and C is not None:

        x2 = t_low / C * _np.log(2)
        x1 = t_high / C * _np.log(2)
        T = t_high + t_low
        freq = 1 / (T)
        duty_cycle = t_high / (T)

        return {
            'time_period': T,
            'frequency': freq,
            'duty_cycle': duty_cycle,
            'R1': x1 - x2,
            'R2': x2
        }
    raise TypeError("Not enough parqmeters are passed")


def ic_555_monostable(R=None, C=None, freq=None, t_high=None, t_low=None):
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
    T = t_high + t_low
    if R is None:
        if not (C is not None and T is not None):
            raise ValueError(
                "To find Resitance, Capacitance and delay time should be "
                "provided"
            )
        return T / (_np.log(3) * C)
    if C is None:
        if not (R is not None and T is not None):
            raise ValueError(
                "To find Capacitance , Resistance and delay time should be "
                "provided"
            )
        return T / (_np.log(3) * R)

    if T is None:
        if not (R is not None and T is not None):
            raise ValueError(
                "To find Time delay , Resistance and Capacitance should be "
                "provided"
            )
        return R * C * _np.log(3)


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
    x = Adb / 20

    R1 = Z0 * (_np.power(10, x) - 1) / (_np.power(10, x) + 1)
    R2 = 2 * Z0 * _np.power(10, x) / (_np.power(10, 2 * x) - 1)

    return R1, R2


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
    x = Adb / 20

    R1 = Z0 * (_np.power(10, x) + 1) / (_np.power(10, x) - 1)
    R2 = (Z0 / 2) * (_np.power(10, x) - (1 / (_np.power(10, x))))

    return R1, R2


# Calculate Zener Diode Resistor
def zener_diode_required_resistor(Vin, Vo, I):
    r"""
    Zener diode required resistance function .

    A zener diode is uses to allow current to flow "backwards" when the zener
    voltage is reached. This function use to calculate the required resistor
    value following below formula:

    .. math:: R = \frac{V_{in(min)} - V_{out}}{I_{load}+0.01}

    .. image:: /static/zenerdiode.png

    Parameters
    ----------
    Vin:        float
                Minimum input Voltage in Volt
    Vo:         float
                Output Voltage in Volt
    I:          float
                Load Current in Ampere

    Returns
    -------
    R:          float
                Load Resistance in Ohm
    """
    # Solve Load Resistance
    R = (Vin - Vo) / (I+0.01)
    return (R)

# Calculate Zener Diode Power


def zener_diode_power(Vin, Vo, R):
    r"""
    Zener diode power loss function.

    A zener diode is uses to allow current to flow "backwards" when the zener
    voltage is reached. This function use to calculate the power in resistor
    following below formula:

    .. math:: P_R = \frac{(V_{out} - V_{in(max)})^2}{R}

    .. image:: /static/zenerdiode.png

    Parameters
    ----------
    Vin:        float
                Maximum input Voltage in Volt
    Vo:         float
                Output Voltage in Volt
    R:          float
                Load Resistance in Ohm

    Returns
    -------
    P:          float
                Power on resistance in Watt
    """
    # Validate Inputs
    if R == 0:
        raise ValueError("Resistance Value can not be zero")

    # Solve Load Resistance
    P = ((Vo - Vin) ** 2) / R
    return (P)


def lm317(r1, r2, v_out):
    r"""
    LM317 linear voltage regulator solver.

    The LM317 is a linear voltage regulator that can be adjusted to supply a
    specific output voltage. The LM317 has three pins, adjust, output and input.
    The LM317 is often connected as in the image below. [1]_


    .. image:: https://www.basictables.com/media/lm317-circuit.png


    Formula to Calculate Output Voltage, R1, R2:

    .. math:: V_{out} = 1.25 * (1+\frac{R2}{R1})

    .. math:: R1 = \frac{1.25*R2}{V_{out}-1.25}

    .. math:: R2 = \frac{R1*V_{out}}{1.25 - R1}

    Parameters
    ----------
    v_out: float, Optional
           Output Voltage in LM317 in Volts
    r1:    float, Optional
           r1 is resistance and is measured in ohm
    r2:    float, Optional
           r2 is resistance and is measured in ohm

    Returns
    -------
    v_out: float
           v_out is the output voltage and is measured in volt (V)
    r1:    float
           r1 is resistance and is measured in ohm
    r2:    float
           r2 is resistance and is measured in ohm


    .. [1] Electronial, "LM317" BasicTables, Accessed May, 2022
       https://www.basictables.com/electronics/lm317
    """
    if r1 is not None and r2 is not None:
        # Returns Voltage
        return 1.25 * (1 + (r2 / r1))

    if r2 is not None and v_out is not None:
        # Returns R1
        return (1.25 * r2) / (v_out - 1.25)

    if r1 is not None and v_out is not None:
        # Returns R2
        return ((r1 * v_out) / 1.25) - r1

    raise ValueError("Invalid arguments")


# Define Module Specific Variables
_name_ = NAME
_version_ = VERSION
__version__ = _version_  # Alias Version for User Ease

# END OF FILE
