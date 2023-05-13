################################################################################
"""
Conversion Utilities Common for Electrical Engineering.

>>> from electricpy import conversions

Filled with simple conversion functions to help manage unit conversions and the
like, this module is very helpful to electrical engineers.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

from electricpy.constants import WATTS_PER_HP, Aabc, A012, KWH_PER_BTU

# Import Required Packages
import numpy as _np


# Define HP to Watts Calculation
def hp_to_watts(hp):
    r"""
    Horsepower to Watts Formula.

    Calculates the power (in watts) given the
    horsepower.

    .. math:: P_{\text{watts}}=P_{\text{horsepower}}\cdot745.699872

    Same as `watts`.

    Parameters
    ----------
    hp:         float
                The horsepower to compute.

    Returns
    -------
    watts:      float
                The power in watts.
    """
    return hp * WATTS_PER_HP


watts = hp_to_watts  # Make Duplicate Name


# Define Watts to HP Calculation
def watts_to_hp(watt):
    r"""
    Watts to Horsepower Function.

    Calculates the power (in horsepower) given
    the power in watts.

    .. math:: P_{\text{horsepower}}=\frac{P_{\text{watts}}}{745.699872}

    Same as `horsepower`.

    Parameters
    ----------
    watt:      float
                The wattage to compute.

    Returns
    -------
    hp:         float
                The power in horsepower.
    """
    return watt / WATTS_PER_HP


horsepower = watts_to_hp  # Make Duplicate Name


# Define kWh to BTU function and vice-versa
def kwh_to_btu(kWh):
    r"""
    Killo-Watt-Hours to BTU Function.

    Converts kWh (killo-Watt-hours) to BTU (British Thermal Units).

    .. math:: \text{BTU} = \text{kWh}\cdot3412.14

    Same as `btu`.

    Parameters
    ----------
    kWh:        float
                The number of killo-Watt-hours

    Returns
    -------
    BTU:        float
                The number of British Thermal Units
    """
    return kWh * KWH_PER_BTU


btu = kwh_to_btu  # Make Duplicate Name


def btu_to_kwh(BTU):
    r"""
    BTU to Kilo-Watt-Hours Function.

    Converts BTU (British Thermal Units) to kWh (kilo-Watt-hours).

    .. math:: \text{kWh} = \frac{\text{BTU}}{3412.14}

    Same as `kwh`.

    Parameters
    ----------
    BTU:        float
                The number of British Thermal Units

    Returns
    -------
    kWh:        float
                The number of kilo-Watt-hours
    """
    return BTU / KWH_PER_BTU


kwh = btu_to_kwh  # Make Duplicate Name


# Define Simple Radians to Hertz Converter
def rad_to_hz(radians):
    r"""
    Radians to Hertz Converter.

    Accepts a frequency in radians/sec and calculates
    the hertz frequency (in Hz).

    .. math:: f_{\text{Hz}} = \frac{f_{\text{rad/sec}}}{2\cdot\pi}

    Same as `hertz`.

    Parameters
    ----------
    radians:    float
                The frequency (represented in radians/sec)

    Returns
    -------
    hertz:      float
                The frequency (represented in Hertz)
    
    Examples
    --------
    >>> from electricpy import pi
    >>> from electricpy import conversions as conv
    >>> conv.rad_to_hz(4*pi) # 4-pi-radians/second
    2.0...
    """
    return radians / (2 * _np.pi)  # Evaluate and Return


hertz = rad_to_hz  # Make Duplicate Name


# Define Simple Hertz to Radians Converter
def hz_to_rad(hz):
    r"""
    Hertz to Radians Converter.

    Accepts a frequency in Hertz and calculates
    the frequency in radians/sec.

    .. math:: f_{\text{rad/sec}} = f_{\text{Hz}}\cdot2\cdot\pi

    Same as `radsec`.

    Parameters
    ----------
    hz:     float
            The frequency (represented in Hertz)

    Returns
    -------
    radians:    float
                The frequency (represented in radians/sec)
    
    Examples
    --------
    >>> from electricpy import conversions as conv
    >>> conv.hz_to_rad(2) # 2 hz
    12.566...
    """
    return hz * (2 * _np.pi)  # Evaluate and Return


radsec = hz_to_rad  # Make Duplicate Name


# Define Sequence Component Conversion Function
def abc_to_seq(Mabc, reference='A'):
    r"""
    Phase-System to Sequence-System Conversion.

    Converts phase-based values to sequence
    components.

    .. math:: M_{\text{012}}=A_{\text{012}}\cdot M_{\text{ABC}}

    Same as phs_to_seq.

    Parameters
    ----------
    Mabc:       list of complex
                Phase-based values to be converted.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'

    Returns
    -------
    M012:       numpy.array
                Sequence-based values in order of 0-1-2

    See Also
    --------
    seq_to_abc: Sequence to Phase Conversion
    sequence:  Phase Impedance to Sequence Converter

    Examples
    --------
    >>> import electricpy as ep
    >>> import electricpy.conversions as conv
    >>> abc_matrix = [
    ...     ep.phasor(167, 0),
    ...     ep.phasor(167, -120),
    ...     ep.phasor(167, -240),
    ... ]
    >>> conv.abc_to_seq(abc_matrix)
    >>> # Will return a list approximately equal to: [0+0j, 167+0j, 0+0j]
    """
    # Condition Reference:
    reference = reference.upper()
    if reference == 'A':
        M = Aabc
    elif reference == 'B':
        M = _np.roll(Aabc, 1, 0)
    elif reference == 'C':
        M = _np.roll(Aabc, 2, 0)
    else:
        raise ValueError("Invalid Phase Reference.")
    return M.dot(Mabc)


# Define Second Name for abc_to_seq
phs_to_seq = abc_to_seq


# Define Phase Component Conversion Function
def seq_to_abc(M012, reference='A'):
    r"""
    Sequence-System to Phase-System Conversion.

    Converts sequence-based values to phase
    components.

    .. math:: M_{\text{ABC}}=A_{\text{012}}^{-1}\cdot M_{\text{012}}

    Same as seq_to_phs.

    Parameters
    ----------
    M012:       list of complex
                Sequence-based values to convert.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'

    Returns
    -------
    Mabc:       numpy.array
                Phase-based values in order of A-B-C

    See Also
    --------
    abc_to_seq: Phase to Sequence Conversion
    sequence:  Phase Impedance to Sequence Converter

    Examples
    --------
    >>> import electricpy as ep
    >>> import electricpy.conversions as conv
    >>> abc_matrix = [
    ...     ep.phasor(167, 0),
    ...     ep.phasor(167, -120),
    ...     ep.phasor(167, -240),
    ... ]
    >>> seq_quantities = conv.abc_to_seq(abc_matrix)
    >>> # Will return a list approximately equal to: [0+0j, 167+0j, 0+0j]
    >>> phs_quantities = conv.seq_to_abc(seq_quantities)
    >>> # Returned Phase Quantities will Approximately Equal the Original Values
    """
    # Compute Dot Product
    M = A012.dot(M012)
    # Condition Reference:
    reference = reference.upper()
    if reference == 'A':
        pass
    elif reference == 'B':
        M = _np.roll(M, 1, 0)
    elif reference == 'C':
        M = _np.roll(M, 2, 0)
    else:
        raise ValueError("Invalid Phase Reference.")
    return M


# Define Second Name for seq_to_abc
seq_to_phs = seq_to_abc


# Define Sequence Impedance Calculator
def sequencez(Zabc, reference='A', resolve=False, diag=False, rounds=3):
    r"""
    Sequence Impedance Calculator.

    Accepts the phase (ABC-domain) impedances for a
    system and calculates the sequence (012-domain)
    impedances for the same system. If the argument
    `resolve` is set to true, the function will
    combine terms into the set of [Z0, Z1, Z2].

    When resolve is False:

    .. math:: Z_{\text{012-M}}=A_{\text{012}}^{-1}Z_{\text{ABC}}A_{\text{012}}

    When resolve is True:

    .. math:: Z_{\text{012}}=A_{\text{012}}Z_{\text{ABC}}A_{\text{012}}^{-1}

    Parameters
    ----------
    Zabc:       numpy.array of complex
                2-D (3x3) matrix of complex values
                representing the pharo impedance
                in the ABC-domain.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    resolve:    bool, optional
                Control argument to force the function to
                evaluate the individual sequence impedance
                [Z0, Z1, Z2], default=False
    diag:       bool, optional
                Control argument to force the function to
                reduce the matrix to its diagonal terms.
    rounds:      int, optional
                Integer denoting number of decimal places
                resulting matrix should be rounded to.
                default=3

    Returns
    -------
    Z012:       numpy.array of complex
                2-D (3x3) matrix of complex values
                representing the sequence impedance
                in the 012-domain

    See Also
    --------
    seq_to_abc: Sequence to Phase Conversion
    abc_to_seq: Phase to Sequence Conversion
    """
    # Condition Reference
    reference = reference.upper()
    roll_rate = {'A': 0, 'B': 1, 'C': 2}
    # Test Validity
    if reference not in roll_rate:
        raise ValueError("Invalad Phase Reference")
    # Determine Roll Factor
    roll = roll_rate[reference]
    # Evaluate Matrices
    M012 = _np.roll(A012, roll, 0)
    min_v = _np.linalg.inv(M012)
    # Compute Sequence Impedance
    if resolve:
        Z012 = M012.dot(Zabc.dot(min_v))
    else:
        Z012 = min_v.dot(Zabc.dot(M012))
    # Reduce to Diagonal Terms if Needed
    if diag:
        Z012 = [Z012[0][0], Z012[1][1], Z012[2][2]]
    return _np.around(Z012, rounds)


# Define Angular Velocity Conversion Functions
def rad_to_rpm(rad):
    """
    Radians-per-Second to RPM Converter.

    Given the angular velocity in rad/sec, this function will evaluate the
    velocity in RPM (Revolutions-Per-Minute).

    Parameters
    ----------
    rad:        float
                The angular velocity in radians-per-second

    Returns
    -------
    rpm:        float
                The angular velocity in revolutions-per-minute (RPM)
    
    Examples
    --------
    >>> from electricpy import pi
    >>> from electricpy import conversions as conv
    >>> conv.rad_to_rpm(2*pi)
    60.0
    """
    rpm = 60 / (2 * _np.pi) * rad
    return rpm


# Define Angular Velocity Conversion Functions
def rpm_to_rad(rpm):
    """
    RPM to Radians-per-Second Converter.

    Given the angular velocity in RPM (Revolutions-Per-Minute), this function
    will evaluate the velocity in rad/sec.

    Parameters
    ----------
    rpm:        float
                The angular velocity in revolutions-per-minute (RPM)

    Returns
    -------
    rad:        float
                The angular velocity in radians-per-second
    
    Examples
    --------
    >>> from electricpy import pi
    >>> from electricpy import conversions as conv
    >>> conv.rpm_to_rad(60)
    6.28...
    """
    rad = 2 * _np.pi / 60 * rpm
    return rad


# Define Angular Velocity Conversion Functions
def hz_to_rpm(hz):
    """
    Hertz to RPM Converter.

    Given the angular velocity in Hertz, this function will evaluate the
    velocity in RPM (Revolutions-Per-Minute).

    Parameters
    ----------
    hz:         float
                The angular velocity in Hertz

    Returns
    -------
    rpm:        float
                The angular velocity in revolutions-per-minute (RPM)

    Examples
    --------
    >>> from electricpy import conversions as conv
    >>> conv.hz_to_rpm(2) # 2 Hz
    120
    """
    return hz * 60


# Define Angular Velocity Conversion Functions
def rpm_to_hz(rpm):
    """
    RPM to Hertz Converter.

    Given the angular velocity in RPM (Revolutions-Per-Minute), this function
    will evaluate the velocity in Hertz.

    Parameters
    ----------
    rpm:        float
                The angular velocity in revolutions-per-minute (RPM)

    Returns
    -------
    hz:         float
                The angular velocity in Hertz

    Examples
    --------
    >>> from electricpy import conversions as conv
    >>> conv.rpm_to_hz(120) # 120 RPM
    2.0
    """
    return rpm / 60


# Define dBW to Watts converter
def dbw_to_watts(dbw):
    """
    Convert dBW to Watts.

    Given the power in the decibel scale, this function will evaluate the
    power in Watts.

    Parameters
    ----------
    dbw:        float
                Power in the decibel scale (dBW)

    Returns
    -------
    watts       float
                Power in Watts
    """
    return 10 ** (dbw / 10)


# Define Watts to dBW converter
def watts_to_dbw(watt):
    """
    Watt to dBW converter.

    Given the power in watts, this function will evaluate the power in the
    decibel scale.

    Parameters
    ----------
    watt:      float
                Power in Watts
    Return
    ------
    dbw:        Power in the decibel scale (dBW)
    """
    return 10 * _np.log10(watt)


# Define dbW to dBmW converter
def dbw_to_dbmw(dbw):
    """
    Convert dBW to dBmW.

    Given the power in the decibel scale, this function will evaluate the power
    in the decibel-milli-watts scale.

    Parameters
    ----------
    dbw:        float
                Power in the decibel scale (dBW)
    Return
    ------
    dbmw:       float
                Power in the decibel-milli-watts scale (dBmW)
    """
    return dbw + 30


# Define dBmW to dBW converter
def dbmw_to_dbw(dbmw):
    """
    Convert dBmW to dBW.

    Given the power in the decibel milli-watts-scale, this function will evaluate
    the power in the decibel scale.

    Parameters
    ----------
    dbmw:       float
                Power in the decibel-milli-watts scale (dBmW)
    Return
    ------
    dbw:        float
                Power in the decibel scale (dBW)
    """
    return dbmw - 30


# Define dBmW to Watts converter
def dbmw_to_watts(dbmw):
    """
    Convert dbmW to Watts.

    Given the power in the decibel milli-watts-scale, this function will evaluate
    the power in watts.

    Parameters
    ----------
    dbmw:       float
                Power in the decibel-milli-watts scale (dBmW)
    Return
    ------
    watt:       float
                Power in Watts
    """
    dbw = dbmw_to_dbw(dbmw)
    return dbw_to_watts(dbw)


# Define Watts to dBmW converter
def watts_to_dbmw(watt):
    """
    Watts to dBmW.

    Given the power in watts, this function will evaluate
    the power in the decibel milli-watt scale.

    Parameters
    ----------
    watt:       float
                Power in Watts
    Return
    ------
    dbmw:       float
                Power in the decibel-milli-watts scale (dBmW)
    """
    dbw = watts_to_dbw(watt)
    return dbw_to_dbmw(dbw)


# Define Voltage to decibel converter
def voltage_to_db(voltage, ref_voltage):
    """
    Voltage to Decibel.

    Given the voltage and reference voltage, this function will evaluate
    the voltage in the decibel scale.

    Parameters
    ----------
    voltage:     float
                 voltage

    ref_voltage: float
                 Reference voltage
    Return
    ------
    decibel:    float
                voltage in the decibel scale
    """
    return 20 * _np.log10(voltage / ref_voltage)


# Define Decibel to reference Voltage
def db_to_vref(db, voltage):
    """
    Decibel to Reference Voltage.

    Given decibel and voltage, this function will evaluate
    the power of reference voltage.

    Parameters
    ----------
    db:          float
                 voltage in Decibel

    voltage:     float
                 Voltage
    Return
    ------
    ref_voltage: float
                 reference voltage
    """
    return voltage * _np.power(10, -(db / 20))


# Define Decibel to reference Voltage
def db_to_voltage(db, ref_voltage):
    """
    Decibel to Reference Voltage.

    Given decibel and voltage, this function will evaluate
    the power of reference voltage.

    Parameters
    ----------
    db:              float
                     voltage in Decibel

    ref_voltage:     float
                     Ref Voltage
    Return
    ------
    voltage:         float
                     Voltage
    """
    return ref_voltage * _np.power(10, -(db / 20))

# END
