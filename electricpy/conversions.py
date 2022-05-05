################################################################################
"""
`electricpy` Package - `conversions` Module.

>>> from electricpy import conversions

Filled with simple conversion functions to help manage unit conversions and the
like, this module is very helpful to electrical engineers.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

# Import Local Requirements
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
                The horspower to compute.

    Returns
    -------
    watts:      float
                The power in watts.
    """
    return (hp * WATTS_PER_HP)


watts = hp_to_watts  # Make Duplicate Name


# Define Watts to HP Calculation
def watts_to_hp(watts):
    r"""
    Watts to Horsepower Function.

    Calculates the power (in horsepower) given
    the power in watts.

    .. math:: P_{\text{horsepower}}=\frac{P_{\text{watts}}}{745.699872}

    Same as `horsepower`.

    Parameters
    ----------
    watts:      float
                The wattage to compute.

    Returns
    -------
    hp:         float
                The power in horsepower.
    """
    return (watts / WATTS_PER_HP)


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
    return (kWh * KWH_PER_BTU)


btu = kwh_to_btu  # Make Duplicate Name


def btu_to_kwh(BTU):
    r"""
    BTU to Killo-Watt-Hours Function.

    Converts BTU (British Thermal Units) to kWh (killo-Watt-hours).

    .. math:: \text{kWh} = \frac{\text{BTU}}{3412.14}

    Same as `kwh`.

    Parameters
    ----------
    BTU:        float
                The number of British Thermal Units

    Returns
    -------
    kWh:        float
                The number of killo-Watt-hours
    """
    return (BTU / KWH_PER_BTU)


kwh = btu_to_kwh  # Make Duplicate Name


# Define Simple Radians to Hertz Converter
def rad_to_hz(radians):
    r"""
    Radians to Hertz Converter.

    Accepts a frequency in radians/sec and calculates
    the hertzian frequency (in Hz).

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
    """
    return (radians / (2 * _np.pi))  # Evaluate and Return


hertz = rad_to_hz  # Make Duplicate Name


# Define Simple Hertz to Radians Converter
def hz_to_rad(hertz):
    r"""
    Hertz to Radians Converter.

    Accepts a frequency in Hertz and calculates
    the frequency in radians/sec.

    .. math:: f_{\text{rad/sec}} = f_{\text{Hz}}\cdot2\cdot\pi

    Same as `radsec`.

    Parameters
    ----------
    hertz:      float
                The frequency (represented in Hertz)

    Returns
    -------
    radians:    float
                The frequency (represented in radians/sec)
    """
    return (hertz * (2 * _np.pi))  # Evaluate and Return


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
    M012:       numpy.ndarray
                Sequence-based values in order of 0-1-2

    See Also
    --------
    seq_to_abc: Sequence to Phase Conversion
    sequencez:  Phase Impedance to Sequence Converter
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
    return (M.dot(Mabc))


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
    Mabc:       numpy.ndarray
                Phase-based values in order of A-B-C

    See Also
    --------
    abc_to_seq: Phase to Sequence Conversion
    sequencez:  Phase Impedance to Sequence Converter
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
    return (M)


# Define Second Name for seq_to_abc
seq_to_phs = seq_to_abc


# Define Sequence Impedance Calculator
def sequencez(Zabc, reference='A', resolve=False, diag=False, round=3):
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
    Zabc:       numpy.ndarray of complex
                2-D (3x3) matrix of complex values
                representing the phasor impedances
                in the ABC-domain.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    resolve:    bool, optional
                Control argument to force the function to
                evaluate the individual sequence impedances
                [Z0, Z1, Z2], default=False
    diag:       bool, optional
                Control argument to force the function to
                reduce the matrix to its diagonal terms.
    round:      int, optional
                Integer denoting number of decimal places
                resulting matrix should be rounded to.
                default=3

    Returns
    -------
    Z012:       numpy.ndarray of complex
                2-D (3x3) matrix of complex values
                representing the sequence impedances
                in the 012-domain

    See Also
    --------
    seq_to_abc: Sequence to Phase Conversion
    abc_to_seq: Phase to Sequence Conversion
    """
    # Condition Reference
    reference = reference.upper()
    rollrate = {'A': 0, 'B': 1, 'C': 2}
    # Test Validity
    if reference not in rollrate:
        raise ValueError("Invalad Phase Reference")
    # Determine Roll Factor
    roll = rollrate[reference]
    # Evaluate Matricies
    M012 = _np.roll(A012, roll, 0)
    Minv = _np.linalg.inv(M012)
    # Compute Sequence Impedances
    if resolve:
        Z012 = M012.dot(Zabc.dot(Minv))
    else:
        Z012 = Minv.dot(Zabc.dot(M012))
    # Reduce to Diagonal Terms if Needed
    if diag:
        Z012 = [Z012[0][0], Z012[1][1], Z012[2][2]]
    return (_np.around(Z012, round))


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
    """
    rpm = 60 / (2 * _np.pi) * rad
    return (rpm)


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
    """
    rad = 2 * _np.pi / 60 * rpm
    return (rad)


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
    """
    rpm = hz * 60
    return (rpm)


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
    """
    hz = rpm / 60
    return (hz)

# END