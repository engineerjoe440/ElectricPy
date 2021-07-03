################################################################################
"""
`electricpy` Package - `passives` Module.

>>> from electricpy import passsives as pas

Filled with calculators, evaluators, and plotting functions related to
capacitors, inductors, and resistors. This package will provide a wide array of
capabilities to any electrical engineer.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################


# Import Required Packages
import numpy as _np


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


# Define Capacitor Voltage Discharge Function
def vcapdischarge(t, Vs, R, C):
    r"""
    Discharging Capacitor Function.

    Function to calculate the voltage of a
    capacitor that is discharging given the time.

    .. math:: V_c=V_s*e^{\frac{-t}{R*C}}

    Parameters
    ----------
    t:          float
                The time at which to calculate the voltage.
    Vs:         float
                The starting voltage for the capacitor.
    R:          float
                The ohmic value of the resistor being used
                to discharge.
    C:          float
                Capacitive value (in Farads).

    Returns
    -------
    Vc:         float
                The calculated voltage of the capacitor.
    """
    Vc = Vs * (_np.exp(-t / (R * C)))
    return (Vc)


# Define Capacitor Voltage Charge Function
def vcapcharge(t, Vs, R, C):
    r"""
    Charging Capacitor Voltage.

    Function to calculate the voltage of a
    capacitor that is charging given the time.

    .. math:: V_c=V_s*(1-e^{\frac{-t}{R*C}})

    Parameters
    ----------
    t:          float
                The time at which to calculate the voltage.
    Vs:         float
                The charging voltage for the capacitor.
    R:          float
                The ohmic value of the resistor being used
                to discharge.
    C:          float
                Capacitive value (in Farads).

    Returns
    -------
    Vc:         float
                The calculated voltage of the capacitor.
    """
    Vc = Vs * (1 - _np.exp(-t / (R * C)))
    return (Vc)


# Define Capacitive Energy Transfer Function
def captransfer(t, Vs, R, Cs, Cd):
    """
    Capacitor Energy Transfer Function.

    Calculate the voltage across a joining
    resistor (R) that connects Cs and Cd, the
    energy-source and -destination capacitors,
    respectively. Calculate the final voltage
    across both capacitors.

    Parameters
    ----------
    t:          float
                Time at which to calculate resistor voltage.
    Vs:         float
                Initial voltage across source-capacitor (Cs).
    R:          float
                Value of resistor that connects capacitors.
    Cs:         float
                Source capacitance value in Farads.
    Cd:         float
                Destination capacitance value in Farads.

    Returns
    -------
    rvolt:      float
                Voltage across the resistor at time t.
    vfinal:     float
                Final voltage that both capacitors settle to.
    """
    tau = (R * Cs * Cd) / (Cs + Cd)
    rvolt = Vs * _np.exp(-t / tau)
    vfinal = Vs * Cs / (Cs + Cd)
    return (rvolt, vfinal)


# Define Inductor Energy Formula
def inductorenergy(L, I):
    r"""
    Energy Stored in Inductor Formula.

    Function to calculate the energy stored in an inductor
    given the inductance (in Henries) and the current.

    .. math:: E=\frac{1}{2}*L*I^2

    Parameters
    ----------
    L:          float
                Inductance Value (in Henries)
    I:          float
                Current traveling through inductor.

    Returns
    -------
    E:          float
                The energy stored in the inductor (in Joules).
    """
    return (1 / 2 * L * I ** 2)


# Define Inductor Charge Function
def inductorcharge(t, Vs, R, L):
    r"""
    Charging Inductor Formula.

    Calculates the Voltage and Current of an inductor
    that is charging/storing energy.

    .. math::
       V_L = V_s*e^{\frac{-R*t}{L}}//
       I_L = \frac{V_s}{R}*(1-e^{\frac{-R*t}{L}})

    Parameters
    ----------
    t:          float
                Time at which to calculate voltage and current.
    Vs:         float
                Charging voltage across inductor and resistor.
    R:          float
                Resistance related to inductor.
    L:          float
                Inductance value in Henries.

    Returns
    -------
    Vl:         float
                Voltage across inductor at time t.
    Il:         float
                Current through inductor at time t.
    """
    Vl = Vs * _np.exp(-R * t / L)
    Il = Vs / R * (1 - _np.exp(-R * t / L))
    return (Vl, Il)


# Define Capacitive Back-to-Back Switching Formula
def capbacktoback(C1, C2, Lm, VLN=None, VLL=None):
    """
    Back to Back Capacitor Transient Current Calculator.

    Function to calculate the maximum current and the
    frequency of the inrush current of two capacitors
    connected in parallel when one (energized) capacitor
    is switched into another (non-engergized) capacitor.

    .. note:: This formula is only valid for three-phase systems.

    Parameters
    ----------
    C1:         float
                The capacitance of the
    VLN:        float, exclusive
                The line-to-neutral voltage experienced by
                any one of the (three) capacitors in the
                three-phase capacitor bank.
    VLL:        float, exclusive
                The line-to-line voltage experienced by the
                three-phase capacitor bank.

    Returns
    -------
    imax:       float
                Maximum Current Magnitude during Transient
    ifreq:      float
                Transient current frequency
    """
    # Evaluate Max Current
    imax = _np.sqrt(2 / 3) * VLL * _np.sqrt((C1 * C2) / ((C1 + C2) * Lm))
    # Evaluate Inrush Current Frequency
    ifreq = 1 / (2 * _np.pi * _np.sqrt(Lm * (C1 * C2) / (C1 + C2)))
    return (imax, ifreq)


# Define Inductor Discharge Function
def inductordischarge(t, Io, R, L):
    r"""
    Discharging Inductor Formula.

    Calculates the Voltage and Current of an inductor
    that is discharging its stored energy.

    .. math::
       I_L=I_0*e^{\frac{-R*t}{L}}//
       V_L=I_0*R*(1-e^{\frac{-R*t}{L}})

    Parameters
    ----------
    t:          float
                Time at which to calculate voltage and current.
    Io:         float
                Initial current traveling through inductor.
    R:          float
                Resistance being discharged to.
    L:          float
                Inductance value in Henries.

    Returns
    -------
    Vl:         float
                Voltage across inductor at time t.
    Il:         float
                Current through inductor at time t.
    """
    Il = Io * _np.exp(-R * t / L)
    Vl = Io * R * (1 - _np.exp(-R * t / L))
    return (Vl, Il)


# Define Apparent Power to Farad Conversion
def farads(VAR, V, freq=60):
    r"""
    Capacitance from Apparent Power Formula.

    Function to calculate the required capacitance
    in Farads to provide the desired power rating
    (VARs).

    .. math:: C = \frac{VAR}{2*\pi*freq*V^2}

    Parameters
    ----------
    VAR:        float
                The rated power to meet.
    V:          float
                The voltage across the capacitor;
                not described as VLL or VLN, merely
                the capacitor voltage.
    freq:       float, optional
                The System frequency

    Returns
    -------
    C:          float
                The evaluated capacitance (in Farads).
    """
    return (VAR / (2 * _np.pi * freq * V ** 2))


# Define Capacitor Energy Calculation
def capenergy(C, V):
    r"""
    Capacitor Energy Formula.

    A simple function to calculate the stored voltage (in Joules)
    in a capacitor with a charged voltage.

    .. math:: E=\frac{1}{2}*C*V^2

    Parameters
    ----------
    C:          float
                Capacitance in Farads.
    V:          float
                Voltage across capacitor.

    Returns
    -------
    energy:     float
                Energy stored in capacitor (Joules).
    """
    energy = 1 / 2 * C * V ** 2
    return (energy)


# Define Capacitor Voltage Discharge Function
def loadedvcapdischarge(t, vo, C, P):
    # noqa: D401    "Loaded" is a valid word for this docstring
    r"""
    Loaded Capacitor Discharge Formula.

    Returns the voltage of a discharging capacitor after time (t -
    seconds) given initial voltage (vo - volts), capacitor size
    (cap - Farads), and load (P - Watts).

    .. math:: V_t=\sqrt{v_0^2-2*P*\frac{t}{C}}

    Parameters
    ----------
    t:          float
                Time at which to calculate voltage.
    vo:         float
                Initial capacitor voltage.
    C:          float
                Capacitance (in Farads)
    P:          float
                Load power consumption (in Watts).

    Returns
    -------
    Vt:         float
                Voltage of capacitor at time t.
    """
    Vt = _np.sqrt(vo ** 2 - 2 * P * t / C)
    return (Vt)


# Define Capacitor Discharge Function
def timedischarge(Vinit, Vmin, C, P, dt=1e-3, RMS=True, Eremain=False):
    r"""
    Capacitor Discharge Time Formula.

    Returns the time to discharge a capacitor to a specified
    voltage given set of inputs.

    Parameters
    ----------
    Vinit:      float
                Initial Voltage (in volts)
    Vmin:       float
                Final Voltage (the minimum allowable voltage) (in volts)
    C:          float
                Capacitance (in Farads)
    P:          float
                Load Power being consumed (in Watts)
    dt:         float, optional
                Time step-size (in seconds) (defaults to 1e-3 | 1ms)
    RMS:        bool, optional
                if true converts RMS Vin to peak
    Eremain:    bool, optional
                if true: also returns the energy remaining in cap

    Returns
    -------
    Returns time to discharge from Vinit to Vmin in seconds.
    May also return remaining energy in capacitor if Eremain=True
    """
    t = 0  # start at time t=0
    if RMS:
        vo = Vinit * _np.sqrt(2)  # convert RMS to peak
    else:
        vo = Vinit
    vc = loadedvcapdischarge(t, vo, C, P)  # set initial cap voltage
    while (vc >= Vmin):
        t = t + dt  # increment the time
        vcp = vc  # save previous voltage
        vc = loadedvcapdischarge(t, vo, C, P)  # calc. new voltage
    if (Eremain):
        E = capenergy(C, vcp)  # calc. energy
        return (t - dt, E)
    else:
        return (t - dt)


# Define Rectifier Capacitor Calculator
def rectifiercap(Iload, fswitch, dVout):
    r"""
    Rectifier Capacitor Formula.

    Returns the capacitance (in Farads) for a needed capacitor in
    a rectifier configuration given the system frequency (in Hz),
    the load (in amps) and the desired voltage ripple.

    .. math:: C=\frac{I_{load}}{f_{switch}*\Delta V_{out}}

    Parameters
    ----------
    Iload:      float
                The load current that must be met.
    fswitch:    float
                The switching frequency of the system.
    dVout:      float
                Desired delta-V on the output.

    Returns
    -------
    C:          float
                Required capacitance (in Farads) to meet arguments.
    """
    C = Iload / (fswitch * dVout)
    return (C)


# Define Delta-Wye Impedance Network Calculator
def delta_wye_netz(delta=None, wye=None, round=None):
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
        za, zb, zc = delta_wye_netz(delta = (z1, z5, z4))
        ze1 = zb + z2
        ze2 = zc + z3
        return za + (ze1*ze2)/(ze1+ze2)


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

# END
