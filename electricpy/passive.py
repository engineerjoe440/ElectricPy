################################################################################
"""
`electricpy.capacitor` Package - Capacitor Specific Methods.

Filled with methods related capacitors.
"""
################################################################################
import numpy as _np
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
    if t<0:
        raise ValueError("Time must be greater than or equal to zero.")
    if R*C == 0:
        raise ValueError("Resistance and Capacitance must be non-zero.")
    Vc = Vs * (_np.exp(-t / (R * C)))
    return (Vc)

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
    if t<0:
        raise ValueError("Time must be greater than or equal to zero.")
    if R*C == 0:
        raise ValueError("Resistance and Capacitance must be non-zero.")
    Vc = Vs * (1 - _np.exp(-t / (R * C)))
    return (Vc)

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
    if t<0:
        raise ValueError("Time must be greater than zero.")
    try:
        tau = (R * Cs * Cd) / (Cs + Cd)
        rvolt = Vs * _np.exp(-t / tau)
    except ZeroDivisionError:
        raise ZeroDivisionError("Sum of Source and Destination Capacitance must be non-zero.")
    vfinal = Vs * Cs / (Cs + Cd)
    return (rvolt, vfinal)

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

def air_core_inductor(coil_diameter: float, coil_length: float, turns: int):
    r"""
    Compute Inductance of Air Core Inductor.

    Air core inductors that consist of a coil of conducting wire with no core.
    They are used in all sorts of electronic devices like radios and computers.

    Parameters
    ---------- 
    coil_diameter: float in meters
    coil_length: float in meters
    turns: int inductor turns

    Returns 
    ------- 
    L: float Inductance of air core inductor in (mH)
    """
    k1 = (1000*coil_diameter*coil_diameter) * (turns*turns)
    k2 = (457418*coil_diameter) + (1016127*coil_length)
    return  k1/k2

def inductive_voltdiv(Vin=None, Vout=None, L1=None, L2=None, find=''):
    r"""
    Inductive voltage divider.

    Inductive voltage divider Inductive voltage dividers are made out of two inductors. 
    One of the inductors is connected from the input to the output and the other one is connected from the output to ground. 
    You can also use other components like resistors and inductors.

    .. math:: V_{out} = \frac{V_{in}*L1}{L1+L2}

    .. image:: /static/inductive-voltage-divider-circuit.png

    Parameters
    ----------
    Vin:    float, optional 
            The input voltage for the system, default=None

    Vout:   float, optional
            The output voltage for the system, default=None

    L1:     float,optional
            Value of the inductor above the output voltage, default=None

    L2:     float,optional
            Value of the inductor below the output voltage, default=None

    find:   str, optional
            Control argument to specify which value
            should be returned.
    
    Returns
    -------
    Vin:    float, optional 
            The input voltage for the system, default=None

    Vout:   float, optional
            The output voltage for the system, default=None

    L1:     float,optional
            Value of the inductor above the output voltage, default=None

    L2:     float,optional
            Value of the inductor below the output voltage
    """
    if Vin!=None and L1!=None and L2!=None:
        Vout = (Vin*L1)/(L1+L2)
    elif Vout!=None and L1!=None and L2!=None:
        Vin = (Vout)*(L1+L2)/(L1)
    elif Vin!=None and Vout!=None and L2!=None:
        L1 = L2*(Vin -Vout)/(Vout)
    elif Vin!=None and Vout!=None and L1!=None:
        L2 = L1*(Vout)/(Vin - Vout)
    else:
        raise ValueError("ERROR: Invalid Parameters or too few" +
                        " parameters given to calculate.")

    find = find.lower()
    
    if find == 'vin':
        return Vin
    elif find == 'vout':
        return Vout
    elif find == 'l1':
        return L1
    elif find == 'l2':
        return L2
    else:
        return (Vin, Vout, L1, L2)