################################################################################
"""
Thermal and Sensory Models for Thermal Arithmetic in Electrical Engineering.

Filled with plotting functions and visualization tools for electrical engineers,
this module is designed to assist engineers visualize their designs.
"""
################################################################################

import numpy as _np

from electricpy.constants import *

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
        if not (0 < Tcj < 70):
            raise ValueError("Temperature out of range.")
    else:
        if not (-20 < Tcj < 70):
            raise ValueError("Temperature out of range.")

    # Define Constant Lookup System
    lookup = ["B", "E", "J", "K", "N", "R", "S", "T"]
    if not (coupletype in lookup):
        raise ValueError("Invalid Thermocouple Type")
    index = lookup.index(coupletype)

    # NOTE:
    # The original implementation always overwrote any user-provided To/Vo/P*/Q*
    # because it unconditionally loaded constants from COLD_JUNCTION_DATA.
    # Here we preserve explicit user inputs when provided, and only fall back
    # to defaults for missing values.

    # Load default constants for this thermocouple type
    defaults = {}
    for var in COLD_JUNCTION_DATA.keys():
        defaults[var] = COLD_JUNCTION_DATA[var][index]

    # Use user-provided values when not None, otherwise use defaults
    To = To if To is not None else defaults["To"]
    Vo = Vo if Vo is not None else defaults["Vo"]
    P1 = P1 if P1 is not None else defaults["P1"]
    P2 = P2 if P2 is not None else defaults["P2"]
    P3 = P3 if P3 is not None else defaults["P3"]
    P4 = P4 if P4 is not None else defaults["P4"]
    Q1 = Q1 if Q1 is not None else defaults["Q1"]
    Q2 = Q2 if Q2 is not None else defaults["Q2"]

    # Define Formula Terms
    tx = (Tcj - To)
    num = tx * (P1 + tx * (P2 + tx * (P3 + P4 * tx)))
    den = 1 + tx * (Q1 + Q2 * tx)
    Vcj = Vo + num / den

    # Round Value if Allowed
    if round is not None:
        Vcj = _np.around(Vcj, round)

    # Return in milivolts
    return Vcj * m


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

    # Work in millivolts internally
    V = V / m  # Scale volts to milivolts (m is milli scaling constant)

    # Define Constant Lookup System
    lookup = ["B", "E", "J", "K", "N", "R", "S", "T"]
    if not (coupletype in lookup):
        raise ValueError("Invalid Thermocouple Type")

    # Determine Cold-Junction-Voltage and apply compensation (in mV)
    if cjt is not None:
        Vcj_mV = coldjunction(cjt, coupletype, To, Vo, P1, P2, P3, P4, Q1, Q2, round=None)
        V += Vcj_mV  # V is in mV, so add mV directly

    # Determine Array Selection (bounds are stored in millivolts)
    vset = THERMO_COUPLE_VOLTAGES[coupletype]
    if V < vset[0]:
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

    # Load default constants for this thermocouple type and segment
    defaults = {}
    for i, key in enumerate(THERMO_COUPLE_KEYS):
        defaults[key] = THERMO_COUPLE_DATA[coupletype][i][select]

    # Preserve user-provided constants if present; otherwise use defaults
    Vo = Vo if Vo is not None else defaults["Vo"]
    To = To if To is not None else defaults["To"]
    P1 = P1 if P1 is not None else defaults["P1"]
    P2 = P2 if P2 is not None else defaults["P2"]
    P3 = P3 if P3 is not None else defaults["P3"]
    P4 = P4 if P4 is not None else defaults["P4"]
    Q1 = Q1 if Q1 is not None else defaults["Q1"]
    Q2 = Q2 if Q2 is not None else defaults["Q2"]
    Q3 = Q3 if Q3 is not None else defaults["Q3"]

    # Calculate Temperature in Degrees C
    dv = (V - Vo)
    num = dv * (P1 + dv * (P2 + dv * (P3 + P4 * dv)))
    den = 1 + dv * (Q1 + dv * (Q2 + Q3 * dv))
    temp = To + num / den

    # Return Temperature
    if fahrenheit:
        temp = (temp * 9 / 5) + 32
    temp = _np.around(temp, round)
    return temp


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
    # Condition Inputs
    rtdtype = rtdtype.upper()

    # Validate rtdtype when defaults are used
    if (Rref is None) or (a is None):
        if rtdtype not in RTD_TYPES:
            raise ValueError("Invalid RTD Type")

    # Load Variables
    if Rref is None:
        Rref = RTD_TYPES[rtdtype][0]
    if Tref is None:
        Tref = 0
    if a is None:
        a = RTD_TYPES[rtdtype][1]

    # Define Terms
    num = RT - Rref + Rref * a * Tref
    den = Rref * a
    temp = num / den

    # Return Temperature
    if fahrenheit:
        temp = (temp * 9 / 5) + 32
    temp = _np.around(temp, round)
    return temp

# END
