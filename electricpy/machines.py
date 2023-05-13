################################################################################
"""
Electrical machines formulas for transformers, motors, generators, etc.

>>> from electricpy import machines
"""
################################################################################

import cmath as _c
import numpy as _np
from scipy.optimize import fsolve as _fsolve

from electricpy.phasors import compose


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
    if (Poc is not None) and (Voc is not None) and (Ioc is not None):
        PF = Poc / (Voc * Ioc)
        Y = _c.rect(Ioc / Voc, -_np.arccos(PF))
        Rc = 1 / Y.real
        Xm = -1 / Y.imag
        OC = True
    # Given Short-Circuit Values
    if (Psc is not None) and (Vsc is not None) and (Isc is not None):
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
        raise ValueError(
            "Not enough arguments were provided for transformertest."
        )


# Define Simple Transformer Phase Shift Function
def phase_shift_transformer(style="DY", shift=30):
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
    >>> import electricpy.machines as machines
    >>> # Find shift of Delta-Wye Transformer w/ 30° shift
    >>> shift = machines.xfmphs(style="DY", shift=30)
    >>> ep.cprint(shift)
    1.0 ∠ 30.0°
    >>> shift = machines.phase_shift_transformer(style="DY", shift=30)
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

# Alias to original Name.
xfmphs = phase_shift_transformer


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
    if Ls is not None:  # Use Ls instead of Lls
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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Vth is None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth is None:
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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
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
    if Vth is None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth is None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Use Terms to Calculate Pem
    Rth = Zth.real
    Xth = Zth.imag
    Tem = 3 * abs(Vth) ** 2 / ((Rr / slip + Rth) ** 2 + Xth) * Rr / (slip * wsyn)
    return Tem


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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Zth is None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Calculate Peak Slip
    s_peak = Rr / abs(Zth)
    return s_peak


# Define Induction Machine Phase-A, Rotor Current Calculator
def indmachiar(poles=0, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0, Llr=0,
               Ls=None, Lr=None, freq=60, calcX=True):
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
    poles:      int, optional
                Number of poles for the induction machine.
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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if poles != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (poles / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Vth is None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth is None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    # Calculate Rotor Current
    Iar = Vth / (Zth.real + Zth)
    return Iar


# Define Induction Machine Peak Torque Calculator
def indmachpktorq(Rr, poles=0, s_pk=None, Iar=None, Vth=None, Zth=None, Vas=0,
                  Rs=0, Lm=0, Lls=0, Llr=0, Ls=None, Lr=None, freq=60,
                  calcX=True):
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
    poles:      int, optional
                Number of poles for the induction machine.
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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if poles != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (poles / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Test for Valid Input Set
    if Vth is None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth is None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    if Iar is None:
        if not all((Vth, Zth)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Ias
        Iar = indmachiar(Vth=Vth, Zth=Zth)
    if s_pk is None:
        if not all((Rr, Zth)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Peak Slip
        s_pk = indmachpkslip(Rr=Rr, Zth=Zth)
    # Use Terms to Calculate Peak Torque
    Tpk = abs(Iar) ** 2 * Rr / s_pk
    return Tpk


# Define Induction Machine Starting Torque Calculator
def indmachstarttorq(Rr, poles=0, Iar=None, Vth=None, Zth=None, Vas=0, Rs=0,
                     Lm=0, Lls=0, Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
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
    poles:      int, optional
                Number of poles for the induction machine.
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
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm
    if Lr is not None:  # Use Lr instead of Llr
        Llr = Lr - Lm
    if poles != 0:  # Calculate Sync. Speed from Num. Poles
        wsyn = w / (poles / 2)
    if calcX:  # Convert Inductances to Reactances
        Lm *= w
        Lls *= w
        Llr *= w
    # Slip is 1 (one) for starting
    slip = 1
    # Test for Valid Input Set
    if Vth is None:
        if not all((Vas, Rs, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Vth
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)
    if Zth is None:
        if not all((Rs, Llr, Lm, Lls)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Zth
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)
    if Iar is None:
        if not all((Vth, Zth)):
            raise ValueError("Invalid Argument Set, too few provided.")
        # Valid Argument Set, Calculate Ias
        Iar = Vth / (Rr / slip + Zth)
    # Use Terms to Calculate Peak Torque
    Tstart = abs(Iar) ** 2 * Rr / slip
    return Tstart


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
    return Ps


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
    return Pr


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
    if Ls is None:  # Use Lls instead of Ls
        Ls = Lls + Lm
    if Lr is None:  # Use Llr instead of Lr
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
        return A, B, C, D, E, F, G, H, I

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
    if Ls is None:  # Use Lls instead of Ls
        Ls = Lls + Lm
    if Lr is None:  # Use Llr instead of Lr
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
    return Eq

# END
