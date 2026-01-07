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


# Internal Helpers
def _as_array(x):
    return _np.asarray(x) if isinstance(x, (list, tuple)) else x


def _validate_pf(PF):
    # Clamp for numerical safety
    if PF is None:
        raise ValueError("Power factor (PF) must be specified.")
    PF = float(PF)
    if PF > 1.0:
        PF = 1.0
    if PF < -1.0:
        PF = -1.0
    return PF


def _derive_leakage(Lm, Lls=0, Llr=0, Ls=None, Lr=None):
    # Preserve original semantics: if Ls/Lr provided, they override leakage via Ls-Lm / Lr-Lm
    if Ls is not None:
        Lls = Ls - Lm
    if Lr is not None:
        Llr = Lr - Lm
    return Lls, Llr


def _to_reactances(Lm, Lls=0, Llr=0, freq=60, calcX=True):
    # If calcX True: inputs are inductances (H) and convert to reactances (ohms)
    # If calcX False: inputs are already reactances (ohms) and should be used directly
    if calcX:
        w = 2 * _np.pi * freq
        return (Lm * w), (Lls * w), (Llr * w), w
    else:
        w = 2 * _np.pi * freq
        return (Lm), (Lls), (Llr), w


# Define Transformer Short-Circuit/Open-Circuit Function
def transformertest(Poc=None, Voc=None, Ioc=None, Psc=None, Vsc=None,
                    Isc=None):
    """
    Electrical Transformer Rated Test Evaluator.

    This function will determine the non-ideal circuit components of
    a transformer (Req and Xeq, or Rc and Xm) given the test-case
    parameters for the open-circuit test and/or the closed-circuit
    test. Requires one or both of two sets: { Poc, Voc, Ioc }, or
    { Psc, Vsc, Isc }.
    All values given must be given as absolute value, not complex.
    All values returned are given with respect to primary.
    """
    SC = False
    OC = False

    # Given Open-Circuit Values
    if (Poc is not None) and (Voc is not None) and (Ioc is not None):
        if Voc == 0 or Ioc == 0:
            raise ValueError("Voc and Ioc must be non-zero for open-circuit test.")
        PF = Poc / (Voc * Ioc)
        PF = _validate_pf(PF)
        # Admittance magnitude = I/V, angle = -acos(PF) (lagging magnetizing current)
        Y = _c.rect(Ioc / Voc, -_np.arccos(PF))
        if Y.real == 0:
            raise ValueError("Invalid open-circuit data produced zero conductance.")
        Rc = 1 / Y.real
        if Y.imag == 0:
            raise ValueError("Invalid open-circuit data produced zero susceptance.")
        Xm = -1 / Y.imag
        OC = True

    # Given Short-Circuit Values
    if (Psc is not None) and (Vsc is not None) and (Isc is not None):
        if Vsc == 0 or Isc == 0:
            raise ValueError("Vsc and Isc must be non-zero for short-circuit test.")
        PF = Psc / (Vsc * Isc)
        PF = _validate_pf(PF)
        # Impedance magnitude = V/I, angle = acos(PF) (assume inductive)
        Zeq = _c.rect(Vsc / Isc, _np.arccos(abs(PF)))
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
    """
    # Define Direction Dictionary
    orientation = {
        "DY": 1,
        "YD": -1,
        "DD": 0,
        "YY": 0,
    }
    key = str(style).upper()
    if key not in orientation:
        raise ValueError("Invalid transformer style. Use one of: 'DY','YD','DD','YY'.")
    # Find Direction
    v = orientation[key]
    # Calculate Shift
    phase = _np.exp(1j * _np.radians(v * shift))
    # Return
    return (phase)

# Alias to original Name.
xfmphs = phase_shift_transformer


# Define Induction Machine Thevenin Voltage Calculator
def indmachvth(Vas, Rs, Lm, Lls=0, Ls=None, freq=60, calcX=True):
    r"""
    Induction Machine Thevenin Voltage Calculator.

    .. math:: V_{th}=\frac{j\omega L_m}{R_s+j\omega(L_{ls}+L_m)}V_{as}
    """
    # Condition Inputs (do not mutate user-provided variables)
    if Ls is not None:  # Use Ls instead of Lls
        Lls = Ls - Lm

    Xlm, Xls, _, _ = _to_reactances(Lm, Lls, 0, freq, calcX)

    # Calculate Thevenin Voltage, Return
    Vth = 1j * Xlm / (Rs + 1j * (Xls + Xlm)) * Vas
    return (Vth)


# Define Induction Machine Thevenin Impedance Calculator
def indmachzth(Rs, Lm, Lls=0, Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Thevenin Impedance Calculator.

    .. math::
       Z_{th} = \frac{(R_s+j\omega L_{ls})j\omega L_m}
       {R_s+j\omega(L_{ls}+L_m)}+j\omega L_{lr}
    """
    # Condition Inputs
    Lls, Llr = _derive_leakage(Lm, Lls, Llr, Ls, Lr)

    Xlm, Xls, Xlr, _ = _to_reactances(Lm, Lls, Llr, freq, calcX)

    # Calculate Thevenin Impedance
    Zth = (Rs + 1j * Xls) * (1j * Xlm) / (Rs + 1j * (Xls + Xlm)) + 1j * Xlr
    return (Zth)


# Define Induction Machine Mechancal Power Calculator
def indmachpem(slip, Rr, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0,
               Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Mechanical Power Calculator for Induction Machines.

    Uses thevenin-equivalent terms.
    """
    if slip == 0:
        raise ValueError("slip must be non-zero.")
    if slip < 0:
        raise ValueError("slip must be positive for motoring conventions used here.")

    # Electrical rad/sec (used in power conversion term)
    w = 2 * _np.pi * freq

    # Test for Valid Input Set and Calculate Vth/Zth if needed
    if Vth is None:
        if not all((Vas, Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Vth.")
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)

    if Zth is None:
        if not all((Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Zth.")
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)

    # Use Terms to Calculate Pem
    Rth = Zth.real
    Xth = Zth.imag
    Pem = (abs(Vth) ** 2 * (Rr / slip)) / (((Rr / slip + Rth) ** 2 + (Xth ** 2)) * w) * (1 - slip)
    return (Pem)


# Define Induction Machine Torque Calculator
def indmachtem(slip, Rr, p=0, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0,
               Llr=0, Ls=None, Lr=None, wsyn=None, freq=60, calcX=True):
    r"""
    Induction Machine Torque Calculator.
    """
    if slip == 0:
        raise ValueError("slip must be non-zero.")
    if not any((p, wsyn)):
        raise ValueError("Poles or Synchronous Speed must be specified.")

    # Electrical rad/sec
    w = 2 * _np.pi * freq

    # Determine synchronous mechanical speed
    if wsyn is None:
        if p == 0:
            raise ValueError("If wsyn is not specified, p must be non-zero.")
        wsyn = w / (p / 2)

    # Calculate Vth/Zth if needed
    if Vth is None:
        if not all((Vas, Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Vth.")
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)

    if Zth is None:
        if not all((Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Zth.")
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)

    # Use Terms to Calculate Tem
    Rth = Zth.real
    Xth = Zth.imag
    Tem = 3 * (abs(Vth) ** 2) * (Rr / slip) / (((Rr / slip + Rth) ** 2 + (Xth ** 2)) * wsyn)
    return Tem


# Define Induction Machine Peak Slip Calculator
def indmachpkslip(Rr, Zth=None, Rs=0, Lm=0, Lls=0, Llr=0, Ls=None,
                  Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Slip at Peak Torque Calculator.

    .. math:: \text{slip} = \frac{R_r}{|Z_{th}|}
    """
    if Rr is None:
        raise ValueError("Rr must be specified.")

    if Zth is None:
        if not all((Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Zth.")
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)

    if abs(Zth) == 0:
        raise ValueError("Invalid Zth: magnitude is zero.")
    s_peak = Rr / abs(Zth)
    return s_peak


# Define Induction Machine Phase-A, Rotor Current Calculator
def indmachiar(poles=0, Vth=None, Zth=None, Vas=0, Rs=0, Lm=0, Lls=0, Llr=0,
               Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Rotor Current Calculator.

    .. math:: I_{a_{\text{rotor}}} = \frac{V_{th}}{|Z_{th}|+Z_{th}}
    """
    if Vth is None:
        if not all((Vas, Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Vth.")
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)

    if Zth is None:
        if not all((Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Zth.")
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)

    den = abs(Zth) + Zth
    if den == 0:
        raise ValueError("Invalid operating point: |Zth| + Zth equals zero.")
    Iar = Vth / den
    return Iar


# Define Induction Machine Peak Torque Calculator
def indmachpktorq(Rr, poles=0, s_pk=None, Iar=None, Vth=None, Zth=None, Vas=0,
                  Rs=0, Lm=0, Lls=0, Llr=0, Ls=None, Lr=None, freq=60,
                  calcX=True):
    r"""
    Induction Machine Peak Torque Calculator.
    """
    if Vth is None:
        if not all((Vas, Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Vth.")
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)

    if Zth is None:
        if not all((Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Zth.")
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)

    if Iar is None:
        Iar = indmachiar(Vth=Vth, Zth=Zth)

    if s_pk is None:
        s_pk = indmachpkslip(Rr=Rr, Zth=Zth)

    if s_pk == 0:
        raise ValueError("s_pk must be non-zero.")
    Tpk = abs(Iar) ** 2 * Rr / s_pk
    return Tpk


# Define Induction Machine Starting Torque Calculator
def indmachstarttorq(Rr, poles=0, Iar=None, Vth=None, Zth=None, Vas=0, Rs=0,
                     Lm=0, Lls=0, Llr=0, Ls=None, Lr=None, freq=60, calcX=True):
    r"""
    Induction Machine Starting Torque Calculator.
    """
    slip = 1

    if Vth is None:
        if not all((Vas, Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Vth.")
        Vth = indmachvth(Vas, Rs, Lm, Lls, Ls, freq, calcX)

    if Zth is None:
        if not all((Rs, Lm)):
            raise ValueError("Invalid Argument Set, too few provided to compute Zth.")
        Zth = indmachzth(Rs, Lm, Lls, Llr, Ls, Lr, freq, calcX)

    if Iar is None:
        Iar = Vth / (Rr / slip + Zth)

    Tstart = abs(Iar) ** 2 * Rr / slip
    return Tstart


# Define Induction Machine Stator Torque Calculator
def pstator(Pem, slip):
    r"""
    Stator Power Calculator for Induction Machine.

    .. math:: P_s=\frac{P_{em}}{1-\text{slip}}
    """
    if (1 - slip) == 0:
        raise ValueError("Invalid slip: 1 - slip equals zero.")
    Ps = Pem / (1 - slip)
    return Ps


# Define Induction Machine Rotor Torque Calculator
def protor(Pem, slip):
    r"""
    Rotor Power Calculator for Induction Machine.

    .. math:: P_r=-\text{slip}\cdot\frac{P_{em}}{1-\text{slip}}
    """
    if (1 - slip) == 0:
        raise ValueError("Invalid slip: 1 - slip equals zero.")
    Pr = -slip * (Pem / (1 - slip))
    return Pr


# Define FOC IM Rated Value Calculator
def indmachfocratings(Rr, Rs, Lm, Llr=0, Lls=0, Lr=None,
                      Ls=None, Vdqs=1, Tem=1, wes=1):
    r"""
    FOC Ind. Machine Rated Operation Calculator.
    """
    # Condition Inputs:
    if Ls is None:  # Use Lls instead of Ls
        Ls = Lls + Lm
    if Lr is None:  # Use Llr instead of Lr
        Lr = Llr + Lm

    # Define Equations Function as Solver
    def equations(val):
        Idr, Iqr, Ids, Iqs, LAMdr, LAMqr, LAMds, LAMqs, wr = val
        # Force Vdqs to be treated as complex: Vd (real) + j Vq (imag)
        Vds_cmd = _np.real(Vdqs)
        Vqs_cmd = _np.imag(Vdqs)

        A = (Rs * Ids - wes * LAMqs) - Vds_cmd
        B = (Rs * Iqs + wes * LAMds) - Vqs_cmd
        C = (Rr * Idr - (wes - wr) * LAMqr)
        D = (Rr * Iqr + (wes - wr) * LAMdr)
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
    if wes == 0:
        raise ValueError("wes must be non-zero.")
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
    """
    # Condition Inputs:
    if Ls is None:  # Use Lls instead of Ls
        Ls = Lls + Lm
    if Lr is None:  # Use Llr instead of Lr
        Lr = Llr + Lm

    if Ls == 0 or Lr == 0:
        raise ValueError("Ls and Lr must be non-zero.")

    # Calculate Additional Constraints
    sigma = (1 - Lm ** 2 / (Ls * Lr))
    accuracy = 1 + s_err
    if accuracy == 0:
        raise ValueError("Invalid s_err: 1 + s_err equals zero.")

    # Command Values (Transient and Steady State)
    if Lm == 0:
        raise ValueError("Lm must be non-zero.")
    Ids = LAMdr_cmd / Lm

    if LAMdr_cmd == 0:
        raise ValueError("LAMdr_cmd must be non-zero to compute Iqs and slip.")
    Iqs = Tem_cmd / ((Lm / Lr) * LAMdr_cmd)

    wslip = (Rr / (Lr * accuracy)) * (Lm * Iqs) / LAMdr_cmd
    wes = wslip + wr_cmd

    # Stator dq Voltages (Steady State)
    Vds = Rs * Ids - wes * sigma * Ls * Iqs
    Vqs = Rs * Iqs + wes * Ls * Ids

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
    """
    PF = _validate_pf(PF)
    Itmag = abs(Itmag)

    # PF sign convention (as stated in docstring):
    # (+) leading, (-) lagging
    phi = _np.arccos(abs(PF))
    phi_signed = -phi if PF >= 0 else phi

    # Current angle relative to terminal voltage
    angV = _np.angle(Vt_pu)
    angI = angV - phi_signed  # lagging: angI = angV - phi; leading: angI = angV + phi
    It_pu = Itmag * _np.exp(1j * angI)

    # Quadrature-axis angle
    th_q = _np.angle(Vt_pu - (Ra * It_pu + 1j * Xq * It_pu))

    # Approximate direct-axis component current phasor (per provided formulation intent)
    Iad_mag = abs(It_pu) * abs(_np.sin(-phi_signed + th_q))
    Iad = Iad_mag * _np.exp(1j * (th_q - _np.pi / 2))

    # Calculate Eq
    Eq = Vt_pu - (Ra * It_pu + 1j * Xq * It_pu + 1j * (Xd - Xq) * Iad)
    return Eq

# END
