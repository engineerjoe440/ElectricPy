################################################################################
"""
Electrical Power Engineering Faults Calculations.

>>> from electricpy import fault
"""
################################################################################

import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import fsolve as _fsolve

from electricpy.constants import *
from electricpy.conversions import seq_to_abc


def _phaseroll(M012, reference):
    # Compute Dot Product
    return seq_to_abc(M012, reference)


# Define Single Line to Ground Fault Function
def single_phase_to_ground_fault(Vth, Zseq, Rf=0, sequence=True, reference='A'):
    r"""
    Single-Phase-to-Ground Fault Calculator.

    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a single-line-to-ground fault.

    .. math:: I_1 = \frac{V_{th}}{Z_0+Z_1+Z_2+3*R_f}

    .. math:: I_2 = I_1

    .. math:: I_0 = I_1
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X0, complex):
        X0 *= 1j
    if not isinstance(X1, complex):
        X1 *= 1j
    if not isinstance(X2, complex):
        X2 *= 1j
    # Calculate Fault Current
    Ifault = Vth / (X0 + X1 + X2 + 3 * Rf)
    Ifault = _np.array([Ifault, Ifault, Ifault])
    # Prepare Value for return
    if not sequence:
        Ifault = _phaseroll(Ifault, reference)  # Convert to ABC-Domain
    # Return Value
    return Ifault

# Alias Original Name
phs1g = single_phase_to_ground_fault


# Define Double Line to Ground Fault Current Calculator
def double_phase_to_ground_fault(Vth, Zseq, Rf=0, sequence=True, reference='A'):
    r"""
    Double-Line-to-Ground Fault Calculator.

    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a double-line-to-ground fault.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X0, complex):
        X0 *= 1j
    if not isinstance(X1, complex):
        X1 *= 1j
    if not isinstance(X2, complex):
        X2 *= 1j
    # Calculate Fault Currents
    If1 = Vth / (X1 + (X2 * (X0 + 3 * Rf)) / (X0 + X2 + 3 * Rf))
    If2 = -(Vth - X1 * If1) / X2
    If0 = -(Vth - X1 * If1) / (X0 + 3 * Rf)
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll(Ifault, reference)  # Convert to ABC-Domain
    return Ifault

# Alias Original Name
phs2g = double_phase_to_ground_fault


# Define Phase-to-Phase Fault Current Calculator
def phase_to_phase_fault(Vth, Zseq, Rf=0, sequence=True, reference='A'):
    r"""
    Line-to-Line Fault Calculator.

    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a phase-to-phase fault.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X0, complex):
        X0 *= 1j
    if not isinstance(X1, complex):
        X1 *= 1j
    if not isinstance(X2, complex):
        X2 *= 1j
    # Calculate Fault Currents
    If0 = 0
    If1 = Vth / (X1 + X2 + Rf)
    If2 = -If1
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll(Ifault, reference)  # Convert to ABC-Domain
    return Ifault

# Alias Original Name
phs2 = phase_to_phase_fault


# Define Three-Phase Fault Current Calculator
def three_phase_fault(Vth, Zseq, Rf=0, sequence=True, reference='A'):
    r"""
    Three-Phase Fault Calculator.

    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a three-phase fault.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X1, complex):
        X1 *= 1j
    # Calculate Fault Currents
    Ifault = Vth / (X1 + Rf)
    Ifault = _np.array([0, Ifault, 0])
    # Prepare to Return Value
    if not sequence:
        Ifault = _phaseroll(Ifault, reference)  # Convert to ABC-Domain
    return Ifault

# Alias Original Name
phs3 = three_phase_fault


# Define Single Pole Open Calculator
def poleopen1(Vth, Zseq, sequence=True, reference='A'):
    r"""
    Single Pole Open Fault Calculator.

    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a single pole open fault.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X0, complex):
        X0 *= 1j
    if not isinstance(X1, complex):
        X1 *= 1j
    if not isinstance(X2, complex):
        X2 *= 1j
    # Calculate Fault Currents
    If1 = Vth / (X1 + (1 / X2 + 1 / X0) ** (-1))
    If2 = -If1 * X0 / (X2 + X0)
    If0 = -If1 * X2 / (X2 + X0)
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll(Ifault, reference)  # Convert to ABC-Domain
    return Ifault


# Define Double Pole Open Calculator
def poleopen2(Vth, Zseq, sequence=True, reference='A'):
    r"""
    Double Pole Open Fault Calculator.

    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a double pole open fault.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X0, complex):
        X0 *= 1j
    if not isinstance(X1, complex):
        X1 *= 1j
    if not isinstance(X2, complex):
        X2 *= 1j
    # Calculate Fault Currents
    If1 = Vth / (X1 + X2 + X0)
    If2 = If1
    If0 = If1
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll(Ifault, reference)  # Convert to ABC-Domain
    return Ifault


# Define MVA Short Circuit
def short_circuit_mva(Zth=None, Isc=None, Vth=1):
    r"""
    Short-Circuit MVA Calculator.
    """
    # Test for too few inputs
    if Zth is None and Isc is None:
        raise ValueError("Either Zth or Isc must be specified.")
    # Condition Inputs
    if Zth is not None:
        Zth = abs(Zth)
    if Isc is not None:
        Isc = abs(Isc)
    if Vth != 1:
        Vth = abs(Vth)
    # Calculate MVA from one of the available methods
    if Zth is not None and Isc is not None:
        MVA = Isc ** 2 * Zth
    elif Zth is not None:
        MVA = Vth ** 2 / Zth
    else:
        MVA = Vth * Isc
    # Return Value
    return MVA

# Alias Original Name
scMVA = short_circuit_mva


# Define Explicitly 3-Phase MVAsc Calculator
def phs3mvasc(Vth, Zseq, Rf=0, Sbase=1):
    r"""
    Three-Phase MVA Short-Circuit Calculator.
    """
    # Calculate Three-Phase MVA
    MVA = abs(Vth) ** 2 / abs(Zseq[1]) * Sbase
    # Scale VA to MVA if Sbase is not 1
    if Sbase != 1:
        MVA = MVA * 1e-6  # Divide by 1e6 (M)
    # Return
    return MVA


# Define Explicitly 1-Phase MVAsc Calculator
def phs1mvasc(Vth, Zseq, Rf=0, Sbase=1):
    r"""
    Single-Phase MVA Short-Circuit Calculator.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if not isinstance(X0, complex):
        X0 *= 1j
    if not isinstance(X1, complex):
        X1 *= 1j
    if not isinstance(X2, complex):
        X2 *= 1j
    # Calculate Fault Current
    Ifault = Vth / (X0 + X1 + X2 + 3 * Rf)
    # Calculate MVA
    MVA = abs(Ifault) ** 2 * abs(X1) * Sbase
    # Scale VA to MVA if Sbase is not 1
    if Sbase != 1:
        MVA = MVA * 1e-6  # Divide by 1e6 (M)
    # Return
    return MVA


# Define Faulted Bus Voltage Calculator
def busvolt(k, n, Vpf, Z0, Z1, Z2, If, sequence=True, reference='A'):
    """
    Faulted Bus Voltage Calculator.
    """
    # Condition Inputs
    k = k - 1
    n = n - 1
    Z0 = _np.asarray(Z0)
    Z1 = _np.asarray(Z1)
    Z2 = _np.asarray(Z2)
    If = _np.asarray(If)
    # Generate Arrays For Calculation
    Vfmat = _np.array([0, Vpf, 0]).T
    Zmat = _np.array([[Z0[k, n], 0, 0],
                      [0, Z1[k, n], 0],
                      [0, 0, Z2[k, n]]])
    # Perform Calculation
    Vf = Vfmat - Zmat.dot(If)
    if not sequence:
        Vf = _phaseroll(Vf, reference)  # Convert to ABC-Domain
    return Vf


# Define CT Saturation Function
def ct_saturation(XoR, Imag, Vrated, Irated, CTR, Rb, Xb, remnance=0, freq=60,
                  ALF=20):
    r"""
    Electrical Current Transformer Saturation Calculator.
    """
    # Define omega
    w = 2 * _np.pi * freq
    # Find Lb
    Lb = Xb / w
    # Re-evaluate Vrated
    Vrated = Vrated * (1 - remnance)
    # Calculate each "term" (multiple)
    t1 = (1 + XoR)
    t2 = (Imag / (Irated * CTR))
    t3 = abs(Rb + 1j * w * Lb) * 100 / Vrated
    # Evaluate
    result = t1 * t2 * t3
    # Test for saturation
    saturation = result >= ALF
    # Return Results
    return result, saturation


# Define C-Class Calculator
def ct_cclass(XoR, Imag, Irated, CTR, Rb, Xb, remnance=0, sat_crit=20):
    r"""
    Electrical Current Transformer (CT) C-Class Function.
    """
    # Calculate each "term" (multiple)
    t1 = (1 + XoR)
    t2 = (Imag / (Irated * CTR))
    t3 = abs(Rb + 1j * Xb) * 100 / sat_crit
    # Evaluate
    Vr_w_rem = t1 * t2 * t3
    c_class = Vr_w_rem / (1 - remnance)
    # Return Result
    return c_class


# Define Saturation Voltage at Rated Burden
def ct_satratburden(Inom, VArat=None, ANSIv=None, ALF=20):
    r"""
    Electrical Current Transformer (CT) Saturation at Rated Burden Calculator.
    """
    # Validate Inputs
    if VArat is None and ANSIv is None:
        raise ValueError("VArat or ANSIv must be specified.")
    elif VArat is None:
        # Calculate VArat from ANSIv
        VArat = Inom * ANSIv / (20)
    # Determine Vsaturation
    Vsat = ALF * VArat / Inom
    return Vsat


# Define CT Vpeak Formula
def ct_vpeak(Zb, Ip, CTR):
    r"""
    Electrical Current Transformer (CT) Peak Voltage Calculator.
    """
    return _np.sqrt(3.5 * abs(Zb) * Ip * CTR)


# Define Saturation Time Calculator
def ct_timetosat(Vknee, XoR, Rb, CTR, Imax, ts=None, npts=100, freq=60,
                 plot=False):
    r"""
    Electrical Current Transformer (CT) Time to Saturation Function.
    """
    # Calculate omega
    w = 2 * _np.pi * freq
    # Calculate Tp
    Tp = XoR / w
    # If ts isn't specified, generate it
    if ts is None:
        ts = _np.linspace(0, 0.1, freq * npts)
    # Calculate inner term
    term = -XoR * (_np.exp(-ts / Tp) - 1)
    # Calculate Vsaturation terms
    Vsat1 = Imax * Rb * (term + 1)
    Vsat2 = Imax * Rb * (term - _np.sin(w * ts))
    Vsat3 = Imax * Rb * (1 - _np.cos(w * ts))
    # If plotting requested
    if plot and isinstance(ts, _np.ndarray):
        _plt.plot(ts, Vsat1, label="Vsat1")
        _plt.plot(ts, Vsat2, label="Vsat2")
        _plt.plot(ts, Vsat3, label="Vsat3")
        _plt.axhline(Vknee, label="V-knee", linestyle='--')
        _plt.title("Saturation Curves")
        _plt.xlabel("Time (ts)")
        _plt.legend()
        _plt.show()
    elif plot:
        print("Unable to plot a single point, *ts* must be a numpy-array.")
    # Determine the crossover points for each saturation curve
    Vsat1c = Vsat2c = Vsat3c = 0
    if isinstance(ts, _np.ndarray):
        for i in range(len(ts)):
            if Vsat1[i] > Vknee and Vsat1c == 0:
                Vsat1c = ts[i - 1]
            if Vsat2[i] > Vknee and Vsat2c == 0:
                Vsat2c = ts[i - 1]
            if Vsat3[i] > Vknee and Vsat3c == 0:
                Vsat3c = ts[i - 1]
        results = (Vsat1c, Vsat2c, Vsat3c)
    else:
        results = (Vsat1, Vsat2, Vsat3)
    return results


# Define Function to Calculate TRV
def pktransrecvolt(C, L, R=0, VLL=None, VLN=None, freq=60):
    """
    Peak Transient Recovery Function.
    """
    # Evaluate alpha, omega-n, and fn
    alpha = R / (2 * L)
    wn = _np.sqrt(max(0.0, 1 / (L * C) - alpha ** 2))
    fn = wn / (2 * _np.pi)
    # Evaluate Vm
    if VLL is not None:
        Vm = _np.sqrt(2 / 3) * VLL
    elif VLN is not None:
        Vm = _np.sqrt(2) * VLN
    else:
        raise ValueError("One voltage must be specified.")
    # Evaluate Vcpk (worst case)
    we = 2 * _np.pi * freq
    denom = (wn ** 2 - we ** 2)
    if denom == 0:
        raise ZeroDivisionError("Invalid parameters: wn equals system frequency (resonance).")
    Vcpk = (wn ** 2 / denom) * Vm * 2
    # Evaluate RRRV
    RRRV = 2 * Vm * fn / 0.5
    return Vcpk, RRRV


# Define TRV Reduction Resistor Function
def trvresistor(C, L, reduction, Rd0=500, wd0=260e3, tpk0=10e-6):
    """
    Transient Recovery Voltage (TRV) Reduction Resistor Function.
    """
    # Evaluate omega-n
    wn = 1 / _np.sqrt(L * C)
    # Generate Constant Factor
    fctr = (1 - reduction) * 2 - 1

    # Define Function Set
    def equations(data):
        Rd, wd, tpk = data
        X = _np.sqrt(wn ** 2 - (1 / (2 * Rd * C)) ** 2) - wd
        Y = _np.exp(-tpk / (2 * Rd * C)) - fctr
        Z = wd * tpk - _np.pi
        return (X, Y, Z)

    Rd, wd, tpk = _fsolve(equations, (Rd0, wd0, tpk0))
    return Rd, wd, tpk


# Define Time-Overcurrent Trip Time Function
def toctriptime(I, Ipickup, TD, curve="U1", CTR=1):
    """
    Time OverCurrent Trip Time Function.
    """
    # Condition Inputs
    curve = curve.upper()
    # Define Dictionary of Constants
    const = {"U1": {"A": 0.0104, "B": 0.2256, "P": 0.02},
             "U2": {"A": 5.95, "B": 0.180, "P": 2.00},
             "U3": {"A": 3.88, "B": 0.0963, "P": 2.00},
             "U4": {"A": 5.67, "B": 0.352, "P": 2.00},
             "U5": {"A": 0.00342, "B": 0.00262, "P": 0.02},
             "C1": {"A": 0.14, "B": 0, "P": 0.02},
             "C2": {"A": 13.5, "B": 0, "P": 2.00},
             "C3": {"A": 80.0, "B": 0, "P": 2.00},
             "C4": {"A": 120.0, "B": 0, "P": 2.00},
             "C5": {"A": 0.05, "B": 0, "P": 0.04}}
    # Load Constants
    A = const[curve]["A"]
    B = const[curve]["B"]
    P = const[curve]["P"]
    # Evaluate M
    M = I / (CTR * Ipickup)
    # Evaluate Trip Time
    tt = TD * (A / (M ** P - 1) + B)
    return tt


# Define Time Overcurrent Reset Time Function
def tocreset(I, Ipickup, TD, curve="U1", CTR=1):
    """
    Time OverCurrent Reset Time Function.
    """
    # Condition Inputs
    curve = curve.upper()
    # Define Dictionary of Constants
    C = {"U1": 1.08, "U2": 5.95, "U3": 3.88,
         "U4": 5.67, "U5": 0.323, "C1": 13.5,
         "C2": 47.3, "C3": 80.0, "C4": 120.0,
         "C5": 4.85}
    # Evaluate M
    M = I / (CTR * Ipickup)
    # Evaluate Reset Time
    tr = TD * (C[curve] / (1 - M ** 2))
    return tr


# Define Pickup Current Calculation
def pickup(Iloadmax, Ifaultmin, scale=0, printout=False, units="A"):
    """
    Electrical Current Pickup Selection Assistant.
    """
    IL2 = 2 * Iloadmax
    IF2 = Ifaultmin / 2
    exponent = len(str(IL2).split('.')[0])
    setpoint = _np.ceil(IL2 * 10 ** (-exponent + 1 + scale)) * 10 ** (exponent - 1 - scale)
    if printout:
        print("Range Min:", IL2, units, "\t\tRange Max:", IF2, units)
    if IF2 < setpoint:
        setpoint = IL2
        if IL2 > IF2:
            raise ValueError("Invalid Parameters.")
    if printout:
        print("Current Pickup:", setpoint, units)
    return setpoint


# Define Time-Dial Coordination Function
def tdradial(I, CTI, Ipu_up, Ipu_dn=0, TDdn=0, curve="U1", scale=2, freq=60,
             CTR_up=1, CTR_dn=1, tfixed=None):
    """
    Radial Time Dial Coordination Function.
    """
    # Condition Inputs
    curve = curve.upper()
    # Define Dictionary of Constants
    const = {"U1": {"A": 0.0104, "B": 0.2256, "P": 0.02},
             "U2": {"A": 5.95, "B": 0.180, "P": 2.00},
             "U3": {"A": 3.88, "B": 0.0963, "P": 2.00},
             "U4": {"A": 5.67, "B": 0.352, "P": 2.00},
             "U5": {"A": 0.00342, "B": 0.00262, "P": 0.02},
             "C1": {"A": 0.14, "B": 0, "P": 0.02},
             "C2": {"A": 13.5, "B": 0, "P": 2.00},
             "C3": {"A": 80.0, "B": 0, "P": 2.00},
             "C4": {"A": 120.0, "B": 0, "P": 2.00},
             "C5": {"A": 0.05, "B": 0, "P": 0.04}}
    # Load Constants
    A = const[curve]["A"]
    B = const[curve]["B"]
    P = const[curve]["P"]
    if tfixed is None:
        # Evaluate in seconds from cycles
        CTI = CTI / freq
        # Evaluate M
        M = I / (CTR_dn * Ipu_dn)
        # Evaluate Trip Time
        tpu_desired = TDdn * (A / (M ** P - 1) + B) + CTI
    else:
        tpu_desired = tfixed + CTI
    # Re-Evaluate M
    M = I / (CTR_up * Ipu_up)
    # Calculate TD setting (use curve exponent P, not hard-coded 2)
    TD = tpu_desired / (A / (M ** P - 1) + B)
    # Scale and Round
    TD = _np.floor(TD * 10 ** scale) / 10 ** scale
    return TD


# Define TAP Calculator
def protectiontap(S, CTR=1, VLN=None, VLL=None):
    """
    Protection TAP Setting Calculator.
    """
    # Condition Voltage(s)
    if VLL is not None:
        V = abs(_np.sqrt(3) * VLL)
    elif VLN is not None:
        V = abs(3 * VLN)
    else:
        raise ValueError("One or more voltages must be provided.")
    # Calculate TAP
    TAP = abs(S) / (V * CTR)
    return TAP


# Define Current Correction Calculator
def correctedcurrents(Ipri, TAP, correction="Y", CTR=1):
    """
    Electrical Transformer Current Correction Function.
    """
    # Define Matrix Lookup
    MAT = {"Y": XFMY0,
           "D+": XFMD1,
           "D-": XFMD11,
           "Z": XFM12}
    # Condition Inputs
    Ipri = _np.asarray(Ipri)
    if isinstance(correction, list):
        mult = MAT[correction[0]]
        for i in correction[1:]:
            mult = mult.dot(MAT[i])
    elif isinstance(correction, str):
        mult = MAT[correction]
    elif isinstance(correction, _np.ndarray):
        mult = correction
    else:
        raise ValueError("Correction must be string or list of strings.")
    # Evaluate Corrected Current
    Isec_corr = 1 / TAP * mult.dot(Ipri / CTR)
    return Isec_corr


# Define Iop/Irt Calculator
def iopirt(IpriHV, IpriLV, TAPHV, TAPLV, corrHV="Y", corrLV="Y", CTRHV=1,
           CTRLV=1):
    """
    Operate/Restraint Current Calculator.
    """
    # Calculate Corrected Currents
    IcorHV = correctedcurrents(IpriHV, TAPHV, corrHV, CTRHV)
    IcorLV = correctedcurrents(IpriLV, TAPLV, corrLV, CTRLV)
    # Calculate Operate/Restraint Currents
    Iop = _np.absolute(IcorHV + IcorLV)
    Irt = _np.absolute(IcorHV) + _np.absolute(IcorLV)
    # Calculate Slopes
    slope = Iop / Irt
    return Iop, Irt, slope


# Define Symmetrical/RMS Current Calculation
def symrmsfaultcur(V, R, X, t=1 / 60, freq=60):
    """
    Symmetrical/RMS Current Calculator.
    """
    # Calculate Z and tau
    Z = _np.sqrt(R ** 2 + X ** 2)
    tau = X / (2 * _np.pi * freq * R)
    # Calculate Symmetrical Fault Current
    Isym = (V / _np.sqrt(3)) / Z
    # Calculate RMS Fault Current
    Irms = _np.sqrt(1 + 2 * _np.exp(-2 * t / tau)) * Isym
    return tau, Isym, Irms


# Define Relay M Formula
def faultratio(I, Ipickup, CTR=1):
    """
    Fault Multiple of Pickup (Ratio) Calculator.
    """
    M = I / (CTR * Ipickup)
    return M


# Define Residual Compensation Factor Function
def residcomp(z1, z0, linelength=1):
    """
    Residual Compensation Factor Function.
    """
    # Evaluate Z1L and Z0L
    Z1L = z1 * linelength
    Z0L = z0 * linelength
    # Calculate Residual Compensation Factor (k0)
    k0 = (Z0L - Z1L) / (3 * Z1L)
    return k0


# Define Relay Measured Impedance Functon for Distance Elements
def distmeasz(VLNmeas, If, Ip, Ipp, CTR=1, VTR=1, k0=None, z1=None, z0=None,
              linelength=1):
    """
    Distance Element Measured Impedance Function.
    """
    # Validate Residual Compensation Inputs
    if k0 is None and z1 is None and z0 is None:
        raise ValueError("Residual compensation arguments must be set.")
    if k0 is None and (z1 is None or z0 is None):
        raise ValueError("Both *z1* and *z0* must be specified.")
    # Calculate Residual Compensation if Necessary
    if k0 is None:
        k0 = residcomp(z1, z0, linelength)
    # Convert Primary Units to Secondary
    V = VLNmeas / VTR
    Ir = (If + Ip + Ipp) / CTR
    I = If / CTR
    # Calculate Measured Impedance
    Zmeas = V / (I + k0 * Ir)
    return Zmeas


# Define Transformer Tap Mismatch Function
def transmismatch(I1, I2, tap1, tap2):
    """
    Electrical Transformer TAP Mismatch Function.
    """
    # Evaluate MR
    MR = min(abs(I1 / I2), abs(tap1 / tap2))
    # Calculate Mismatch
    mismatch = (abs(I1 / I2) - abs(tap1 / tap2)) * 100 / MR
    return mismatch


# Define High-Impedance Bus Protection Pickup Function
def highzvpickup(I, RL, Rct, CTR=1, threephase=False, Ks=1.5,
                 Vstd=400, Kd=0.5):
    """
    High Impedance Pickup Setting Function.
    """
    # Condition Based on threephase Argument
    n = 2
    if threephase: n = 1
    # Evaluate Secure Voltage Pickup
    Vsens = Ks * (n * RL + Rct) * I / CTR
    # Evaluate Dependible Voltage Pickup
    Vdep = Kd * Vstd
    return Vsens, Vdep


# Define Minimum Current Pickup for High-Impedance Bus Protection
def highzmini(N, Ie, Irly=None, Vset=None, Rrly=2000, Imov=0, CTR=1):
    """
    Minimum Current for High Impedance Protection Calculator.
    """
    # Validate Inputs
    if Irly is None and Vset is None:
        raise ValueError("Relay Current Required.")
    # Condition Inputs
    Ie = abs(Ie)
    Imov = abs(Imov)
    if Irly is None:
        Vset = abs(Vset)
        Irly = Vset / Rrly
    else:
        Irly = abs(Irly)
    # Evaluate Minimum Current Pickup
    Imin = (N * Ie + Irly + Imov) * CTR
    return (Imin)


# Define Instantaneous Overcurrent Pickup Formula
def instoc(Imin, CTR=1, Ki=0.5):
    """
    Instantaneous OverCurrent Pickup Calculator.
    """
    # Evaluate Overcurrent Pickup Setting
    Ipu = Ki * abs(Imin) / CTR
    return Ipu


# Define Generator Loss of Field Element Function
def genlossfield(Xd, Xpd, Zbase=1, CTR=1, VTR=1):
    """
    Electric Generator Loss of Field Function.
    """
    # Condition Inputs
    Xd = abs(Xd)
    Xpd = abs(Xpd)
    Zbase = abs(Zbase)
    # Evaluate Xd_sec and Xpd_sec
    Xd_sec = Xd * Zbase * (CTR / VTR)
    Xpd_sec = Xpd * Zbase * (CTR / VTR)
    # Determine Zone Offset
    ZoneOff = Xpd_sec / 2
    # Evaluate Z1 Diameter and Z2 Diameter
    Z1dia = Zbase * CTR / VTR
    Z2dia = Xd_sec
    # Return
    return ZoneOff, Z1dia, Z2dia


# Define Thermal Time Limit Calculator
def thermaltime(In, Ibase, tbase):
    r"""
    Thermal Time Limit Calculator.
    """
    # Perform Calculation
    tn = (Ibase ** 2 * tbase) / (In ** 2)
    return tn


# Define Synch. Machine Fault Current Calculator
def synmach_Isym(t, Eq, Xd, Xdp, Xdpp, Tdp, Tdpp):
    r"""
    Synch. Machine Symmetrical Fault Current Calc.
    """
    # Calculate Time-Constant Term
    t_c = (
        1 / Xd + 
        (1 / Xdp - 1 / Xd) * _np.exp(-t / Tdp) + 
        (1 / Xdpp - 1 / Xdp) * _np.exp(-t / Tdpp)
    )
    # Calculate Fault Current
    Ia = _np.sqrt(2) * abs(Eq) * t_c
    return Ia


# Define Synch. Machine Asymmetrical Current Calculator
def synmach_Iasym(t, Eq, Xdpp, Xqpp, Ta):
    r"""
    Synch. Machine Asymmetrical Fault Current Calc.
    """
    # Calculate Time Constant Term
    t_c = 1 / Xdpp + 1 / Xqpp
    # Calculate Asymmetrical Current
    Iasym = _np.sqrt(2) * abs(Eq) * 1 / 2 * t_c * _np.exp(-t / Ta)
    return Iasym


# Define Induction Machine Eigenvalue Calculator
def indmacheigenvalues(Lr, Ls, Lm, Rr, Rs, wrf=0, freq=60):
    """
    Induction Machine Eigenvalue Calculator.
    """
    # Calculate Required Values
    omega_e_base = 2 * _np.pi * freq
    omega_rf = wrf
    torque_s = Ls / (omega_e_base * Rs)
    torque_r = Lr / (omega_e_base * Rr)
    alpha = torque_r / torque_s
    phi = 1 - Lm ** 2 / (Ls * Lr)
    omega_r = omega_e_base
    # Calculate k1
    k1 = -1 / (2 * phi * torque_r) * (1 + alpha)
    k1 += 1j * (omega_r / 2 - omega_rf)
    # Calculate k2
    k2 = 1 / (2 * phi * torque_r)
    k2 *= _np.sqrt(
        (1 + alpha) ** 2 - 4 * phi * alpha - (omega_r * phi * torque_r) ** 2 +
        2j * (alpha - 1) * omega_r * phi * torque_r
    )
    # Evaluate Eigenvalues and Return
    lam1 = k1 + k2
    lam2 = k1 - k2
    return lam1, lam2


# Define IM 3-Phase SC Current Calculator
def indmachphs3sc(t, Is0, Lr, Ls, Lm, Rr, Rs, wrf=0, freq=60, real=True):
    """
    Induction Machine 3-Phase SC Calculator.
    """
    # Calculate Required Values
    omega_r = 2 * _np.pi * freq
    torque_s = Ls / (omega_r * Rs)
    phi = 1 - Lm ** 2 / (Ls * Lr)
    # Calculate Eigenvalues
    lam1, lam2 = indmacheigenvalues(Lr, Ls, Lm, Rr, Rs, wrf, freq)
    # Calculate pIs0
    pIs0 = -(1 / (phi * torque_s) + 1j * (1 - phi) / phi * omega_r) * Is0
    # Calculate Constants
    C1 = (lam2 * Is0 - pIs0) / (lam2 - lam1)
    C2 = (pIs0 - lam1 * Is0) / (lam2 - lam1)
    # Calculate ias and Return
    ias = C1 * _np.exp(lam1 * t) + C2 * _np.exp(lam2 * t)
    if real:
        ias = _np.real(ias)
    return ias


# Define IM Torque Calculation
def indmachphs3torq(t, Is0, Lr, Ls, Lm, Rr, Rs, wrf=0, freq=60):
    """
    Induction Machine 3-Phase Torque Calculator.
    """
    # Calculate Required Values
    omega_r = 2 * _np.pi * freq
    torque_s = Ls / (omega_r * Rs)
    phi = 1 - Lm ** 2 / (Ls * Lr)
    # Calculate Eigenvalues
    lam1, lam2 = indmacheigenvalues(Lr, Ls, Lm, Rr, Rs, wrf, freq)
    # Calculate pIs0
    pIs0 = -(1 / (phi * torque_s) + 1j * (1 - phi) / phi * omega_r) * Is0
    # Calculate Constants
    C1 = (lam2 * Is0 - pIs0) / (lam2 - lam1)
    C2 = (pIs0 - lam1 * Is0) / (lam2 - lam1)
    # Calculate ias and Return
    idqs = C1 * _np.exp(lam1 * t) + C2 * _np.exp(lam2 * t)
    idqr = C2 * _np.exp(lam1 * t) + C1 * _np.exp(lam2 * t)
    # Calculate Lambda
    lamdqr = Lm * idqs + Lr * idqr
    # Calculate Torque
    Tem = Lm / Lr * (lamdqr.real * idqs.imag - lamdqr.imag * idqs.real)
    return Tem


# Define Complete Sync. Mach. Fault Current Function
def synmach_ifault(t, Ea, alpha, Xd, Xdp, Xdpp, Xqpp, Tdp, Tdpp, Ta, freq=60):
    # noqa: D401   "Synchronous" is intentional descriptor
    """
    Synchronous Machine Fault Current Calculator.
    """
    # Calculate we Component
    we = 2 * _np.pi * freq
    # Condition Inputs
    Ea = abs(Ea)
    alpha = _np.radians(alpha)
    # Define Constant Term
    const = _np.sqrt(2) * Ea

    # Symmetrical AC component magnitude term
    isym_mag = (
        (1 / Xd) +
        (1 / Xdp - 1 / Xd) * _np.exp(-t / Tdp) +
        (1 / Xdpp - 1 / Xdp) * _np.exp(-t / Tdpp)
    )

    # DC offset / asymmetry magnitude term
    if Xqpp != 0:
        val = 1 / Xqpp
    else:
        val = 0
    asym_mag = 1 / 2 * (1 / Xdpp + val) * _np.exp(-t / Ta)

    # Define Symmetrical Portion
    isym = const * isym_mag * _np.sin(we * t + alpha)

    # Define Asymmetrical (DC offset) Portion
    iasym = const * asym_mag * _np.sin(alpha)

    # Define Double Frequency Term
    idbl = const * 1 / 2 * asym_mag * _np.sin(2 * we * t + alpha)

    # Compose Complete Current Value
    ias = isym - iasym - idbl
    return ias

# END
