################################################################################
"""
`electricpy.fault` - Electrical Power Engineering Faults Calculations.

>>> from electricpy import fault
"""
################################################################################

# Import Necessary Dependencies
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import fsolve as _fsolve

# Import Local Dependencies
from .constants import *

def _phaseroll(M012,reference):
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
    return(M)

# Define Single Line to Ground Fault Function
def phs1g(Vth,Zseq,Rf=0,sequence=True,reference='A'):
    r"""
    Single-Phase-to-Ground Fault Calculator.
    
    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a single-line-to-ground fault.
    
    .. math:: I_1 = \frac{V_{th}}{Z_0+Z_1+Z_2+3*R_f}
    
    .. math:: I_2 = I_1
    
    .. math:: I_0 = I_1
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    Rf:         complex, optional
                The fault resistance, default=0
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    
    Returns
    -------
    Ifault:     list of complex,
                The Array of Fault Currents as (If0, If1, If2)
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X0, complex)): X0 *= 1j
    if(not isinstance(X1, complex)): X1 *= 1j
    if(not isinstance(X2, complex)): X2 *= 1j
    # Calculate Fault Current
    Ifault = Vth / (X0 + X1 + X2 + 3*Rf)
    Ifault = _np.array([ Ifault, Ifault, Ifault ])
    # Prepare Value for return
    if not sequence:
        Ifault = _phaseroll( Ifault, reference ) # Convert to ABC-Domain
    # Return Value
    return(Ifault)
    
# Define Double Line to Ground Fault Current Calculator
def phs2g(Vth,Zseq,Rf=0,sequence=True,reference='A'):
    r"""
    Double-Line-to-Ground Fault Calculator.
    
    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a double-line-to-ground fault.
    
    .. math:: I_1 = \frac{V_{th}}{Z_1+\frac{Z_2*(Z_0+3*R_f)}{Z_0+Z_2+3*R_f}}
    
    .. math:: I_2 = -\frac{V_{th}-Z_1*I_1}{X_2}
    
    .. math:: I_0 = -\frac{V_{th}-Z_1*I_1}{X_0+3*R_f}
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    Rf:         complex, optional
                The fault resistance, default=0
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    
    Returns
    -------
    Ifault:     list of complex,
                The Array of Fault Currents as (If0, If1, If2)
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X0, complex)): X0 *= 1j
    if(not isinstance(X1, complex)): X1 *= 1j
    if(not isinstance(X2, complex)): X2 *= 1j
    # Calculate Fault Currents
    If1 = Vth / (X1 + (X2*(X0+3*Rf))/(X0+X2+3*Rf))
    If2 = -(Vth - X1*If1)/X2
    If0 = -(Vth - X1*If1)/(X0+3*Rf)
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll( Ifault, reference ) # Convert to ABC-Domain
    return(Ifault)

# Define Phase-to-Phase Fault Current Calculator
def phs2(Vth,Zseq,Rf=0,sequence=True,reference='A'):
    r"""
    Line-to-Line Fault Calculator.
    
    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a phase-to-phase fault.
    
    .. math:: I_1 = \frac{V_{th}}{Z_1+Z_2+R_f}
    
    .. math:: I_2 = -I_1
    
    .. math:: I_0 = 0
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    Rf:         complex, optional
                The fault resistance, default=0
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    
    Returns
    -------
    Ifault:     list of complex,
                The Array of Fault Currents as (If0, If1, If2)
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X0, complex)): X0 *= 1j
    if(not isinstance(X1, complex)): X1 *= 1j
    if(not isinstance(X2, complex)): X2 *= 1j
    # Calculate Fault Currents
    If0 = 0
    If1 = Vth / (X1 + X2 + Rf)
    If2 = -If1
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll( Ifault, reference ) # Convert to ABC-Domain
    return(Ifault)

# Define Three-Phase Fault Current Calculator
def phs3(Vth,Zseq,Rf=0,sequence=True,reference='A'):
    r"""
    Three-Phase Fault Calculator.
    
    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a three-phase fault.
    
    .. math:: I_1 = \frac{V_{th}}{Z_1+R_1}
    
    .. math:: I_2 = 0
    
    .. math:: I_0 = 0
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    Rf:         complex, optional
                The fault resistance, default=0
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    
    Returns
    -------
    Ifault:     list of complex
                The Fault Current, equal for 0, pos., and neg. seq.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X1, complex)): X1 *= 1j
    # Calculate Fault Currents
    Ifault = Vth/(X1 + Rf)
    Ifault = _np.array([ 0, Ifault, 0 ])
    # Prepare to Return Value
    if not sequence:
        Ifault = _phaseroll( Ifault, reference ) # Convert to ABC-Domain
    return(Ifault)

# Define Single Pole Open Calculator
def poleopen1(Vth,Zseq,sequence=True,reference='A'):
    r"""
    Single Pole Open Fault Calculator.
    
    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a single pole open fault.
    
    .. math:: I_1 = \frac{V_{th}}{Z_1+(\frac{1}{Z_2}+\frac{1}{Z_0})^-1}
    
    .. math:: I_2 = -I_1 * \frac{Z_0}{Z_2+Z_0}
    
    .. math:: I_0 = -I_1 * \frac{Z_2}{Z_2+Z_0}
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference, or the
                faulted phase indicator; default='A'
    
    Returns
    -------
    Ifault:     list of complex,
                The Array of Fault Currents as (If0, If1, If2)
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X0, complex)): X0 *= 1j
    if(not isinstance(X1, complex)): X1 *= 1j
    if(not isinstance(X2, complex)): X2 *= 1j
    # Calculate Fault Currents
    If1 = Vth / (X1 + (1/X2 + 1/X0)**(-1))
    If2 = -If1 * X0/(X2 + X0)
    If0 = -If1 * X2/(X2 + X0)
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll( Ifault, reference ) # Convert to ABC-Domain
    return(Ifault)

# Define Double Pole Open Calculator
def poleopen2(Vth,Zseq,sequence=True,reference='A'):
    r"""
    Single Pole Open Fault Calculator.
    
    This function will evaluate the Zero, Positive, and Negative
    sequence currents for a single pole open fault.
    
    .. math:: I_1 = \frac{V_{th}}{Z_1+Z_2+Z_0}
    
    .. math:: I_2 = I_1
    
    .. math:: I_0 = I_1
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference, or the
                faulted phase indicator; default='A'
    
    Returns
    -------
    Ifault:     list of complex,
                The Array of Fault Currents as (If0, If1, If2)
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X0, complex)): X0 *= 1j
    if(not isinstance(X1, complex)): X1 *= 1j
    if(not isinstance(X2, complex)): X2 *= 1j
    # Calculate Fault Currents
    If1 = Vth / (X1 + X2 + X0)
    If2 = If1
    If0 = If1
    Ifault = _np.array([If0, If1, If2])
    # Return Currents
    if not sequence:
        Ifault = _phaseroll( Ifault, reference ) # Convert to ABC-Domain
    return(Ifault)

# Define MVA Short Circuit
def scMVA(Zth=None,Isc=None,Vth=1):
    r"""
    Short-Circuit MVA Calculator.
    
    Function defines a method of interpretively
    calculating the short-circuit MVA value
    given two of the three arguments. The formulas
    are all based around the following:
    
    .. math:: MVA_{sc} = V_{th}*I_{sc}
    
    .. math:: V_{th} = I_{sc}*Z_{th}
    
    Parameters
    ----------
    Zth:        float
                The Thevenin-Equivalent-Impedance
    Isc:        float, optional
                Short-Circuit-Current, if left as
                None, will force function to use
                default setting for Vth.
                default=None
    Vth:        float, optional
                The Thevenin-Equivalent-Voltage,
                defaults to a 1-per-unit value.
                default=1
    
    Returns
    -------
    MVA:        float
                Short-Circuit MVA, not described
                as three-phase or otherwise, such
                determination is dependent upon
                inputs.
    """
    # Test for too few inputs
    if not any((Zth,Isc)):
        raise ValueError("Either Zth or Isc must be specified.")
    # Condition Inputs
    if Zth is not None:
        Zth = abs(Zth)
    if Isc is not None:
        Isc = abs(Isc)
    if Vth != 1:
        Vth = abs(Vth)
    # Calculate MVA from one of the available methods
    if all((Zth,Isc)):
        MVA = Isc**2 * Zth
    elif all((Zth,Vth)):
        MVA = Vth**2 / Zth
    else:
        MVA = Vth * Isc
    # Return Value
    return(MVA)

# Define Explicitly 3-Phase MVAsc Calculator
def phs3mvasc(Vth,Zseq,Rf=0,Sbase=1):
    r"""
    Three-Phase MVA Short-Circuit Calculator.
    
    Calculator to evaluate the Short-Circuit MVA of a three-phase fault given the system
    parameters of Vth, Zseq, and an optional Rf. Uses the formula as follows:
    
    .. math:: MVA_{sc} = \frac{\left|V_{th}^2\right|}{|Z_1|} * Sbase
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    Rf:         complex, optional
                The fault resistance, default=0
    Sbase:      real, optional
                The per-unit base for power. default=1
    
    Returns
    -------
    MVA:        real
                Three-Phase Short-Circuit MVA.
    """
    # Calculate Three-Phase MVA
    MVA = abs(Vth)**2 / abs(Zseq[1]) * Sbase
    # Scale VA to MVA if Sbase is not 1
    if Sbase != 1:
        MVA = MVA * 1e-6 # Divide by 1e6 (M)
    # Return
    return(MVA)
    
    
# Define Explicitly 1-Phase MVAsc Calculator
def phs1mvasc(Vth,Zseq,Rf=0,Sbase=1):
    r"""
    Single-Phase MVA Short-Circuit Calculator.
    
    Calculator to evaluate the Short-Circuit MVA of a single-phase fault given the system
    parameters of Vth, Zseq, and an optional Rf. Uses the formula as follows:
    
    .. math:: MVA_{sc} = \left|I_1^2\right|*|Z_1| * Sbase
    
    where:
    
    .. math:: I_1 = \frac{V_{th}}{Z_0+Z_1+Z_2+3*R_f}
    
    Parameters
    ----------
    Vth:        complex
                The Thevenin-Equivalent-Voltage
    Zseq:       list of complex
                Tupple of sequence reactances as (Z0, Z1, Z2)
    Rf:         complex, optional
                The fault resistance, default=0
    Sbase:      real, optional
                The per-unit base for power. default=1
    
    Returns
    -------
    MVA:        real
                Single-Phase Short-Circuit MVA.
    """
    # Decompose Reactance Tuple
    X0, X1, X2 = Zseq
    # Ensure that X-components are imaginary
    if(not isinstance(X0, complex)): X0 *= 1j
    if(not isinstance(X1, complex)): X1 *= 1j
    if(not isinstance(X2, complex)): X2 *= 1j
    # Calculate Fault Current
    Ifault = Vth / (X0 + X1 + X2 + 3*Rf)
    # Calculate MVA
    MVA = abs(Ifault)**2 * abs(X1) * Sbase
    # Scale VA to MVA if Sbase is not 1
    if Sbase != 1:
        MVA = MVA * 1e-6 # Divide by 1e6 (M)
    # Return
    return(MVA)

# Define Faulted Bus Voltage Calculator
def busvolt(k,n,Vpf,Z0,Z1,Z2,If,sequence=True,reference='A'):
    """
    Faulted Bus Voltage Calculator.
    
    This function is designed to calculate the bus voltage(s)
    given a specific set of fault characteristics.
    
    Parameters
    ----------
    k:          float
                Bus index at which to calculate faulted voltage
    n:          float
                Bus index at which fault occurred
    Vpf:        complex
                Voltage Pre-Fault, Singular Number
    Z0:         ndarray
                Zero-Sequence Impedance Matrix
    Z1:         ndarray
                Positive-Sequence Impedance Matrix
    Z2:         ndarray
                Negative-Sequence Impedance Matrix
    If:         complex
                Sequence Fault Current Evaluated at Bus *n*
    sequence:   bool, optional
                Control argument to force return into symmetrical-
                or phase-domain values.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
    
    Returns
    -------
    Vf:         complex
                The Fault Voltage, set of sequence or phase voltages as
                specified by *sequence*
    """
    # Condition Inputs
    k = k-1
    n = n-1
    Z0 = _np.asarray(Z0)
    Z1 = _np.asarray(Z1)
    Z2 = _np.asarray(Z2)
    If = _np.asarray(If)
    # Generate Arrays For Calculation
    Vfmat = _np.array([0, Vpf, 0]).T
    Zmat = _np.array([[Z0[k,n], 0, 0],
                     [0, Z1[k,n], 0],
                     [0, 0, Z2[k,n]]])
    # Perform Calculation
    Vf = Vfmat - Zmat.dot(If)
    if not sequence:
        Vf = _phaseroll( Vf, reference ) # Convert to ABC-Domain
    return(Vf)


# Define CT Saturation Function
def ct_saturation(XoR,Imag,Vrated,Irated,CTR,Rb,Xb,remnance=0,freq=60,ALF=20):
    r"""
    Electrical Current Transformer Saturation Calculator.
    
    A function to determine the saturation value and a boolean indicator
    showing whether or not CT is -in fact- saturated.
    
    To perform this evaluation, we must satisfy the equation:
    
    .. math::
       20\geq(1+\frac{X}{R})*\frac{|I_{mag}|}{I_{rated}*CTR}
       *\frac{\left|R_{burden}+j*\omega*\frac{X_{burden}}
       {\omega}\right|*100}{V_{rated}*(1-remnanc)}
    
    Parameters
    ----------
    XoR:        float
                The X-over-R ratio of the system.
    Imag:       float
                The (maximum) current magnitude to use for calculation,
                typically the fault current.
    Vrated:     float
                The rated voltage (accompanying the C-Class value) of
                the CT.
    Irated:     float
                The rated secondary current for the CT.
    CTR:        float
                The CT Ratio (primary/secondary, N) to be used.
    Rb:         float
                The total burden resistance in ohms.
    Xb:         float
                The total burden reactance in ohms.
    remnance:   float, optional
                The system flux remnance, default=0.
    freq:       float, optional
                The system frequency in Hz, default=60.
    ALF:        float, optional
                The Saturation Constant which must be satisfied,
                default=20.
    
    Returns
    -------
    result:     float
                The calculated Saturation value.
    saturation: bool
                Boolean indicator to mark presence of saturation.
    """
    # Define omega
    w = 2*_np.pi*freq
    # Find Lb
    Lb = Xb/w
    # Re-evaluate Vrated
    Vrated = Vrated*(1-remnance)
    # Calculate each "term" (multiple)
    t1 = (1+XoR)
    t2 = (Imag/(Irated*CTR))
    t3 = abs(Rb+1j*w*Lb)*100/Vrated
    # Evaluate
    result = t1*t2*t3
    # Test for saturation
    saturation = result >= ALF
    # Return Results
    return(result,saturation)


# Define C-Class Calculator
def ct_cclass(XoR,Imag,Irated,CTR,Rb,Xb,remnance=0,sat_crit=20):
    r"""
    Electrical Current Transformer (CT) C-Class Function.
    
    A function to determine the C-Class rated voltage for a CT.
    The formula shown below demonstrates the standard formula
    which is normally used to evaluate the saturation criteria.
    Worth noting here, is the fact that :math:`V_{rated}` is the
    CT C-Class.
    
    .. math::
       \text{Saturation Criteria}=\frac{(1+\frac{X}{R})\cdot
       \frac{|I_{mag}|}{I_{rated}\cdot CTR}\cdot\frac{\left|
       R_{burden}+j\cdot X_{burden}\right|\cdot100}{V_{rated}}}
       {1-remnance}
    
    For the purposes of this function, the above formula is applied
    as follows to evaluate the CT C-Class such as to satisfy the
    saturation criteria defined.
    
    .. math::
       \text{CT C-Class}=\frac{(1+\frac{X}{R})\cdot
       \frac{|I_{mag}|}{I_{rated}\cdot CTR}\cdot\frac{
       \left|R_{burden}+j\cdot X_{burden}\right|\cdot100}
       {\text{Saturation Criteria (i.e., 20)}}}{1-remnance}
    
    Parameters
    ----------
    XoR:        float
                The X-over-R ratio of the system.
    Imag:       float
                The (maximum) current magnitude to use for calculation,
                typically the fault current.
    Irated:     float
                The rated secondary current for the CT.
    CTR:        float
                The CT Ratio (primary/secondary, N) to be used.
    Rb:         float
                The total burden resistance in ohms.
    Xb:         float
                The total burden reactance in ohms.
    remnance:   float, optional
                The system flux remnance, default=0.
    sat_crit:   float, optional
                The saturation criteria which must be satisfied,
                typically such that CT saturation will not occur,
                default=20.
    
    Returns
    -------
    c_class:    float
                The calculated C-Class rated voltage.
    """
    # Calculate each "term" (multiple)
    t1 = (1+XoR)
    t2 = (Imag/(Irated*CTR))
    t3 = abs(Rb+1j*Xb)*100/sat_crit
    # Evaluate
    Vr_w_rem = t1*t2*t3
    c_class = Vr_w_rem/(1-remnance)
    # Return Result
    return(c_class)


# Define Saturation Voltage at Rated Burden
def ct_satratburden(Inom,VArat=None,ANSIv=None,ALF=20,):
    r"""
    Electrical Current Transformer (CT) Saturation at Rated Burden Calculator.
    
    A function to determine the Saturation at rated burden.
    
    .. math:: V_{saturated}=ALF*\frac{VA_{rated}}{I_{nominal}}
    
    where:
    
    .. math:: VA_{rated}=I_{nominal}*\frac{ANSI_{voltage}}{20}
    
    Parameters
    ----------
    Inom:       float
                Nominal Current
    VArat:      float, optional, exclusive
                The apparent power (VA) rating of the CT.
    ANSIv:      float, optional, exclusive
                The ANSI voltage requirement to meet.
    ALF:        float, optional
                Accuracy Limit Factor, default=20.
    
    Returns
    -------
    Vsat:       float
                The saturated voltage.
    """
    # Validate Inputs
    if VArat == None and ANSIv == None:
        raise ValueError("VArat or ANSIv must be specified.")
    elif VArat==None:
        # Calculate VArat from ANSIv
        VArat = Inom*ANSIv/(20)
    # Determine Vsaturation
    Vsat = ALF * VArat/Inom
    return(Vsat)


# Define CT Vpeak Formula
def ct_vpeak(Zb,Ip,CTR):
    r"""
    Electrical Current Transformer (CT) Peak Voltage Calculator.
    
    Simple formula to calculate the Peak Voltage of a CT.
    
    .. math:: \sqrt{3.5*|Z_burden|*I_{peak}*CTR}
    
    Parameters
    ----------
    Zb:         float
                The burden impedance magnitude (in ohms).
    Ip:         float
                The peak current for the CT.
    CTR:        float
                The CTR turns ratio of the CT.
    
    Returns
    -------
    Vpeak:      float
                The peak voltage.
    """
    return(_np.sqrt(3.5*abs(Zb)*Ip*CTR))


# Define Saturation Time Calculator
def ct_timetosat(Vknee,XoR,Rb,CTR,Imax,ts=None,npts=100,freq=60,plot=False):
    r"""
    Electrical Current Transformer (CT) Time to Saturation Function.
    
    Function to determine the "time to saturate" for an underrated C-Class
    CT using three standard curves described by Juergen Holbach.
    
    Parameters
    ----------
    Vknee:      float
                The knee-voltage for the CT.
    XoR:        float
                The X-over-R ratio of the system.
    Rb:         float
                The total burden resistance in ohms.
    CTR:        float
                The CT Ratio (primary/secondary, N) to be used.
    Imax:       float
                The (maximum) current magnitude to use for calculation,
                typically the fault current.
    ts:         numpy.ndarray or float, optional
                The time-array or particular (floatint point) time at which
                to calculate the values. default=_np.linspace(0,0.1,freq*npts)
    npts:       float, optional
                The number of points (per cycle) to calculate if ts is not
                specified, default=100.
    freq:       float, optional
                The system frequency in Hz, default=60.
    plot:       bool, optional
                Control argument to enable plotting of calculated curves,
                default=False.
    """
    # Calculate omega
    w = 2*_np.pi*freq
    # Calculate Tp
    Tp = XoR/w
    # If ts isn't specified, generate it
    if ts==None:
        ts = _np.linspace(0,0.1,freq*npts)
    # Calculate inner term
    term = -XoR*(_np.exp(-ts/Tp)-1)
    # Calculate Vsaturation terms
    Vsat1 = Imax*Rb*(term+1)
    Vsat2 = Imax*Rb*(term-_np.sin(w*ts))
    Vsat3 = Imax*Rb*(1-_np.cos(w*ts))
    # If plotting requested
    if plot and isinstance(ts,_np.ndarray):
        _plt.plot(ts,Vsat1,label="Vsat1")
        _plt.plot(ts,Vsat2,label="Vsat2")
        _plt.plot(ts,Vsat3,label="Vsat3")
        _plt.axhline(Vknee,label="V-knee",linestyle='--')
        _plt.title("Saturation Curves")
        _plt.xlabel("Time (ts)")
        _plt.legend()
        _plt.show()
    elif plot:
        print("Unable to plot a single point, *ts* must be a numpy-array.")
    # Determine the crossover points for each saturation curve
    Vsat1c = Vsat2c = Vsat3c = 0
    if isinstance(ts,_np.ndarray):
        for i in range(len(ts)):
            if Vsat1[i]>Vknee and Vsat1c==0:
                Vsat1c = ts[i-1]
            if Vsat2[i]>Vknee and Vsat2c==0:
                Vsat2c = ts[i-1]
            if Vsat3[i]>Vknee and Vsat3c==0:
                Vsat3c = ts[i-1]
        results = (Vsat1c,Vsat2c,Vsat3c)
    else:
        results = (Vsat1,Vsat2,Vsat3)
    return(results)

# Define Function to Calculate TRV
def pktransrecvolt(C,L,R=0,VLL=None,VLN=None,freq=60):
    """
    Peak Transient Recovery Function.
    
    Peak Transient Recovery Voltage calculation function, evaluates the peak
    transient recovery voltage (restriking voltage) and the
    Rate-of-Rise-Recovery Voltage.
    
    Parameters
    ----------
    C:          float
                Capacitance Value in Farads.
    L:          float
                Inductance in Henries.
    R:          float, optional
                The resistance of the system used for
                calculation, default=0.
    VLL:        float, exclusive
                Line-to-Line voltage, exclusive
                optional argument.
    VLN:        float, exclusive
                Line-to-Neutral voltage, exclusive
                optional argument.
    freq:       float, optional
                System frequency in Hz.
    
    Returns
    -------
    Vcpk:       float
                Peak Transient Recovery Voltage in volts.
    RRRV:       float
                The RRRV (Rate-of-Rise-Recovery Voltage)
                calculated given the parameters in volts
                per second.
    """
    # Evaluate alpha, omega-n, and fn
    alpha = R/(2*L)
    wn = 1/_np.sqrt(L*C) - alpha
    fn = wn/(2*_np.pi)
    # Evaluate Vm
    if VLL!=None:
        Vm = _np.sqrt(2/3)*VLL
    elif VLN!=None:
        Vm = _np.sqrt(2)*VLN
    else:
        raise ValueError("One voltage must be specified.")
    # Evaluate Vcpk (worst case)
    Vcpk = wn**2/(wn**2-2*_np.pi*freq)*Vm*2
    # Evaluate RRRV
    RRRV = 2*Vm*fn/0.5
    return(Vcpk,RRRV)

# Define TRV Reduction Resistor Function
def trvresistor(C,L,reduction,Rd0=500,wd0=260e3,tpk0=10e-6):
    """
    Transient Recovery Voltage (TRV) Reduction Resistor Function.
    
    Function to find the resistor value that will reduce the TRV by a specified
    percentage.
    
    Parameters
    ----------
    C:          float
                Capacitance Value in Farads.
    L:          float
                Inductance in Henries.
    reduction:  float
                The percentage that the TRV should be reduced by.
    Rd0:        float, optional
                Damping Resistor Evaluation Starting Point, default=500
    wd0:        float, optional
                Omega-d evaluation starting point, default=260*k
    tpk0:       float, optional
                Time of peak voltage evaluation starting point, default=10*u
    
    Returns
    -------
    Rd:         float
                Damping resistor value, in ohms.
    wd:         float
                Omega-d
    tpk:        float
                Time of peak voltage.
    """
    # Evaluate omega-n
    wn = 1/_np.sqrt(L*C)
    # Generate Constant Factor
    fctr = (1-reduction)*2 - 1
    # Define Function Set
    def equations(data):
        Rd, wd, tpk = data
        X = _np.sqrt(wn**2-(1/(2*Rd*C))**2) - wd
        Y = _np.exp(-tpk/(2*Rd*C))-fctr
        Z = wd*tpk - _np.pi
        return(X,Y,Z)
    Rd, wd, tpk = _fsolve(equations, (Rd0,wd0,tpk0))
    return(Rd, wd, tpk)

# Define Time-Overcurrent Trip Time Function
def toctriptime(I,Ipickup,TD,curve="U1",CTR=1):
    """
    Time OverCurrent Trip Time Function.
    
    Time-OverCurrent Trip Time Calculator, evaluates the time
    to trip for a specific TOC (51) element given the curve
    type, current characteristics and time-dial setting.
    
    Parameters
    ----------
    I:          float
                Measured Current in Amps
    Ipickup:    float
                Fault Current Pickup Setting (in Amps)
    TD:         float
                Time Dial Setting
    curve:      string, optional
                Name of specified TOC curve, may be entry from set:
                {U1,U2,U3,U4,U5,C1,C2,C3,C4,C5}, default=U1
    CTR:        float, optional
                Current Transformer Ratio, default=1
    
    Returns
    -------
    tt:         float
                Time-to-Trip for characterized element.
    """
    # Condition Inputs
    curve = curve.upper()
    # Define Dictionary of Constants
    const = {   "U1" : {"A": 0.0104, "B": 0.2256, "P": 0.02},
                "U2" : {"A": 5.95, "B": 0.180, "P": 2.00},
                "U3" : {"A": 3.88, "B": 0.0963, "P": 2.00},
                "U4" : {"A": 5.67, "B": 0.352, "P": 2.00},
                "U5" : {"A": 0.00342, "B": 0.00262, "P": 0.02},
                "C1" : {"A": 0.14, "B":0, "P": 0.02},
                "C2" : {"A": 13.5, "B":0, "P": 2.00},
                "C3" : {"A": 80.0, "B":0, "P": 2.00},
                "C4" : {"A": 120.0, "B":0, "P": 2.00},
                "C5" : {"A": 0.05, "B":0, "P": 0.04}}
    # Load Constants
    A = const[curve]["A"]
    B = const[curve]["B"]
    P = const[curve]["P"]
    # Evaluate M
    M = I / (CTR * Ipickup)
    # Evaluate Trip Time
    tt = TD * (A/(M**P-1)+B)
    return(tt)

# Define Time Overcurrent Reset Time Function
def tocreset(I,Ipickup,TD,curve="U1",CTR=1):
    """
    Time OverCurrent Reset Time Function.
    
    Function to calculate the time to reset for a TOC
    (Time-OverCurrent, 51) element.
    
    Parameters
    ----------
    I:          float
                Measured Current in Amps
    Ipickup:    float
                Fault Current Pickup Setting (in Amps)
    TD:         float
                Time Dial Setting
    curve:      string, optional
                Name of specified TOC curve, may be entry from set:
                {U1,U2,U3,U4,U5,C1,C2,C3,C4,C5}, default=U1
    CTR:        float, optional
                Current Transformer Ratio, default=1
    
    Returns
    -------
    tr:         float
                Time-to-Reset for characterized element.
    """
    # Condition Inputs
    curve = curve.upper()
    # Define Dictionary of Constants
    C = {   "U1" : 1.08,"U2" : 5.95,"U3" : 3.88,
            "U4" : 5.67,"U5" : 0.323,"C1" : 13.5,
            "C2" : 47.3,"C3" : 80.0,"C4" : 120.0,
            "C5" : 4.85}
    # Evaluate M
    M = I / (CTR * Ipickup)
    # Evaluate Reset Time
    tr = TD * (C[curve]/(1-M**2))
    return(tr)

# Define Pickup Current Calculation
def pickup(Iloadmax,Ifaultmin,scale=0,printout=False,units="A"):
    """
    Electrical Current Pickup Selection Assistant.
    
    Used to assist in evaluating an optimal phase-over-current pickup
    setting. Uses maximum load and minimum fault current to provide
    user assistance.
    
    Parameters
    ----------
    Iloadmax:   float
                The maximum load current in amps.
    Ifaultmin:  float
                The minimum fault current in amps.
    scale:      int, optional
                Control scaling to set number of significant figures.
                default=0
    printout:   boolean, optional
                Control argument to enable printing of intermediate
                stages, default=False.
    units:      string, optional
                String to be appended to any printed output denoting
                the units of which are being printed, default="A"
    
    Returns
    -------
    setpoint:   float
                The evaluated setpoint at which the function suggests
                the phase-over-current pickup setting be placed.
    """
    IL2 = 2*Iloadmax
    IF2 = Ifaultmin/2
    exponent = len(str(IL2).split('.')[0])
    setpoint = _np.ceil(IL2*10**(-exponent+1+scale))*10**(exponent-1-scale)
    if printout:
        print("Range Min:",IL2,units,"\t\tRange Max:",IF2,units)
    if IF2 < setpoint:
        setpoint = IL2
        if IL2 > IF2:
            raise ValueError("Invalid Parameters.")
    if printout:
        print("Current Pickup:",setpoint,units)
    return(setpoint)

# Define Time-Dial Coordination Function
def tdradial(I,CTI,Ipu_up,Ipu_dn=0,TDdn=0,curve="U1",scale=2,freq=60,
                  CTR_up=1,CTR_dn=1,tfixed=None):
    """
    Radial Time Dial Coordination Function.
    
    Function to evaluate the Time-Dial (TD) setting in radial schemes
    where the Coordinating Time Interval (CTI) and the up/downstream
    pickup settings are known along with the TD setting for the
    downstream protection.
    
    Parameters
    ----------
    I:          float
                Measured fault current in Amps, typically set using the
                maximum fault current available.
    CTI:        float
                Coordinating Time Interval in cycles.
    Ipu_up:     float
                Pickup setting for upstream protection,
                specified in amps
    Ipu_dn:     float, optional
                Pickup setting for downstream protection,
                specified in amps, default=0
    TDdn:       float, optional
                Time-Dial setting for downstream protection,
                specified in seconds, default=0
    curve:      string, optional
                Name of specified TOC curve, may be entry from set:
                {U1,U2,U3,U4,U5,C1,C2,C3,C4,C5}, default=U1
    scale:      int, optional
                Scaling value used to evaluate a practical TD
                setting, default=2
    freq:       float, optional
                System operating frequency, default=60
    CTR_up:     float, optional
                Current Transformer Ratio for upstream relay.
                default=1
    CTR_dn:     float, optional
                Current Transformer Ratio for downstream relay.
                default=1
    tfixed:     float, optional
                Used to specify a fixed time delay for coordinated
                protection elements, primarily used for coordinating
                TOC elements (51) with OC elements (50) with a fixed
                tripping time. Overrides downstream TOC arguments
                including *Ipu_dn* and *TDdn*.
    
    Returns
    -------
    TD:         float
                Calculated Time-Dial setting according to radial
                scheme logical analysis.
    """
    # Condition Inputs
    curve = curve.upper()
    # Define Dictionary of Constants
    const = {   "U1" : {"A": 0.0104, "B": 0.2256, "P": 0.02},
                "U2" : {"A": 5.95, "B": 0.180, "P": 2.00},
                "U3" : {"A": 3.88, "B": 0.0963, "P": 2.00},
                "U4" : {"A": 5.67, "B": 0.352, "P": 2.00},
                "U5" : {"A": 0.00342, "B": 0.00262, "P": 0.02},
                "C1" : {"A": 0.14, "B":0, "P": 0.02},
                "C2" : {"A": 13.5, "B":0, "P": 2.00},
                "C3" : {"A": 80.0, "B":0, "P": 2.00},
                "C4" : {"A": 120.0, "B":0, "P": 2.00},
                "C5" : {"A": 0.05, "B":0, "P": 0.04}}
    # Load Constants
    A = const[curve]["A"]
    B = const[curve]["B"]
    P = const[curve]["P"]
    if tfixed == None:
        # Evaluate in seconds from cycles
        CTI = CTI/freq
        # Evaluate M
        M = I / (CTR_dn * Ipu_dn)
        # Evaluate Trip Time
        tpu_desired = TDdn * (A/(M**P-1)+B) + CTI
    else:
        tpu_desired = tfixed + CTI
    # Re-Evaluate M
    M = I / (CTR_up * Ipu_up)
    # Calculate TD setting
    TD = tpu_desired / (A/(M**2-1)+B)
    # Scale and Round
    TD = _np.floor(TD*10**scale)/10**scale
    return(TD)

# Define TAP Calculator
def protectiontap(S,CTR=1,VLN=None,VLL=None):
    """
    Protection TAP Setting Calculator.
    
    Evaluates the required TAP setting based on the rated power of
    a transformer (the object being protected) and the voltage
    (either primary or secondary) in conjunction with the CTR
    (current transformer ratio) for the side in question (primary/
    secondary).
    
    Parameters
    ----------
    CTR:        float
                The Current Transformer Ratio.
    S:          float
                Rated apparent power magnitude (VA/VAR/W).
    VLN:        float, exclusive
                Line-to-Neutral voltage in volts.
    VLL:        float, exclusive
                Line-to-Line voltage in volts.
    
    Returns
    -------
    TAP:        float
                The TAP setting required to meet the specifications.
    """
    # Condition Voltage(s)
    if VLL != None:
        V = abs(_np.sqrt(3)*VLL)
    elif VLN != None:
        V = abs(3 * VLN)
    else:
        raise ValueError("One or more voltages must be provided.")
    # Calculate TAP
    TAP = abs(S) / (V*CTR)
    return(TAP)



# Define Current Correction Calculator
def correctedcurrents(Ipri,TAP,correction="Y",CTR=1):
    """
    Electrical Transformer Current Correction Function.
    
    Function to evaluate the currents as corrected for microprocessor-
    based relay protection schemes.
    
    Parameters
    ----------
    Ipri:       list of complex
                Three-phase set (IA, IB, IC) of primary currents.
    TAP:        float
                Relay's TAP setting.
    correction: string, optional
                String defining correction factor, may be one of:
                (Y, D+, D-, Z); Y denotes Y (Y0) connection, D+
                denotes Dab (D1) connection, D- denotes Dac (D11)
                connection, and Z (Z12) denotes zero-sequence
                removal. default="Y"
    CTR:        float
                Current Transformer Ratio, default=1
    
    Returns
    -------
    Isec_corr:  list of complex
                The corrected currents to perform operate/restraint
                calculations with.
    """
    # Define Matrix Lookup
    MAT = {   "Y"  : XFMY0,
              "D+" : XFMD1,
              "D-" : XFMD11,
              "Z"  : XFM12}
    # Condition Inputs
    Ipri = _np.asarray(Ipri)
    if isinstance(correction,list):
        mult = MAT[correction[0]]
        for i in correction[1:]:
            mult = mult.dot(MAT[i])
    elif isinstance(correction,str):
        mult = MAT[correction]
    elif isinstance(correction,_np.ndarray):
        mult = correction
    else:
        raise ValueError("Correction must be string or list of strings.")
    # Evaluate Corrected Current
    Isec_corr = 1/TAP * mult.dot(Ipri/CTR)
    return(Isec_corr)



# Define Iop/Irt Calculator
def iopirt(IpriHV,IpriLV,TAPHV,TAPLV,corrHV="Y",corrLV="Y",CTRHV=1,CTRLV=1):
    """
    Operate/Restraint Current Calculator.
    
    Calculates the operating current (Iop) and the restraint
    current (Irt) as well as the slope.
    
    Parameters
    ----------
    IpriHV:     list of complex
                Three-phase set (IA, IB, IC) of primary currents
                on the high-voltage side of power transformer.
    IpriLV      list of complex
                Three-phase set (IA, IB, IC) of primary currents
                on the low-voltage side of power transformer.
    TAPHV       float
                Relay's TAP setting for high-voltage side of
                power transformer.
    TAPLV       float
                Relay's TAP setting for low-voltage side of
                power transformer.
    corrHV      string, optional
                String defining correction factor on high-voltage
                side of power transformer, may be one of:
                (Y, D+, D-, Z); Y denotes Y (Y0) connection, D+
                denotes Dab (D1) connection, D- denotes Dac (D11)
                connection, and Z (Z12) denotes zero-sequence
                removal. default="Y"
    corrLV      string, optional
                String defining correction factor on low-voltage
                side of power transformer, may be one of:
                (Y, D+, D-, Z); Y denotes Y (Y0) connection, D+
                denotes Dab (D1) connection, D- denotes Dac (D11)
                connection, and Z (Z12) denotes zero-sequence
                removal. default="Y"
    CTRHV       float
                Current Transformer Ratio for high-voltage side
                of power transformer, default=1
    CTRLV       float
                Current Transformer Ratio for low-voltage side
                of power transformer, default=1
    
    Returns
    -------
    Iop:        list of float
                The operating currents for phases A, B, and C.
    Irt:        list of float
                The restraint currents for phases A, B, and C.
    slope:      list of float
                The calculated slopes for phases A, B, and C.
    """
    # Calculate Corrected Currents
    IcorHV = correctedcurrents(IpriHV,TAPHV,corrHV,CTRHV)
    IcorLV = correctedcurrents(IpriLV,TAPLV,corrLV,CTRLV)
    # Calculate Operate/Restraint Currents
    Iop = _np.absolute( IcorHV + IcorLV )
    Irt = _np.absolute(IcorHV) + _np.absolute(IcorLV)
    # Calculate Slopes
    slope = Iop/Irt
    return(Iop,Irt,slope)

# Define Symmetrical/RMS Current Calculation
def symrmsfaultcur(V,R,X,t=1/60,freq=60):
    """
    Symmetrical/RMS Current Calculator.
    
    Function to evaluate the time-constant tau, the symmetrical fault current,
    and the RMS current for a faulted circuit.
    
    Parameters
    ----------
    V:          float
                Voltage magnitude at fault point,
                not described as line-to-line or
                line-to-neutral.
    R:          float
                The fault resistance in ohms.
    X:          float
                The fault impedance in ohms.
    t:          float, optional
                The time in seconds.
    freq:       float, optional
                The system frequency in Hz.
    
    Returns
    -------
    tau:        float
                The time-constant tau in seconds.
    Isym:       float
                Symmetrical fault current in amps.
    Irms:       float
                RMS fault current in amps.
    """
    # Calculate Z and tau
    Z = _np.sqrt(R**2+X**2)
    tau = X/(2*_np.pi*freq*R)
    # Calculate Symmetrical Fault Current
    Isym = (V/_np.sqrt(3))/Z
    # Calculate RMS Fault Current
    Irms = _np.sqrt(1+2*_np.exp(-2*t/tau))*Isym
    return(tau,Isym,Irms)

# Define Relay M Formula
def faultratio(I,Ipickup,CTR=1):
    """
    Fault Multiple of Pickup (Ratio) Calculator.
    
    Evaluates the CTR-scaled pickup measured to pickup current ratio.
    
    M = meas / pickup
    
    Parameters
    ----------
    I:          float
                Measured Current in Amps
    Ipickup:    float
                Fault Current Pickup Setting (in Amps)
    CTR:        float, optional
                Current Transformer Ratio for relay,
                default=1
    
    Returns
    -------
    M:          float
                The measured-to-pickup ratio
    """
    M = I/(CTR * Ipickup)
    return(M)

# Define Residual Compensation Factor Function
def residcomp(z1,z0,linelength=1):
    """
    Residual Compensation Factor Function.
    
    Evaluates the residual compensation factor based on the line's positive and
    zero sequence impedance characteristics.
    
    Parameters
    ----------
    z1:         complex
                The positive-sequence impedance
                characteristic of the line, specified in 
                ohms-per-unit where the total line length
                (of same unit) is specified in
                *linelength* argument.
    z0:         complex
                The zero-sequence impedance characteristic
                of the line, specified in ohms-per-unit
                where the total line length (of same unit)
                is specified in *linelength* argument.
    linelength: float, optional
                The length (in same base unit as impedance
                characteristics) of the line. default=1
    
    Returns
    -------
    k0:         complex
                The residual compensation factor.
    """
    # Evaluate Z1L and Z0L
    Z1L = z1*linelength
    Z0L = z0*linelength
    # Calculate Residual Compensation Factor (k0)
    k0 = (Z0L - Z1L)/(3*Z1L)
    return(k0)

# Define Relay Measured Impedance Functon for Distance Elements
def distmeasz(VLNmeas,If,Ip,Ipp,CTR=1,VTR=1,k0=None,z1=None,z0=None,linelength=1):
    """
    Distance Element Measured Impedance Function.
    
    Function to evaluate the Relay-Measured-Impedance as calculated from
    the measured voltage, current, and line parameters.
    
    Parameters
    ----------
    VLNmeas:    complex
                Measured Line-to-Neutral voltage for the
                faulted phase in primary volts.
    If:         complex
                Faulted phase current measured in primary
                amps.
    Ip:         complex
                Secondary phase current measured in primary
                amps.
    Ipp:        complex
                Terchiary phase current measured in primary
                amps.
    CTR:        float, optional
                Current transformer ratio, default=1
    VTR:        float, optional
                Voltage transformer ratio, default=1
    k0:         complex, optional
                Residual Compensation Factor
    z1:         complex, optional
                The positive-sequence impedance
                characteristic of the line, specified in 
                ohms-per-unit where the total line length
                (of same unit) is specified in
                *linelength* argument.
    z0:         complex, optional
                The zero-sequence impedance characteristic
                of the line, specified in ohms-per-unit
                where the total line length (of same unit)
                is specified in *linelength* argument.
    linelength: float, optional
                The length (in same base unit as impedance
                characteristics) of the line. default=1
    
    Returns
    -------
    Zmeas:      complex
                The "measured" impedance as calculated by the relay.
    """
    # Validate Residual Compensation Inputs
    if k0 == z1 == z0 == None:
        raise ValueError("Residual compensation arguments must be set.")
    if k0 == None and (z1==None or z0==None):
        raise ValueError("Both *z1* and *z0* must be specified.")
    # Calculate Residual Compensation if Necessary
    if k0 == None:
        k0 = residcomp(z1,z0,linelength)
    # Convert Primary Units to Secondary
    V = VLNmeas/VTR
    Ir = (If + Ip + Ipp)/CTR
    I = If/CTR
    # Calculate Measured Impedance
    Zmeas = V / (I+k0*Ir)
    return(Zmeas)

# Define Transformer Tap Mismatch Function
def transmismatch(I1,I2,tap1,tap2):
    """
    Electrical Transformer TAP Mismatch Function.
    
    Function to evaluate the transformer ratio mismatch for protection.
    
    Parameters
    ----------
    I1:         complex
                Current (in amps) on transformer primary side.
    I2:         complex
                Current (in amps) on transformer secondary.
    tap1:       float
                Relay TAP setting on the primary side.
    tap2:       float
                Relay TAP setting on the secondary side.
    
    Returns
    -------
    mismatch:   float
                Transformer CT mismatch value associated with
                relay.
    """
    # Evaluate MR
    MR = min( abs(I1/I2), abs(tap1/tap2) )
    # Calculate Mismatch
    mismatch = (abs(I1/I2) - abs(tap1/tap2))*100/MR
    return(mismatch)

# Define High-Impedance Bus Protection Pickup Function
def highzvpickup(I,RL,Rct,CTR=1,threephase=False,Ks=1.5,
                 Vstd=400,Kd=0.5):
    """
    High Impedance Pickup Setting Function.
    
    Evaluates the voltage pickup setting for a high
    impedance bus protection system.
    
    Parameters
    ----------
    I:          float
                Fault current on primary side (in amps)
    RL:         float
                One-way line resistance in ohms
    Rct:        float
                Current Transformer Resistance in ohms
    CTR:        float, optional
                Current Transformer Ratio, default=1
    threephase: bool, optional
                Control argument to set the function to
                evaluate the result for a three-phase 
                fault or unbalanced fault. default=False
    Ks:         float, optional
                Security Factor for secure voltage pickup
                setting, default=1.5
    Vstd:       float, optional
                C-Class Voltage rating (i.e. C-400),
                default=400
    Kd:         float, optional
                The dependability factor for dependable
                voltage pickup setting, default=0.5
    
    Returns
    -------
    Vsens:      float
                The calculated sensetive voltage-pickup.
    Vdep:       float
                The calculated dependable voltage-pickup.
    """
    # Condition Based on threephase Argument
    n = 2
    if threephase: n = 1
    # Evaluate Secure Voltage Pickup
    Vsens = Ks*(n*RL+Rct)*I/CTR
    # Evaluate Dependible Voltage Pickup
    Vdep = Kd*Vstd
    return(Vsens,Vdep)

# Define Minimum Current Pickup for High-Impedance Bus Protection
def highzmini(N,Ie,Irly=None,Vset=None,Rrly=2000,Imov=0,CTR=1):
    """
    Minimum Current for High Impedance Protection Calculator.
    
    Evaluates the minimum pickup current required to cause
    high-impedance bus protection element pickup.
    
    Parameters
    ----------
    N:          int
                Number of Current Transformers included in scheme
    Ie:         float
                The excitation current at the voltage setting
    Irly:       float, optional
                The relay current at voltage setting
    Vset:       float, optional
                The relay's voltage pickup setting in volts.
    Rrly:       float, optional
                The relay's internal resistance in ohms, default=2000
    Imov:       float, optional
                The overvoltage protection current at the
                voltage setting. default=0.0
    CTR:        float, optional
                Current Transformer Ratio, default=1
    
    Returns
    -------
    Imin:       float
                Minimum current required to cause high-impedance
                bus protection element pickup.
    """
    # Validate Inputs
    if Irly == Vset == None:
        raise ValueError("Relay Current Required.")
    # Condition Inputs
    Ie = abs(Ie)
    Imov = abs(Imov)
    if Irly == None:
        Vset = abs(Vset)
        Irly = Vset / Rrly
    else:
        Irly = abs(Irly)
    # Evaluate Minimum Current Pickup
    Imin = (N*Ie+Irly+Imov)*CTR
    return(Imin)

# Define Instantaneous Overcurrent Pickup Formula
def instoc(Imin,CTR=1,Ki=0.5):
    """
    Instantaneous OverCurrent Pickup Calculator.
    
    Using a sensetivity factor and the CTR, evaluates the secondary-level pickup
    setting for an instantaneous overcurrent element.
    
    Parameters
    ----------
    Imin:       float
                The minimum fault current in primary amps.
    CTR:        float, optional
                Current Transformer Ratio, default=1
    Ki:         Sensetivity factor, default=0.5
    
    Returns
    -------
    Ipu:        float
                The pickup setting for the instantaneous
                overcurrent element as referred to the
                secondary side.
    """
    # Evaluate Overcurrent Pickup Setting
    Ipu = Ki * abs(Imin)/CTR
    return(Ipu)

# Define Generator Loss of Field Element Function
def genlossfield(Xd,Xpd,Zbase=1,CTR=1,VTR=1):
    """
    Electric Generator Loss of Field Function.
    
    Generates the Loss-of-Field Element settings for a generator using the Xd
    value and per-unit base information.
    
    Parameters
    ----------
    Xd:         float
                The Transient Reactance (Xd) term. May be
                specified in ohms or Per-Unit ohms if
                *Zbase* is set.
    Xpd:        float
                The Sub-Transient Reactance (X'd) term. May
                be specified in ohms or Per-Unit ohms if
                *Zbase* is set.
    Zbase:      float, optional
                Base impedance, used to convert per-unit
                Xd and Xpd to secondary values. default=1
    CTR:        float, optional
                Current Transformer Ratio, default=1
    VTR:        float, optional
                Voltage Transformer Ratio, default=1
    
    Returns
    -------
    ZoneOff:    float
                Zone Offset in ohms.
    Z1dia:      float
                Zone 1 diameter in ohms.
    Z2dia:      float
                Zone 2 diameter in ohms.
    """
    # Condition Inputs
    Xd = abs(Xd)
    Xpd = abs(Xpd)
    Zbase = abs(Zbase)
    # Evaluate Xd_sec and Xpd_sec
    Xd_sec = Xd*Zbase*(CTR/VTR)
    Xpd_sec = Xd*Zbase*(CTR/VTR)
    # Determine Zone Offset
    ZoneOff = Xpd_sec/2
    # Evaluate Z1 Diameter and Z2 Diameter
    Z1dia = Zbase*CTR/VTR
    Z2dia = Xd_sec
    # Return
    return(ZoneOff,Z1dia,Z2dia)

# Define Thermal Time Limit Calculator
def thermaltime(In,Ibase,tbase):
    r"""
    Thermal Time Limit Calculator.
    
    Computes the maximum allowable time for a specified current `In` given
    parameters for a maximum current and time at some other level, (`Ibase`,
    `tbase`).
    
    Uses the following formula:
    
    .. math:: t_n=\frac{I_{base}^2*t_{base}}{I_n^2}
    
    Parameters
    ----------
    In:         float
                Current at which to calculate max time.
    Ibase:      float
                Base current, at which maximum time
                `tbase` is allowable.
    tbase:      float
                Base time for which a maximum allowable
                current `Ibase` is specified. Unitless.
    
    Returns
    -------
    tn:         float
                Time allowable for specified current,
                `In`.
    """
    # Perform Calculation
    tn = (Ibase**2*tbase)/(In**2)
    return(tn)
    

# Define Synch. Machine Fault Current Calculator
def synmach_Isym(t,Eq,Xd,Xdp,Xdpp,Tdp,Tdpp):
    r"""
    Synch. Machine Symmetrical Fault Current Calc.
    
    Determines the Symmetrical Fault Current of a synchronous
    machine given the machine parameters, the internal voltage,
    and the time for which to calculate.
    
    .. math:: I_a(t)=\sqrt{2}\left|E_q\right|\left[
       \frac{1}{X_d}+\left(\frac{1}{X'_d}-\frac{1}{X_d}
       \right)\cdot e^{\frac{-t}{T'_d}}+\left(\frac{1}
       {X"_d}-\frac{1}{X'_d}\right)\cdot e^{\frac{-t}{T"_d}}
       \right]
    
    Parameters
    ----------
    t:          float
                Time at which to calculate the fault current
    Eq:         float
                The internal machine voltage in per-unit-volts
    Xd:         float
                The Xd (d-axis) reactance in per-unit-ohms
    Xdp:        float
                The X"d (d-axis transient) reactance in
                per-unit-ohms
    Xdpp:       float
                The X"d (d-axis subtransient) reactance in
                per-unit-ohms
    Tdp:        float
                The T'd (d-axis transient) time constant of the
                machine in seconds
    Tdpp:       float
                The T"d (d-axis subtransient) time constant of
                the machine in seconds
    
    Returns
    -------
    Ia:         float
                Peak symmetrical fault current in per-unit-amps
    """
    # Calculate Time-Constant Term
    t_c = 1/Xd+(1/Xdp-1/Xd)*_np.exp(-t/Tdp)+(1/Xdpp-1/Xdp)*_np.exp(-t/Tdpp)
    # Calculate Fault Current
    Ia = _np.sqrt(2)*abs(Eq)*t_c
    return(Ia)

# Define Synch. Machine Asymmetrical Current Calculator
def synmach_Iasym(t,Eq,Xdpp,Xqpp,Ta):
    r"""
    Synch. Machine Asymmetrical Fault Current Calc.
    
    Determines the asymmetrical fault current of a synchronous
    machine given the machine parameters, the internal voltage,
    and the time for which to calculate.
    
    .. math:: I_{asym}=\sqrt{2}\left|E_q\right|\frac{1}{2}
       \left(\frac{1}{X"_d}+\frac{1}{X"_q}\right)e^{\frac{-t}
       {T_a}}
    
    Parameters
    ----------
    t:          float
                Time at which to calculate the fault current
    Eq:         float
                The internal machine voltage in per-unit-volts
    Xdpp:       float
                The X"d (d-axis subtransient) reactance in
                per-unit-ohms
    Xqpp:       float
                The X"q (q-axis subtransient) reactance in
                per-unit-ohms
    Ta:         float
                Armature short-circuit (DC) time constant in seconds
    
    Returns
    -------
    Iasym:      float
                Peak asymmetrical fault current in per-unit-amps
    """
    # Calculate Time Constant Term
    t_c = 1/Xdpp + 1/Xqpp
    # Calculate Asymmetrical Current
    Iasym = _np.sqrt(2)*abs(Eq)*1/2*t_c*_np.exp(-t/Ta)
    return(Iasym)

# Define Induction Machine Eigenvalue Calculator
def indmacheigenvalues(Lr,Ls,Lm,Rr,Rs,wrf=0,freq=60):
    """
    Induction Machine Eigenvalue Calculator.
    
    Calculates the pertinent eigenvalues for an unloaded
    induction machine given a specific set of machine
    parameters.
    
    Parameters
    ----------
    Lr:         float
                Inductance of the Rotor (in Henrys).
    Ls:         float
                Inductance of the Stator (in Henrys).
    Lm:         float
                Inductance of the Magnetizing branch
                (in Henrys).
    Rr:         float
                Resistance of the Rotor (in Ohms).
    Rs:         float
                Resistance of the Stator (in Ohms).
    wrf:        float, optional
                Frequency (in radians/sec) of the rotor slip.
                default=0
    freq:       float, optional
                Base frequency of the system (in Hertz).
                default=60
    
    Returns
    -------
    lam1:       complex
                The First Eigenvalue
    lam2:       complex
                The Second Eigenvalue
    """
    # Calculate Required Values
    omega_e_base = 2*_np.pi*freq
    omega_rf = wrf
    torque_s = Ls/(omega_e_base*Rs)
    torque_r = Lr/(omega_e_base*Rr)
    alpha = torque_r / torque_s
    phi = 1 - Lm**2/(Ls*Lr)
    omega_r = omega_e_base
    # Calculate k1
    k1 = -1/(2*phi*torque_r)*(1+alpha)
    k1 += 1j*(omega_r/2-omega_rf)
    # Calculate k2
    k2 = 1/(2*phi*torque_r)
    k2 *= _np.sqrt((1+alpha)**2-4*phi*alpha-(omega_r*phi*torque_r)**2
                 +2j*(alpha-1)*omega_r*phi*torque_r)
    # Evaluate Eigenvalues and Return
    lam1 = k1+k2
    lam2 = k1-k2
    return(lam1,lam2)

# Define IM 3-Phase SC Current Calculator
def indmachphs3sc(t,Is0,Lr,Ls,Lm,Rr,Rs,wrf=0,freq=60,real=True):
    """
    Induction Machine 3-Phase SC Calculator.
    
    Determines the short-circuit current at a specified time for a three-phase
    fault on an unloaded induction machine.
    
    Parameters
    ----------
    t:          array_like
                The time at which to find the
                current, may be int, float, or
                numpy array.
    Is0:        complex
                The initial (t=0) current on
                the stator.
    Lr:         float
                Inductance of the Rotor (in Henrys).
    Ls:         float
                Inductance of the Stator (in Henrys).
    Lm:         float
                Inductance of the Magnetizing branch
                (in Henrys).
    Rr:         float
                Resistance of the Rotor (in Ohms).
    Rs:         float
                Resistance of the Stator (in Ohms).
    wrf:        float, optional
                Frequency (in radians/sec) of the rotor slip.
                default=0
    freq:       float, optional
                Base frequency of the system (in Hertz).
                default=60
    real:       bool, optional
                Control argument to force returned value
                to be real part only. default=True
    
    Returns
    -------
    ias:        array_like
                Fault Current
    """
    # Calculate Required Values
    omega_r = 2*_np.pi*freq
    torque_s = Ls/(omega_r*Rs)
    phi = 1 - Lm**2/(Ls*Lr)
    # Calculate Eigenvalues
    lam1, lam2 = indmacheigenvalues(Lr,Ls,Lm,Rr,Rs,wrf,freq)
    # Calculate pIs0
    pIs0 = -(1/(phi*torque_s)+1j*(1-phi)/phi*omega_r)*Is0
    # Calculate Constants
    C1 = (lam2*Is0-pIs0)/(lam2-lam1)
    C2 = (pIs0-lam1*Is0)/(lam2-lam1)
    # Calculate ias and Return
    ias = C1*_np.exp(lam1*t)+C2*_np.exp(lam2*t)
    if real:
        ias = _np.real(ias)
    return(ias)

# Define IM Torque Calculation
def indmachphs3torq(t,Is0,Lr,Ls,Lm,Rr,Rs,wrf=0,freq=60):
    """
    Induction Machine 3-Phase Torque Calculator.
    
    Determines the torque exerted during a three-phase fault on an induction
    machine.
    
    Parameters
    ----------
    t:          array_like
                The time at which to find the
                current, may be int, float, or
                numpy array.
    Is0:        complex
                The initial (t=0) current on
                the stator.
    Lr:         float
                Inductance of the Rotor (in Henrys).
    Ls:         float
                Inductance of the Stator (in Henrys).
    Lm:         float
                Inductance of the Magnetizing branch
                (in Henrys).
    Rr:         float
                Resistance of the Rotor (in Ohms).
    Rs:         float
                Resistance of the Stator (in Ohms).
    p:          int
                Number of electrical poles.
    wrf:        float, optional
                Frequency (in radians/sec) of the rotor slip.
                default=0
    freq:       float, optional
                Base frequency of the system (in Hertz).
                default=60
    
    Returns
    -------
    Tem:        array_like
                Induction machine torque in N*m
    """
    # Calculate Required Values
    omega_r = 2*_np.pi*freq
    torque_s = Ls/(omega_r*Rs)
    phi = 1 - Lm**2/(Ls*Lr)
    # Calculate Eigenvalues
    lam1, lam2 = indmacheigenvalues(Lr,Ls,Lm,Rr,Rs,wrf,freq)
    # Calculate pIs0
    pIs0 = -(1/(phi*torque_s)+1j*(1-phi)/phi*omega_r)*Is0
    # Calculate Constants
    C1 = (lam2*Is0-pIs0)/(lam2-lam1)
    C2 = (pIs0-lam1*Is0)/(lam2-lam1)
    # Calculate ias and Return
    idqs = C1*_np.exp(lam1*t)+C2*_np.exp(lam2*t)
    idqr = C2*_np.exp(lam1*t)+C1*_np.exp(lam2*t)
    # Calculate Lambda
    lamdqr = Lm*idqs+Lr*idqr
    # Calculate Torque
    Tem = Lm/Lr * (lamdqr.real*idqs.imag - lamdqr.imag*idqs.real)
    return(Tem)

# Define Complete Sync. Mach. Fault Current Function
def synmach_ifault(t,Ea,alpha,Xd,Xdp,Xdpp,Xqpp,Tdp,Tdpp,Ta,freq=60):
    # noqa: D401   "Synchronous" is intentional descriptor
    """
    Synchronous Machine Fault Current Calculator.
    
    Given machine parameters, fault inception angle, and time at
    which to calculate fault current, this function will identify
    the complete (symmetrical, asymmetrical, and double frequency)
    fault current.
    
    .. image:: /static/synmach_ifault_formula.png
    
    Parameters
    ----------
    t:          float
                Time at which to calculate the fault current
    Eq:         float
                The internal machine voltage in per-unit-volts
    alpha:      float
                Fault inception angle (in degrees)
    Xd:         float
                The Xd (d-axis) reactance in per-unit-ohms
    Xdp:        float
                The X"d (d-axis transient) reactance in
                per-unit-ohms
    Xdpp:       float
                The X"d (d-axis subtransient) reactance in
                per-unit-ohms
    Xqpp:       float
                The X"q (q-axis subtransient) reactance in
                per-unit-ohms
    Tdp:        float
                The T'd (d-axis transient) time constant of the
                machine in seconds
    Tdpp:       float
                The T"d (d-axis subtransient) time constant of
                the machine in seconds
    Ta:         float
                Armature short-circuit (DC) time constant in seconds
    freq:       float, optional
                System (electrical) frequency (in degrees),
                default=60
    
    Returns
    -------
    ias:        float
                Synchronous machine fault current (symmetrical,
                asymmetrical, and double frequency component) in
                amps
    """
    # Calculate we Component
    we = 2*_np.pi*freq
    # Condition Inputs
    Ea = abs(Ea)
    alpha = _np.radians(alpha)
    # Define Constant Term
    const = _np.sqrt(2)*Ea
    if Xqpp != 0:
        val = 1/Xqpp
    else:
        val = 0
    asym = 1/2*(1/Xdpp+val)*_np.exp(t/Ta)
    # Define Symmetrical Portion
    isym = const*(1/Xd+(1/Xdp-1/Xd)*_np.exp(-t/Tdp)
               +(1/Xdpp-1/Xdp)*_np.exp(-t/Tdpp))*_np.sin(we*t+alpha)
    # Define Asymmetrical Portion
    iasym = const*asym*_np.sin(alpha)
    # Define Double Frequency Term
    idbl = const*1/2*asym*_np.sin(2*we*t+alpha)
    # Compose Complet Current Value
    ias = isym - iasym - idbl
    return(ias)
    
    

# END OF FILE