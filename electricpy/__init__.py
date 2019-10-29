###################################################################
"""
---------------
`electricpy.py`
---------------

 `A library of functions, constants and more
 that are related to Power in Electrical Engineering.`

 Written by Joe Stanley


Included Constants
------------------
 - Pico Multiple:                            p
 - Nano Multiple:                            n
 - Micro (mu) Multiple:                      u
 - Mili Multiple:                            m
 - Kilo Multiple:                            k
 - Mega Multiple:                            M
 - 'A' Operator for Symmetrical Components:  a
 - Not a Number value (NaN):                 NAN
 
Additional constants may not be listed. For a complete list,
visit constants page.

Symmetrical Components Matricies
--------------------------------
 - ABC to 012 Conversion:        Aabc
 - 012 to ABC Conversion:        A012

Included Functions
------------------
 - Phasor V/I Generator:                    phasor
 - Phasor Array V/I Generator:              phasorlist
 - Phasor Data Genorator:                   phasordata
 - Phase Angle Generator:                   phs
 - Time of Number of Cycles:                tcycle        
 - Phasor Impedance Generator:              phasorz
 - Complex Display Function:                cprint
 - Complex LaTeX Display Function:          clatex
 - Transfer Function LaTeX Generator:       tflatex
 - Parallel Impedance Adder:                parallelz
 - V/I Line/Phase Converter:                phaseline
 - Power Set Values:                        powerset
 - Power Triangle Function:                 powertriangle
 - Transformer SC OC Tests:                 transformertest
 - Phasor Plot Generator:                   phasorplot
 - Total Harmonic Distortion:               thd
 - Total Demand Distortion:                 tdd
 - Reactance Calculator:                    reactance
 - Non-Linear PF Calc:                      nlinpf
 - Harmonic Limit Calculator:               harmoniclimit
 - Power Factor Distiortion:                pfdist
 - Short-Circuit RL Current:                iscrl
 - Voltage Divider:                         voltdiv
 - Current Divider:                         curdiv
 - Instantaneous Power Calc.:               instpower
 - Delta-Wye Network Converter:             dynetz
 - Single Line Power Flow:                  powerflow
 - Thermocouple Temperature:                thermocouple
 - Cold Junction Voltage:                   coldjunction
 - RTD Temperature Calculator:              rtdtemp
 - Horsepower to Watts:                     hp_to_watts
 - Watts to Horsepower:                     watts_to_hp
 - Inductor Charge:                         inductorcharge
 - Inductor Discharge:                      inductordischarge
 - Inductor Stored Energy:                  inductorenergy      
 - Back-to-Back Cap. Surge:                 capbacktoback  
 - Capacitor Stored Energy:                 energy
 - Cap. Voltage after Time:                 VafterT
 - Cap. Voltage Discharge:                  vcapdischarge
 - Cap. Voltage Charge:                     vcapcharge
 - Rectifier Cap. Calculation:              rectifiercap
 - Cap. VAR to FARAD Conversion:            farads
 - VSC DC Bus Voltage Calculator:           vscdcbus
 - PLL-VSC Gains Calculator:                vscgains
 - RMS Calculator:                          rms
 - Step Function:                           step
 - Multi-Argument Convolution:              convolve
 - Convolution Bar Graph Visualizer:        convbar
 - Gaussian Function:                       gaussian
 - Gaussian Distribution Calculator:        gausdist
 - Probability Density Calculator:          probdensity
 - Real FFT Evaluator:                      rfft
 - Normalized Power Spectrum:               wrms
 - Hartley's Data Capacity Equation:        hartleydata
 - Shannon's Data Capacity Equation:        shannondata
 - String to Bit-String Converter:          string_to_bits
 - CRC Message Generator:                   crcsender
 - CRC Remainder Calculator:                crcremainder
 - kWh to BTU:                              kwh_to_btu
 - BTU to kWh:                              btu_to_kwh
 - Per-Unit Impedance Calculator:           zpu
 - Per-Unit Current Calculator:             ipu
 - Per-Unit Change of Base Formula:         puchgbase
 - Per-Unit to Ohmic Impedance:             zrecompose
 - X over R to Ohmic Impedance:             rxrecompose
 - Generator Internal Voltage Calc:         geninternalv
 - Phase to Sequence Conversion:            abc_to_seq
 - Sequence to Phase Conversion:            seq_to_abc
 - Sequence Impedance Calculator:           sequencez
 - Function Harmonic (FFT) Evaluation:      funcfft
 - Dataset Harmonic (FFT) Evaluation:       sampfft
 - Harmonic (FFT) Component Plotter:        fftplot
 - Harmonic (FFT) Summation Plotter:        fftsumplot 
 - Harmonic System Generator:               harmonics   
 - Motor Startup Capacitor Formula:         motorstartcap
 - Power Factor Correction Formula:         pfcorrection
 - AC Power/Voltage/Current Relation:       acpiv
 - Transformer Primary Conversion:          primary
 - Transformer Secondary Conversion:        secondary
 - Natural Frequency Calculator             natfreq
 - 3-Phase Voltage/Current Unbalance:       unbalance
 - Characteristic Impedance Calculator:     characterz

Additional Available Sub-Modules
--------------------------------
 - fault.py
 - bode.py
 - sim.py

Functions Available in `electricpy.fault.py`
--------------------------------------------
 - Single Line to Ground                 phs1g
 - Double Line to Ground                 phs2g
 - Line to Line                          phs2
 - Three-Phase Fault                     phs3
 - Faulted Bus Voltage                   busvolt
 - CT Saturation Function                ct_saturation
 - CT C-Class Calculator                 ct_cclass
 - CT Sat. V at rated Burden             ct_satratburden
 - CT Voltage Peak Formula               ct_vpeak
 - CT Time to Saturation                 ct_timetosat
 - Transient Recovery Voltage Calc.      pktransrecvolt
 - TRV Reduction Resistor                trvresistor
 - TOC Trip Time                         toctriptime
 - TOC Reset Time                        tocreset
 - Pickup Setting Assistant              pickup
 - Radial TOC Coordination Tool          tdradial
 - TAP Setting Calculator                protectiontap
 - Transformer Current Correction        correctedcurrents
 - Operate/Restraint Current Calc.       iopirt
 - Symmetrical/RMS Fault Current Calc:   symrmsfaultcur
 - TOC Fault Current Ratio:              faultratio
 - Residual Compensation Factor Calc:    residcomp
 - Distance Elem. Impedance Calc:        distmeasz
 - Transformer Mismatch Calculator:      transmismatch
 - High-Impedance Voltage Pickup:        highzvpickup
 - High-Impedance Minimum Current PU:    highzmini
 - Instantaneous Overcurrent PU:         instoc
 - Generator Loss of Field Settings:     genlossfield
 - Thermal Time Limit Calculator:        thermaltime

Functions Available in `electricpy.bode.py`
-------------------------------------------
 - Transfer Function Bode Plotter:       bode
 - S-Domain Bode Plotter:                sbode
 - Z-Domain Bode Plotter:                zbode

Functions Available in `electricpy.sim.py`
------------------------------------------
 - Digital Filter Simulator:             digifiltersim
 - Step Response Filter Simulator:       step_response
 - Ramp Response Filter Simulator:       ramp_response
 - Parabolic Response Filter Simulator:  parabolic_response
 - State-Space System Simulator:         statespace
 - Newton Raphson Calculator:            NewtonRaphson
 - Power Flow System Generator:          nr_pq
 - Multi-Bus Power Flow Calculator:      mbuspowerflow
"""
###################################################################

# Define Module Specific Variables
_name_ = "electricpy"
_version_ = "0.0.6"
# Version Breakdown:
# MAJOR CHANGE . MINOR CHANGE . MICRO CHANGE

# Import Submodules
from .constants import *
from . import fault
from . import bode
from . import sim

# Import Supporting Modules
import numpy as _np
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt
import cmath as _c


# Define Phase Angle Generator
def phs( ang ):
    """
    Complex Phase Angle Generator
    
    Generate a complex value given the phase angle
    for the complex value.
    
    Parameters
    ----------
    ang:        float
                The angle (in degrees) for which
                the value should be calculated.
    """
    # Return the Complex Angle Modulator
    return(_np.exp(1j*_np.radians( ang )))
    

# Define Phasor Generator
def phasor( mag, ang=None ):
    """
    Complex Phasor Generator
    
    Generates the standard Pythonic complex representation
    of a phasor voltage or current when given the magnitude
    and angle of the specific voltage or current.
    
    Parameters
    ----------
    mag:        float
                The Magnitude of the Voltage/Current
    ang:        float
                The Angle (in degrees) of the Voltage/Current
    
    Returns
    -------
    phasor:     complex
                Standard Pythonic Complex Representation of
                the specified voltage or current.
    
    Examples
    --------
    >>> import electricpy as ep
    >>> ep.phasor(67, 120) # 67 volts at angle 120 degrees
    (-33.499999999999986+58.02370205355739j)
    
    See Also
    --------
    phasorlist: Phasor Generator for List or Array
    cprint:     Complex Variable Printing Function
    phasorz:    Impedance Phasor Generator    
    """
    # Test for Tuple/List Arg
    if isinstance(mag, (tuple,list,_np.ndarray)):
        ang = mag[1]
        mag = mag[0]
    return( _c.rect( mag, _np.radians( ang ) ) )

# Define Phasor Array Generator
def phasorlist( arr ):
    """
    Complex Phasor Generator for 2-D Array or 2-D List
    
    Generates the standard Pythonic complex representation
    of a phasor voltage or current when given the magnitude
    and angle of the specific voltage or current for a list
    or array of values.
    
    Parameters
    ----------
    arr:        numpy.ndarray
                2-D array or list of magnitudes and angles.
                Each item must be set of magnitude and angle
                in form of: [mag, ang].
    
    Returns
    -------
    phasor:     complex
                Standard Pythonic Complex Representation of
                the specified voltage or current.
    
    Examples
    --------
    >>> import numpy as np
    >>> import electricpy as ep
    >>> voltages = _np.array([[67,0],
                             [67,-120],
                             [67,120]])
    >>> Vset = ep.phasorlist( voltages )
    >>> print(Vset)
    
    See Also
    --------
    phasor:     Phasor Generating Function
    cprint:     Complex Variable Printing Function
    phasorz:    Impedance Phasor Generator
    """
    # Iteratively Process
    outarr = _np.array([])
    for i in arr:
        outarr = _np.append(outarr, phasor( i ))
    # Return Array
    return(outarr)

# Define Phasor Data Generator
def phasordata(mn,mx=None,npts=1000,mag=1,ang=0,freq=60,
               retstep=False,rettime=False,sine=False):
    """
    Complex Phasor Data Generator
    
    Generates a sinusoidal data set with minimum, maximum,
    frequency, magnitude, and phase angle arguments.
    
    Parameters
    ----------
    mn:         float, optional
                Minimum time (in seconds) to generate data for.
                default=0
    mx:         float
                Maximum time (in seconds) to generate data for.
    npts:       float, optional
                Number of data samples. default=1000
    mag:        float, optional
                Sinusoid magnitude, default=1
    ang:        float, optional
                Sinusoid angle in degrees, default=0
    freq:       float, optional
                Sinusoid frequency in Hz
    retstep:    bool, optional
                Control argument to request return of time
                step size (dt) in seconds.
    sine:       bool, optional
                Control argument to require data be generated
                with a sinusoidal function instead of cosine.
                
    Returns
    -------
    data:       numpy.ndarray
                The resultant data array.
    """
    # Test Inputs for Min/Max
    if mx == None:
        # No Minimum provided, use Value as Maximum
        mx = mn
        mn = 0
    # Generate Omega
    w = 2*_np.pi*freq
    # Generate Time Array
    t,dt = _np.linspace(mn,mx,npts,retstep=True)
    # Generate Data Array
    if not sine:
        data = mag * _np.cos(w*t + _np.radians(ang))
    else:
        data = mag * _np.sin(w*t + _np.radians(ang))
    # Generate Return Data Set
    dataset = [data]
    if retstep:
        dataset.append(dt)
    if rettime:
        dataset.append(t)
    # Return Dataset
    if len(dataset) == 1:
        return(dataset[0])
    return(dataset)

# Define Complex LaTeX Generator
def clatex(val,round=3,polar=True,predollar=True,postdollar=True,double=False):
    """
    Complex Value Latex Generator
    
    Function to generate a LaTeX string of complex value(s)
    in either polar or rectangular form. May generate both dollar
    signs.
    
    Parameters
    ----------
    val:        complex
                The complex value to be printed, if value
                is a list or numpy array, the result will be
                demonstrated as a matrix.
    round:      int, optional
                Control to specify number of decimal places
                that should displayed. default=True
    polar:      bool, optional
                Control argument to force result into polar
                coordinates instead of rectangular. default=True
    predollar:  bool, optional
                Control argument to enable/disable the dollar
                sign before the string. default=True
    postdollar: bool, optional
                Control argument to enable/disable the dollar
                sign after the string. default=True
    double:     bool, optional
                Control argument to specify whether or not
                LaTeX dollar signs should be double or single,
                default=False
    
    Returns
    -------
    latex:      str
                LaTeX string for the complex value.
    """
    # Define Interpretation Functions
    def polarstring( val, round ):
        mag, ang_r = _c.polar(val) #Convert to polar form
        ang = _np.degrees(ang_r) #Convert to degrees
        mag = _np.around( mag, round ) #Round
        ang = _np.around( ang, round ) #Round
        latex = str(mag) + '∠' + str(ang) + '°'
        return(latex)
    def rectstring( val, round ):
        real = _np.around( val.real, round ) #Round
        imag = _np.around( val.imag, round ) #Round
        if imag > 0:
            latex = str(real) + "+j" + str(imag)
        else:
            latex = str(real) + "-j" + str(abs(imag))
        return(latex)
    # Interpret as numpy array if simple list
    if isinstance(val, list):
        val = _np.asarray(val) # Ensure that input is array
    # Find length of the input array
    if isinstance(val,_np.ndarray):
        shp = val.shape
        try:
            row, col = shp # Interpret Shape of Object
        except:
            row = shp[0]
            col = 1
        sz = val.size
        # Open Matrix
        latex = r'\begin{bmatrix}'
        # Iteratively Process Each Item in Array
        for ri in range(row):
            if ri != 0: # Insert Row Separator
                latex += r'\\'
            if col > 1:
                for ci in range(col):
                    if ci != 0: # Insert Column Separator
                        latex += r' & '
                    # Add Complex Represetation of Value
                    if polar: latex += polarstring( val[ri][ci], round )
                    else: latex += rectstring( val[ri][ci], round )
            else:
                # Add Complex Represetation of Value
                if polar: latex += polarstring( val[ri], round )
                else: latex += rectstring( val[ri], round )
        # Close Matrix
        latex += r'\end{bmatrix}'
    elif isinstance(val, complex):
        # Treat as Polar When Directed
        if polar:
            latex = polarstring( val, round )
        else:
            latex = rectstring( val, round )
    else:
        raise ValueError("Invalid Input Type")
    # Add Dollar Sign pre-post
    if double:
        dollar = r'$$'
    else:
        dollar = r'$'
    if predollar:
        latex = dollar + latex
    if postdollar:
        latex = latex + dollar
    return( latex )

# Define Transfer Function LaTeX Generator
def tflatex(sys,sysp=None,var='s',predollar=True,
            postdollar=True,double=False,tolerance=1e-8):
    """
    Transfer Function LaTeX String Generator
    
    LaTeX string generating function to create a transfer
    function string in LaTeX. Particularly useful for
    demonstrating systems in Interactive Python Notebooks.
    
    Parameters
    ----------
    sys:        list
                If provided in conjunction with optional
                parameter `sysp`, the parameter `sys` will
                act as the numerator set. Otherwise, can be
                passed as a list containing two sublists,
                the first being the numerator set, and the
                second being the denominator set.
    sysp:       list, optional
                If provided, this input will act as the
                denominator of the transfer function.
    var:        str, optional
                The variable that should be printed for each
                term (i.e. 's' or 'j\\omega'). default='s'
    predollar:  bool, optional
                Control argument to enable/disable the dollar
                sign before the string. default=True
    postdollar: bool, optional
                Control argument to enable/disable the dollar
                sign after the string. default=True
    double:     bool, optional
                Control argument to specify whether or not
                LaTeX dollar signs should be double or single,
                default=False
    tolerance:  float, optional
                The floating point tolerance cutoff to evaluate
                each term against. If the absolute value of the
                particular term is greater than the tolerance,
                the value will be printed, if not, it will not
                be printed. default=1e-8
    
    Returns
    -------
    latex:      str
                LaTeX string for the transfer function.
    """
    # Collect Numerator and Denominator Terms
    if isinstance(sysp, (list,tuple,_np.ndarray)):
        num = sys
        den = sysp
    else:
        num, den = sys
    # Generate String Function
    def genstring( val ):
        length = len(val)
        strg = ''
        for i,v in enumerate(val):
            # Add Each Term to String
            if abs(v) > tolerance:
                # Add '+' Symbol After Each Term
                if i != 0:
                    strg += r'+'
                strg += str(v)
                # Determine Exponent
                xpnt = length-i-1
                if xpnt == 1:
                    strg += var
                elif xpnt == 0:
                    pass # Don't Do Anything
                else:
                    strg += var+r'^{'+str(xpnt)+r'}'
        return( strg )
    # Generate Total TF String
    latex = r'\frac{' + genstring( num ) + r'}{'
    latex += genstring( den ) + r'}'
    # Add Dollar Sign pre-post
    if double:
        dollar = r'$$'
    else:
        dollar = r'$'
    if predollar:
        latex = dollar + latex
    if postdollar:
        latex = latex + dollar
    return( latex )

# Define Cycle Time Function
def tcycle(ncycles=1,freq=60):
    """
    Time of Electrical Cycles
    
    Evaluates the time for a number of n
    cycles given the system frequency.
    
    .. math:: t = \\frac{n_{cycles}}{freq}
    
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
    """
    # Evaluate the time for ncycles
    return(ncycles/freq)

# Define Reactance Calculator
def reactance(z,freq=60,sensetivity=1e-12):
    """
    Capacitance/Inductance from Impedance
    
    Calculates the Capacitance or Inductance in Farads or Henreys
    (respectively) provided the impedance of an element.
    Will return capacitance (in Farads) if ohmic impedance is
    negative :eq:`cap`, or inductance (in Henrys) if ohmic impedance is
    positive :eq:`ind`. If imaginary: calculate with j factor
    (imaginary number).
    
    .. math:: C = \\frac{1}{\\omega*Z}
       :label: cap
    
    .. math:: L = \\frac{Z}{\\omega}
       :label: ind
    
    This requires that the radian frequency is found as follows:
    
    .. math:: \\omega = 2*\\pi*freq
    
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
    w = 2*_np.pi*freq
    # Input is Complex
    if isinstance(z, complex):
        # Test for Resistance
        if(abs(z.real) > sensetivity):
            R = z.real
        else:
            R = 0
        if (z.imag > 0):
            out = z/(w*1j)
        else:
            out = 1/(w*1j*z)
        out = abs(out)
        # Combine with resistance if present
        if(R!=0): out = (R, out)
    else:
        if (z > 0):
            out = z/(w)
        else:
            out = 1/(w*z)
        out = abs(out)
    # Return Output
    return(out)

# Define display function
def cprint(val,unit=None,label=None,pretty=False,printval=True,ret=False,round=3):
    """
    Phasor (Complex) Printing Function
    
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
    pretty:     bool, optional
                Control argument to force printed result to a *pretty*
                format without array braces. default=False
    printval:   bool, optional
                Control argument enabling/disabling printing of the string.
                default=True
    ret:        bool, optional
                Control argument allowing the evaluated value to be returned.
                default=False
    round:      int, optional
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
    >>> import electricpy as ep
    >>> v = ep.phasor(67, 120)
    >>> ep.cprint(v)
    67.0 ∠ 120.0°
    >>> voltages = _np.array([[67,0],
                             [67,-120],
                             [67,120]])
    >>> Vset = ep.phasorlist( voltages )
    >>> ep.cprint(Vset)
    [['67.0 ∠ 0.0°']
    ['67.0 ∠ -120.0°']
    ['67.0 ∠ 120.0°']]

    
    See Also
    --------
    phasor:     Phasor Generating Function
    phasorlist: Phasor Generating Function for Lists or Arrays
    phasorz:    Impedance Phasor Generator
    """
    # Interpret as numpy array if simple list
    if isinstance(val, list):
        val = _np.asarray(val) # Ensure that input is array
    # Find length of the input array
    if isinstance(val,_np.ndarray):
        shp = val.shape
        try:
            row, col = shp # Interpret Shape of Object
        except:
            row = shp[0]
            col = 1
        sz = val.size
        # Handle Label as a List or Array
        if isinstance(label, (list,_np.ndarray)):
            if len(label)==1:
                tmp = label
                for _ in range(sz):
                    label = _np.append(label,[tmp])
            elif sz != len(label):
                raise ValueError("Too Few Label Arguments")
        # Handle Label as String
        elif isinstance(label, str):
            tmp = label
            for _ in range(sz):
                label = _np.append(label,[tmp])
        # Handle Lack of Label
        elif label == None:
            label = _np.array([])
            for _ in range(sz):
                label = _np.append(label,None)
        # Handle all Other Cases
        else:
            raise ValueError("Invalid Label")
        # Handle Unit as a List or Array
        if isinstance(unit, (list,_np.ndarray)):
            if len(unit)==1:
                tmp = unit
                for _ in range(sz):
                    unit = _np.append(unit,[tmp])
            elif sz != len(unit):
                raise ValueError("Too Few Unit Arguments")
        # Handle Unit as String
        elif isinstance(unit, str):
            tmp = unit
            for _ in range(sz):
                unit = _np.append(unit,[tmp])
        # Handle Lack of Unit
        elif unit == None:
            unit = _np.array([])
            for _ in range(sz):
                unit = _np.append(unit,None)
        # Handle all Other Cases
        else:
            raise ValueError("Invalid Unit")
        # Generate Default Arrays
        printarr = _np.array([]) # Empty array
        numarr = _np.array([]) # Empty array
        # Operate on List/Array
        for i in range(row):
            _val = val[i]
            _label = label[i]
            _unit = unit[i]
            mag, ang_r = _c.polar(_val) #Convert to polar form
            ang = _np.degrees(ang_r) #Convert to degrees
            mag = _np.around( mag, round ) #Round
            ang = _np.around( ang, round ) #Round
            strg = ""
            if _label != None:
                strg += _label + " "
            strg += str(mag)+" ∠ "+str(ang)+"°"
            if _unit != None:
                strg += " " + _unit
            printarr = _np.append(printarr, strg)
            numarr = _np.append(numarr, [mag, ang])
        # Reshape Arrays
        printarr = _np.reshape(printarr, (row,col))
        numarr = _np.reshape(numarr, (sz, 2))
        # Print
        if printval and row==1:
            print(strg)
        elif printval and pretty:
            strg = ''
            for i in printarr:
                strg += str(i[0]) + '\n'
            print(strg)
        elif printval:
            print(printarr)
        # Return if Necessary
        if ret:
            return(numarr)
    elif isinstance(val, (int,float,complex)):
        # Handle Invalid Unit/Label
        if unit != None and not isinstance(unit, str):
            raise ValueError("Invalid Unit Type for Value")
        if label != None and not isinstance(label, str):
            raise ValueError("Invalid Label Type for Value")
        mag, ang_r = _c.polar(val) #Convert to polar form
        ang = _np.degrees(ang_r) #Convert to degrees
        mag = _np.around( mag, round ) #Round
        ang = _np.around( ang, round ) #Round
        strg = ""
        if label != None:
            strg += label + " "
        strg += str(mag)+" ∠ "+str(ang)+"°"
        if unit != None:
            strg += " " + unit
        # Print values (by default)
        if printval:
            print(strg)
        # Return values when requested
        if ret:
            return([mag, ang])
    else:
        raise ValueError("Invalid Input Type")

# Define Impedance Conversion function
def phasorz(C=None,L=None,f=60,complex=True):
    """
    Phasor Impedance Generator
    
    This function's purpose is to generate the phasor-based
    impedance of the specified input given as either the
    capacitance (in Farads) or the inductance (in Henreys).
    The function will return the phasor value (in Ohms).
    
    .. math:: Z = \\frac{-j}{\\omega*C}
    
    .. math:: Z = j*\\omega*L
    
    where:
    
    .. math:: \\omega = 2*\\pi*freq
    
    Parameters
    ----------
    C:          float, optional
                The capacitance value (specified in Farads),
                default=None
    L:          float, optional
                The inductance value (specified in Henreys),
                default=None
    f:          float, optional
                The system frequency to be calculated upon, default=60
    complex:    bool, optional
                Control argument to specify whether the returned
                value should be returned as a complex value.
                default=True
    
    Returns
    -------
    Z:      complex
            The ohmic impedance of either C or L (respectively).
    """
    w = 2*_np.pi*f
    #C Given in ohms, return as Z
    if (C!=None):
        Z = -1/(w*C)
    #L Given in ohms, return as Z
    if (L!=None):
        Z = w*L
    #If asked for imaginary number
    if (complex):
        Z *= 1j
    return(Z)

# Define Parallel Impedance Adder
def parallelz(*args):
    """
    Parallel Impedance Calculator
    
    This function is designed to generate the total parallel
    impedance of a set (tuple) of impedances specified as real
    or complex values.
    
    .. math::
       Z_{eq}=(\\frac{1}{Z_1}+\\frac{1}{Z_2}+\\dots+\\frac{1}{Z_n})^{-1}
    
    Parameters
    ----------
    Z:      tuple of complex
            The tupled input set of impedances, may be a tuple
            of any size greater than 2. May be real, complex, or
            a combination of the two.
    
    Returns
    -------
    Zp:     complex
            The calculated parallel impedance of the input tuple.
    """
    # Gather length (number of elements in tuple)
    L = len(args)
    if L==1:
        Z = args[0] # Only One Tuple Provided
        try:
            L = len(Z)
            if(L==1):
                Zp = Z[0] # Only one impedance, burried in tuple
            else:
                # Inversely add the first two elements in tuple
                Zp = (1/Z[0]+1/Z[1])**(-1)
                # If there are more than two elements, add them all inversely
                if(L > 2):
                    for i in range(2,L):
                        Zp = (1/Zp+1/Z[i])**(-1)
        except:
            Zp = Z # Only one impedance
    else:
        Z = args # Set of Args acts as Tuple
        # Inversely add the first two elements in tuple
        Zp = (1/Z[0]+1/Z[1])**(-1)
        # If there are more than two elements, add them all inversely
        if(L > 2):
            for i in range(2,L):
                Zp = (1/Zp+1/Z[i])**(-1)
    return(Zp)

# Define Phase/Line Converter
def phaseline(VLL=None,VLN=None,Iline=None,Iphase=None,complex=False):
    """
    Line-Line to Line-Neutral Converter
    
    This function is designed to return the phase- or line-equivalent
    of the voltage/current provided. It is designed to be used when
    converting delta- to wye-connections and vice-versa.
    Given a voltage of one type, this function will return the
    voltage of the opposite type. The same is true for current.
    
    .. math:: V_{LL} = \\sqrt{3}∠30° * V_{LN}
       :label: voltages
    
    Typical American (United States) standard is to note voltages in
    Line-to-Line values (VLL), and often, the Line-to-Neutral voltage
    is of value, this function uses the voltage :eq:`voltages` relation
    to evaluate either voltage given the other.
    
    .. math:: I_{Φ} = \\frac{I_{line}}{\\sqrt{3}∠-30°}
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
    complex:    bool, optional
                Control to return value in complex form; default=False
    """
    output = 0
    #Given VLL, convert to VLN
    if VLL is not None:
        VLN = VLL/(VLLcVLN)
        output = VLN
    #Given VLN, convert to VLL
    elif VLN is not None:
        VLL = VLN*VLLcVLN
        output = VLL
    #Given Iphase, convert to Iline
    elif Iphase is not None:
        Iline = Iphase*ILcIP
        output = Iline
    #Given Iline, convert to Iphase
    elif Iline is not None:
        Iphase = Iline/ILcIP
        output = Iphase
    #None given, error encountered
    else:
        print("ERROR: No value given"+
                "or innapropriate value"+
                "given.")
        return(0)
    #Return as complex only when requested
    if complex:
        return( output )
    return(abs( output ))

# Define Power Set Function
def powerset(P=None,Q=None,S=None,PF=None):
    """
    Power Triangle Conversion Function
    
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
    """
    #Given P and Q
    if (P!=None) and (Q!=None):
        S = _np.sqrt(P**2+Q**2)
        PF = P/S
        if Q<0:
            PF=-PF
    #Given S and PF
    elif (S!=None) and (PF!=None):
        P = abs(S*PF)
        Q = _np.sqrt(S**2-P**2)
        if PF<0:
            Q=-Q
    #Given P and PF
    elif (P!=None) and (PF!=None):
        S = P/PF
        Q = Q = _np.sqrt(S**2-P**2)
        if PF<0:
            Q=-Q
    else:
        raise ValueError("ERROR: Invalid Parameters or too few"+
                        " parameters given to calculate.")
    
    # Return Values!
    return(P,Q,S,PF)

# Define Power Triangle Function
def powertriangle(P=None,Q=None,S=None,PF=None,color="red",
                  text="Power Triangle",printval=False):
    """
    Power Triangle Plotting Function
    
    This function is designed to draw a power triangle given
    values for the complex power system.
    
    Parameters
    ----------
    P:          float
                Real Power, unitless; default=None
    Q:          float
                Reactive Power, unitless; default=None
    S:          float
                Apparent Power, unitless; default=None
    PF:         float
                Power Factor, unitless, provided as a
                decimal value, lagging is positive,
                leading is negative; default=None
    color:      string, optional
                The color of the power triangle lines;
                default="red"
    text:       string, optional
                The title of the power triangle plot,
                default="Power Triangle"
    printval:   bool, optional
                Control argument to allow the numeric
                values to be printed on the plot,
                default="False"
    """
    # Calculate all values if not all are provided
    if( P==None or Q==None or S==None or PF==None):
        P,Q,S,PF = powerset(P,Q,S,PF)

    #Generate Lines
    Plnx = [0,P]
    Plny = [0,0]
    Qlnx = [P,P]
    Qlny = [0,Q]
    Slnx = [0,P]
    Slny = [0,Q]

    # Plot Power Triangle
    _plt.figure(1)
    _plt.title(text)
    _plt.plot(Plnx,Plny,color=color)
    _plt.plot(Qlnx,Qlny,color=color)
    _plt.plot(Slnx,Slny,color=color)
    _plt.xlabel("Real Power (W)")
    _plt.ylabel("Reactive Power (VAR)")
    mx = max(abs(P),abs(Q))

    if P>0:
        _plt.xlim(0,mx*1.1)
        x=mx
    else:
        _plt.xlim(-mx*1.1,0)
        x=-mx
    if Q>0:
        _plt.ylim(0,mx*1.1)
        y=mx
    else:
        _plt.ylim(-mx*1.1,0)
        y=-mx
    if PF > 0:
        PFtext = " Lagging"
    else:
        PFtext = " Leading"
    text = "P:   "+str(P)+" W\n"
    text = text+"Q:   "+str(Q)+" VAR\n"
    text = text+"S:   "+str(S)+" VA\n"
    text = text+"PF:  "+str(abs(PF))+PFtext+"\n"
    text = text+"ΘPF: "+str(_np.degrees(_np.arccos(PF)))+"°"+PFtext
    # Print all values if asked to
    if printval:
         _plt.text(x/20,y*4/5,text,color=color)
    _plt.show()

# Define Transformer Short-Circuit/Open-Circuit Function
def transformertest(Poc=False,Voc=False,Ioc=False,Psc=False,Vsc=False,
               Isc=False):
    """
    Transformer Rated Test Evaluator
    
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
    if (Poc!=None) and (Voc!=None) and (Ioc!=None):
        PF = Poc/(Voc*Ioc)
        Y = _c.rect(Ioc/Voc,-_np.arccos(PF))
        Rc = 1/Y.real
        Xm = -1/Y.imag
        OC = True
    # Given Short-Circuit Values
    if (Psc!=None) and (Vsc!=None) and (Isc!=None):
        PF = Psc/(Vsc*Isc)
        Zeq = _c.rect(Vsc/Isc,_np.arccos(PF))
        Req = Zeq.real
        Xeq = Zeq.imag
        SC = True
    # Return All if Found
    if OC and SC:
        return(Req,Xeq,Rc,Xm)
    elif OC:
        return(Rc,Xm)
    elif SC:
        return(Req,Xeq)
    else:
        print("An Error Was Encountered.\n"+
                "Not enough arguments were provided.")

# Define Phasor Plot Generator
def phasorplot(phasor,title="Phasor Diagram",legend=False,bg=None,
               radius=None,colors=None):
    """
    Phasor Plotting Function
    
    This function is designed to plot a phasor-diagram with angles in degrees
    for up to 12 phasor sets. Phasors must be passed as a complex number set,
    (e.g. [ m+ja, m+ja, m+ja, ... , m+ja ] ).
    
    Parameters
    ----------
    phasor:     list of complex
                The set of phasors to be plotted.
    title:      string, optional
                The Plot Title, default="Phasor Diagram"
    legend:     bool, optional
                Control argument to enable displaying the legend, must be passed
                as an array or list of strings, default=False
    bg:         string, optional
                Background-Color control, default="#d5de9c"
    radius:     float, optional
                The diagram radius, unless specified, automatically scales
    colors:     list of str, optional
                List of hexidecimal color strings denoting the line colors to use.
    """
    # Manage Colors
    if colors==None:
        colors = ["#FF0000","#800000","#FFFF00","#808000","#00ff00","#008000",
                  "#00ffff","#008080","#0000ff","#000080","#ff00ff","#800080"]
    # Scale Radius
    if radius==None:
        radius = _np.abs(phasor).max()
    # Set Background Color
    if bg==None:
        bg = "#FFFFFF"
    # Check for more phasors than colors
    numphs = len(phasor)
    numclr = len(colors)
    if numphs > numclr:
        raise ValueError("ERROR: Too many phasors provided. Specify more line colors.")
    
    # Force square figure and square axes looks better for polar, IMO
    width, height = _matplotlib.rcParams['figure.figsize']
    size = min(width, height)
    # Make a square figure
    fig = _plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, facecolor=bg)
    ax.set_rmax(radius)
    _plt.grid(True)
    
    # Plot the diagram
    _plt.title(title+"\n")
    handles=_np.array([]) # Empty array for plot handles
    for i in range(numphs):
        mag, ang_r = _c.polar(phasor[i])
        if legend!=False:
            hand = _plt.arrow(0,0,ang_r,mag,color=colors[i],label=legend[i])
            handles = _np.append(handles,[hand])
        else: _plt.arrow(0,0,ang_r,mag,color=colors[i])
    if legend!=False: _plt.legend((handles),legend)
    _plt.show()

# Define Non-Linear Power Factor Calculator
def nlinpf(PFtrue=False,PFdist=False,PFdisp=False):
    """
    Non-Linear Power Factor Evaluator
    
    This function is designed to evaluate one of three unknowns
    given the other two. These particular unknowns are the arguments
    and as such, they are described in the representative sections
    below.
    
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
    {unknown}:  This function will return the unknown variable from
                the previously described set of variables.
    """
    if(PFtrue!=None and PFdist!=None and PFdisp!=None):
        raise ValueError("ERROR: Too many constraints, no solution.") 
    elif ( PFdist!=None and PFdisp!=None ):
        return( PFdist * PFdisp )
    elif ( PFtrue!=None and PFdisp!=None ):
        return( PFtrue / PFdisp )
    elif ( PFtrue!=None and PFdist!=None ):
        return( PFtrue / PFdist )
    else:
        raise ValueError("ERROR: Function requires at least two arguments.")

# Define Short-Circuit RL Current Calculator
def iscrl(V,Z,t=None,f=None,mxcurrent=True,alpha=None):
    """
    Short-Circuit-Current (ISC) Calculator
    
    The Isc-RL function (Short Circuit Current for RL Circuit)
    is designed to calculate the short-circuit current for an
    RL circuit.
    
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
    if(f!=None): omega = 2*_np.pi*f
    else: omega = None
    R = abs(Z.real)
    X = abs(Z.imag)
    theta = _np.arctan( X/R )
    
    # If Maximum Current is Desired and No alpha provided
    if(mxcurrent and alpha==None):
        alpha = theta - _np.pi/2
    elif(mxcurrent and alpha!=None):
        raise ValueError("ERROR: Inappropriate Arguments Provided.\n"+
                         "Not both mxcurrent and alpha can be provided.")
    
    # Calculate Asymmetrical (total) Current if t != None
    if(t!=None and f!=None):
        # Calculate RMS if none of the angular values are provided
        if(alpha==None and omega==None):
            # Calculate tau
            tau = t/(1/60)
            K = _np.sqrt(1 + 2*_np.exp(-4*_np.pi*tau/(X/R)) )
            IAC = abs(V/Z)
            Irms = K*IAC
            # Return Values
            return(Irms,IAC,K)
        elif(alpha==None or omega==None):
            raise ValueError("ERROR: Inappropriate Arguments Provided.")
        # Calculate Instantaneous if all angular values provided
        else:
            # Convert Degrees to Radians
            omega = _np.radians(omega)
            alpha = _np.radians(alpha)
            theta = _np.radians(theta)
            # Calculate T
            T = X/(2*_np.pi*f*R) # seconds
            # Calculate iAC and iDC
            iAC = _np.sqrt(2)*V/Z*_np.sin(omega*t+alpha-theta)
            iDC = -_np.sqrt(2)*V/Z*_np.sin(alpha-theta)*_np.exp(-t/T)
            i = iAC + iDC
            # Return Values
            return(i,iAC,iDC,T)
    elif( (t!=None and f==None) or (t==None and f!=None) ):
        raise ValueError("ERROR: Inappropriate Arguments Provided.\n"+
                         "Must provide both t and f or neither.")
    else:
        IAC = abs(V/Z)
        return(Iac)

# Define Voltage Divider Calculator
def voltdiv(Vin,R1,R2,Rload=None):
    """
    Voltage Divider Function
    
    This function is designed to calculate the output
    voltage of a voltage divider given the input voltage,
    the resistances (or impedances) and the load resistance
    (or impedance) if present.
    
    .. math:: V_{out} = V_{in} * \\frac{R_2}{R_1+R+2}
    
    .. math:: V_{out}=V_{in}*\\frac{R_2||R_{load}}{R_1+(R_2||R_{load})}
    
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
    """
    # Determine whether Rload is given
    if(Rload==None): # No Load Given
        Vout = Vin * R2 / (R1+R2)
    else:   # Load was given
        Rp = parallelz((R2,Rload))
        Vout = Vin * Rp / (R1+Rp)
    return(Vout)

# Define Current Divider Calculator
def curdiv(Ri,Rset,Vin=None,Iin=None,Vout=False):
    """
    Current Divider Function
    
    This function is disigned to accept the input current, or input
    voltage to a resistor (or impedance) network of parallel resistors
    (impedances) and calculate the current through a particular element.
    
    Parameters
    ----------
    Ri:     float
            The Particular Resistor of Interest, should not be included in
            the tuple passed to Rset.
    Rset:   float
            Tuple of remaining resistances (impedances) in network.
    Vin:    float, optional
            The input voltage for the system, default=None
    Iin:    float, optional
            The input current for the system, default=None
    Vout:   bool, optional
            Control Argument to enable return of the voltage across the
            resistor (impecance) of interest (Ri)
    
    Returns
    -------
    Opt1 - Ii:          The Current through the resistor (impedance) of interest
    Opt2 - (Ii,Vi):     The afore mentioned current, and voltage across the
                        resistor (impedance) of interest
    """
    # Validate Tuple
    if not isinstance(Rset,tuple):
        Rset = (Rset,) # Set as Tuple
    # Calculate The total impedance
    Rtot = parallelz( Rset + (Ri,) ) # Combine tuples, then calculate total resistance
    # Determine Whether Input was given as Voltage or Current
    if(Vin!=None and Iin==None): # Vin Provided
        Iin = Vin / Rtot # Calculate total current
        Ii = Iin * Rtot/Ri # Calculate the current of interest
    elif(Vin==None and Iin!=None): # Iin provided
        Ii = Iin * Rtot/Ri # Calculate the current of interest
    else:
        raise ValueError("ERROR: Too many or too few constraints provided.")
    if(Vout): # Asked for voltage across resistor of interest
        Vi = Ii * Ri
        return(Ii, Vi)
    else:
        return(Ii)

# Define Instantaneous Power Calculator
def instpower(P,Q,t,f=60):
    """
    Instantaneous Power Function
    
    This function is designed to calculate the instantaneous power at a
    specified time t given the magnitudes of P and Q.
    
    .. math:: P_{inst} = P+P*cos(2*\\omega*t)-Q*sin(2*\\omega*t)
    
    Parameters
    ----------
    P:  float
        Magnitude of Real Power
    Q:  float
        Magnitude of Reactive Power
    t:  float
        Time at which to evaluate
    f:  float, optional
        System frequency (in Hz), default=60
    
    Returns
    -------
    Pinst:  float
            Instantaneous Power at time t
    """
    # Evaluate omega
    w = 2*_np.pi*f
    # Calculate
    Pinst = P + P*_np.cos(2*w*t) - Q*_np.sin(2*w*t)
    return(Pinst)

# Define Delta-Wye Impedance Network Calculator
def dynetz(delta=None,wye=None,round=None):
    """
    Delta-Wye Impedance Converter
    
    This function is designed to act as the conversion utility
    to transform delta-connected impedance values to wye-
    connected and vice-versa.
    
    .. math:: 
       Z_{sum} = Z_{1/2} + Z_{2/3} + Z_{3/1}//
       Z_1 = \\frac{Z_{1/2}*Z_{3/1}}{Z_{sum}}//
       Z_2 = \\frac{Z_{1/2}*Z_{2/3}}{Z_{sum}}//
       Z_3 = \\frac{Z_{2/3}*Z_{3/1}}{Z_{sum}}
    
    .. math::
       Z_{ms} = Z_1*Z_2 + Z_2*Z_3 + Z_3*Z_1//
       Z_{2/3} = \\frac{Z_{ms}}{Z_1}//
       Z_{3/1} = \\frac{Z_{ms}}{Z_2}//
       Z_{1/2} = \\frac{Z_{ms}}{Z_3}
    
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
    if(delta!=None and wye==None):
        Z12, Z23, Z31 = delta # Gather particular impedances
        Zsum = Z12 + Z23 + Z31 # Find Sum
        # Calculate Wye Impedances
        Z1 = Z12*Z31 / Zsum
        Z2 = Z12*Z23 / Zsum
        Z3 = Z23*Z31 / Zsum
        Zset = ( Z1, Z2, Z3 )
        if round!=None: Zset = _np.around(Zset,round)
        return(Zset) # Return Wye Impedances
    elif(delta==None and wye!=None):
        Z1, Z2, Z3 = wye # Gather particular impedances
        Zmultsum = Z1*Z2 + Z2*Z3 + Z3*Z1
        Z23 = Zmultsum / Z1
        Z31 = Zmultsum / Z2
        Z12 = Zmultsum / Z3
        Zset = ( Z12, Z23, Z31 )
        if round!=None: Zset = _np.around(Zset,round)
        return(Zset) # Return Delta Impedances
    else:
        raise ValueError("ERROR: Either delta or wye impedances must be specified.")

# Define Single Line Power Flow Calculator
def powerflow( Vsend, Vrec, Zline ):
    """
    Simple Power-Flow Calculator
    
    This function is designed to calculate the ammount of real
    power transferred from the sending end to the recieving end
    of an electrical line given the sending voltage (complex),
    the receiving voltage (complex) and the line impedance.
    
    .. math::
       P_{flow}=\\frac{|V_{send}|*|V_{rec}|}{Z_{line}}*sin(\\theta_{send}
       -\\theta_{rec})
    
    Parameters
    ----------
    Vsend:      complex
                The sending-end voltage, should be complex
    Vrec:       complex
                The receiving-end voltage, should be complex
    Zline:      complex
                The line impedance, should be complex
    
    Returns
    -------
    pflow:      complex
                The power transferred from sending-end to
                receiving-end, positive values denote power
                flow from send to receive, negative values
                denote vice-versa.
    """
    # Evaluate the Input Terms
    Vs = abs( Vsend )
    ds = _c.phase( Vsend )
    Vr = abs( Vrec )
    dr = _c.phase( Vrec )
    # Calculate Power Flow
    pflow = (Vs * Vr)/(Zline) * _np.sin( ds-dr )
    return( pflow )

# Define Impedance From Power and X/R
def zsource(S,V,XoR,Sbase=None,Vbase=None,perunit=True):
    """
    Source Impedance Calculator
    
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
    if Vbase == None:
        Vbase = V
    if Sbase == None:
        Sbase = S
    # Prevent scaling if per-unit already applied
    if Vbase == True:
        Vbase = 1
    if Sbase == True:
        Sbase = 1
    # Set to per-unit
    Spu = S/Sbase
    Vpu = V/Vbase
    # Evaluate Zsource Magnitude
    Zsource_pu = Vpu**2/Spu
    # Evaluate the angle
    nu = _np.degrees(_np.arctan(XoR))
    # Conditionally Evaluate Phasor Impedance
    if isinstance(nu, (list,_np.ndarray)):
        Zsource_pu = []
        for angle in nu:
            Zsource_pu.append(phasor(Zsource_pu, angle))
    else:
        Zsource_pu = phasor(Zsource_pu, nu)
    if not perunit:
        Zsource = Zsource_pu * Vbase**2/Sbase
        return(Zsource)
    return(Zsource_pu)

# Define Impedance Decomposer
def zdecompose(Zmag,XoR):
    """
    Impedance Decomposition Function
    
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
    R = Zmag/_np.sqrt(XoR**2+1)
    # Evaluate Reactance
    X = R * XoR
    # Return
    return(R,X)

# Define HP to Watts Calculation
def hp_to_watts(hp):
    """
    Horsepower to Watts Formula
    
    Calculates the power (in watts) given the
    horsepower.
    
    Parameters
    ----------
    hp:         float
                The horspower to compute.
    
    Returns
    -------
    watts:      float
                The power in watts.
    """
    return(hp * 745.699872)

# Define Watts to HP Calculation
def watts_to_hp(watts):
    """
    Watts to Horsepower Function
    
    Calculates the power (in horsepower) given
    the power in watts.
    
    Parameters
    ----------
    watts:      float
                The wattage to compute.
    
    Returns
    -------
    hp:         float
                The power in horsepower.
    """
    return(watts / 745.699872)
    
# Define Power Reactance Calculator
def powerimpedance(S,V,PF=None,parallel=False,terms=False):
    """
    Impedance from Apparent Power Formula
    
    Function to determine the ohmic resistance/reactance
    (impedance) represented by the apparent power (S).
    
    .. math:: Z = \\frac{V^2}{S}
       :label: series
    
    .. math:: Z = \\frac{V^2}{(3*S)}
       :label: parallel
    
    This function can evaluate the component values for
    both series :eq:`series` and parallel :eq:`parallel`
    connected circuits.
    
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
    if isinstance(S,complex) or PF != None:
        if PF != None:
            # Evaluate Elements
            P,Q,S,PF = powerset(S=S,PF=PF)
        else:
            P = S.real
            Q = S.imag
        # Compute Elements
        if parallel:
            R = V**2 / (3*P)
            X = V**2 / (3*Q)
        else:
            R = V**2 / (P)
            X = V**2 / (Q)
        # Conditionally Return as Impedance
        if terms:
            return( R, X )
        return( R + 1j*X )
    # Not Complex (just R)
    R = V**2 / S
    return( R )

# Define Cold-Junction-Voltage Calculator
def coldjunction(Tcj,coupletype="K",To=None,Vo=None,P1=None,P2=None,
                 P3=None,P4=None,Q1=None,Q2=None,round=None):
    """
    Thermocouple Cold-Junction Formula
    
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
    Q3:         float, optional
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
        if not (0 < Tcj and Tcj < 70):
            raise ValueError("Temperature out of range.")
    else:
        if not (-20 < Tcj and Tcj < 70):
            raise ValueError("Temperature out of range.")
    # Define Constant Lookup System
    lookup = ["B","E","J","K","N","R","S","T"]
    if not (coupletype in lookup):
        raise ValueError("Invalid Thermocouple Type")
    index = lookup.index(coupletype)
    # Define Constant Dictionary
    constants = {   "To" : [4.2000000E+01,2.5000000E+01,2.5000000E+01,2.5000000E+01,7.0000000E+00,2.5000000E+01,2.5000000E+01,2.5000000E+01],
                    "Vo" : [3.3933898E-04,1.4950582E+00,1.2773432E+00,1.0003453E+00,1.8210024E-01,1.4067016E-01,1.4269163E-01,9.9198279E-01],
                    "P1" : [2.1196684E-04,6.0958443E-02,5.1744084E-02,4.0514854E-02,2.6228256E-02,5.9330356E-03,5.9829057E-03,4.0716564E-02],
                    "P2" : [3.3801250E-06,-2.7351789E-04,-5.4138663E-05,-3.8789638E-05,-1.5485539E-04,2.7736904E-05,4.5292259E-06,7.1170297E-04],
                    "P3" : [-1.4793289E-07,-1.9130146E-05,-2.2895769E-06,-2.8608478E-06,2.1366031E-06,-1.0819644E-06,-1.3380281E-06,6.8782631E-07],
                    "P4" : [-3.3571424E-09,-1.3948840E-08,-7.7947143E-10,-9.5367041E-10,9.2047105E-10,-2.3098349E-09,-2.3742577E-09,4.3295061E-11],
                    "Q1" : [-1.0920410E-02,-5.2382378E-03,-1.5173342E-03,-1.3948675E-03,-6.4070932E-03,2.6146871E-03,-1.0650446E-03,1.6458102E-02],
                    "Q2" : [-4.9782932E-04,-3.0970168E-04,-4.2314514E-05,-6.7976627E-05,8.2161781E-05,-1.8621487E-04,-2.2042420E-04,0.0000000E+00]
                }
    # Load Data Into Terms
    if To == None:
        To = constants["To"][index]
    if Vo == None:
        Vo = constants["Vo"][index]
    if P1 == None:
        P1 = constants["P1"][index]
    if P2 == None:
        P2 = constants["P2"][index]
    if P3 == None:
        P3 = constants["P3"][index]
    if P4 == None:
        P4 = constants["P4"][index]
    if Q1 == None:
        Q1 = constants["Q1"][index]
    if Q2 == None:
        Q2 = constants["Q2"][index]
    # Define Formula Terms
    tx = (Tcj-To)
    num = tx*(P1+tx*(P2+tx*(P3+P4*tx)))
    den = 1+tx*(Q1+Q2*tx)
    Vcj = Vo + num/den
    # Round Value if Allowed
    if round != None:
        Vcj = _np.around(Vcj, round)
    # Return in milivolts
    return(Vcj*m)
    
# Define Thermocouple Temperature Calculation
def thermocouple(V,coupletype="K",fahrenheit=False,cjt=None,To=None,Vo=None,P1=None,
                 P2=None,P3=None,P4=None,Q1=None,Q2=None,Q3=None,round=1):
    """
    Thermocouple Temperature Calculator
    
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
    V = V/m # Scale volts to milivolts
    # Determine Cold-Junction-Voltage
    if cjt != None:
        Vcj = coldjunction(cjt,coupletype,To,Vo,P1,P2,P3,P4,Q1,Q2,round)
        V += Vcj/m
    # Define Constant Lookup System
    lookup = ["B","E","J","K","N","R","S","T"]
    if not (coupletype in lookup):
        raise ValueError("Invalid Thermocouple Type")
    # Define Voltage Ranges
    voltages = {"J" : [-8.095, 0,      21.840, 45.494, 57.953, 69.553],
                "K" : [-6.404, -3.554, 4.096,  16.397, 33.275, 69.553],
                "B" : [0.291,  2.431,  13.820, None,   None,   None],
                "E" : [-9.835, -5.237, 0.591,  24.964, 53.112, 76.373],
                "N" : [-4.313, 0,      20.613, 47.513, None,   None],
                "R" : [-0.226, 1.469,  7.461,  14.277, 21.101, None],
                "S" : [-0.236, 1.441,  6.913,  12.856, 18.693, None],
                "T" : [-6.18,  -4.648, 0,      9.288,  20.872, None]}
    # Determine Array Selection
    vset = voltages[coupletype]
    if V < vset[0]*m:
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
    # Define Dictionary of Arrays
    data = {"J" : [[-6.4936529E+01,2.5066947E+02,6.4950262E+02,9.2510550E+02,1.0511294E+03],
                   [-3.1169773E+00,1.3592329E+01,3.6040848E+01,5.3433832E+01,6.0956091E+01],
                   [2.2133797E+01,1.8014787E+01,1.6593395E+01,1.6243326E+01,1.7156001E+01],
                   [2.0476437E+00,-6.5218881E-02,7.3009590E-01,9.2793267E-01,-2.5931041E+00],
                   [-4.6867532E-01,-1.2179108E-02,2.4157343E-02,6.4644193E-03,-5.8339803E-02],
                   [-3.6673992E-02,2.0061707E-04,1.2787077E-03,2.0464414E-03,1.9954137E-02],
                   [1.1746348E-01,-3.9494552E-03,4.9172861E-02,5.2541788E-02,-1.5305581E-01],
                   [-2.0903413E-02,-7.3728206E-04,1.6813810E-03,1.3682959E-04,-2.9523967E-03],
                   [-2.1823704E-03,1.6679731E-05,7.6067922E-05,1.3454746E-04,1.1340164E-03]],
            "K" : [[-1.2147164E+02,-8.7935962E+00,3.1018976E+02,6.0572562E+02,1.0184705E+03],
                   [-4.1790858E+00,-3.4489914E-01,1.2631386E+01,2.5148718E+01,4.1993851E+01],
                   [3.6069513E+01,2.5678719E+01,2.4061949E+01,2.3539401E+01,2.5783239E+01],
                   [3.0722076E+01,-4.9887904E-01,4.0158622E+00,4.6547228E-02,-1.8363403E+00],
                   [7.7913860E+00,-4.4705222E-01,2.6853917E-01,1.3444400E-02,5.6176662E-02],
                   [5.2593991E-01,-4.4869203E-02,-9.7188544E-03,5.9236853E-04,1.8532400E-04],
                   [9.3939547E-01,2.3893439E-04,1.6995872E-01,8.3445513E-04,-7.4803355E-02],
                   [2.7791285E-01,-2.0397750E-02,1.1413069E-02,4.6121445E-04,2.3841860E-03],
                   [2.5163349E-02,-1.8424107E-03,-3.9275155E-04,2.5488122E-05,0.0]],
            "B" : [[5.0000000E+02,1.2461474E+03],
                   [1.2417900E+00,7.2701221E+00],
                   [1.9858097E+02,9.4321033E+01],
                   [2.4284248E+01,7.3899296E+00],
                   [-9.7271640E+01,-1.5880987E-01],
                   [-1.5701178E+01,1.2681877E-02],
                   [3.1009445E-01,1.0113834E-01],
                   [-5.0880251E-01,-1.6145962E-03],
                   [-1.6163342E-01,-4.1086314E-06]],
            "E" : [[-1.1721668E+02,-5.0000000E+01,2.5014600E+02,6.0139890E+02,8.0435911E+02],
                   [-5.9901698E+00,-2.7871777E+00,1.7191713E+01,4.5206167E+01,6.1359178E+01],
                   [2.3647275E+01,1.9022736E+01,1.3115522E+01,1.2399357E+01,1.2759508E+01],
                   [1.2807377E+01,-1.7042725E+00,1.1780364E+00,4.3399963E-01,-1.1116072E+00],
                   [2.0665069E+00,-3.5195189E-01,3.6422433E-02,9.1967085E-03,3.5332536E-02],
                   [8.6513472E-02,4.7766102E-03,3.9584261E-04,1.6901585E-04,3.3080380E-05],
                   [5.8995860E-01,-6.5379760E-02,9.3112756E-02,3.4424680E-02,-8.8196889E-02],
                   [1.0960713E-01,-2.1732833E-02,2.9804232E-03,6.9741215E-04,2.8497415E-03],
                   [6.1769588E-03,0.0,3.3263032E-05,1.2946992E-05,0.0]],
            "N" : [[-5.9610511E+01,3.1534505E+02,1.0340172E+03],
                   [-1.5000000E+00,9.8870997E+00,3.7565475E+01],
                   [4.2021322E+01,2.7988676E+01,2.6029492E+01],
                   [4.7244037E+00,1.5417343E+00,-6.0783095E-01],
                   [-6.1153213E+00,-1.4689457E-01,-9.7742562E-03],
                   [-9.9980337E-01,-6.8322712E-03,-3.3148813E-06],
                   [1.6385664E-01,6.2600036E-02,-2.5351881E-02],
                   [-1.4994026E-01,-5.1489572E-03,-3.8746827E-04],
                   [-3.0810372E-02,-2.8835863E-04,1.7088177E-06]],
            "R" : [[1.3054315E+02,5.4188181E+02,1.0382132E+03,1.5676133E+03],
                   [8.8333090E-01,4.9312886E+00,1.1014763E+01,1.8397910E+01],
                   [1.2557377E+02,9.0208190E+01,7.4669343E+01,7.1646299E+01],
                   [1.3900275E+02,6.1762254E+00,3.4090711E+00,-1.0866763E+00],
                   [3.3035469E+01,-1.2279323E+00,-1.4511205E-01,-2.0968371E+00],
                   [-8.5195924E-01,1.4873153E-02,6.3077387E-03,-7.6741168E-01],
                   [1.2232896E+00,8.7670455E-02,5.6880253E-02,-1.9712341E-02],
                   [3.5603023E-01,-1.2906694E-02,-2.0512736E-03,-2.9903595E-02],
                   [0.0,0.0,0.0,-1.0766878E-02]],
            "S" : [[1.3792630E+02,4.7673468E+02,9.7946589E+02,1.6010461E+03],
                   [9.3395024E-01,4.0037367E+00,9.3508283E+00,1.6789315E+01],
                   [1.2761836E+02,1.0174512E+02,8.7126730E+01,8.4315871E+01],
                   [1.1089050E+02,-8.9306371E+00,-2.3139202E+00,-1.0185043E+01],
                   [1.9898457E+01,-4.2942435E+00,-3.2682118E-02,-4.6283954E+00],
                   [9.6152996E-02,2.0453847E-01,4.6090022E-03,-1.0158749E+00],
                   [9.6545918E-01,-7.1227776E-02,-1.4299790E-02,-1.2877783E-01],
                   [2.0813850E-01,-4.4618306E-02,-1.2289882E-03,-5.5802216E-02],
                   [0.0,1.6822887E-03,0.0,-1.2146518E-02]],
            "T" : [[-1.9243000E+02,-6.0000000E+01,1.3500000E+02,3.0000000E+02],
                   [-5.4798963E+00,-2.1528350E+00,5.9588600E+00,1.4861780E+01],
                   [5.9572141E+01,3.0449332E+01,2.0325591E+01,1.7214707E+01],
                   [1.9675733E+00,-1.2946560E+00,3.3013079E+00,-9.3862713E-01],
                   [-7.8176011E+01,-3.0500735E+00,1.2638462E-01,-7.3509066E-02],
                   [-1.0963280E+01,-1.9226856E-01,-8.2883695E-04,2.9576140E-04],
                   [2.7498092E-01,6.9877863E-03,1.7595577E-01,-4.8095795E-02],
                   [-1.3768944E+00,-1.0596207E-01,7.9740521E-03,-4.7352054E-03],
                   [-4.5209805E-01,-1.0774995E-02,0.0,0.0]]}
    # Load Data Into Terms
    if To == None:
        To = data[coupletype][0][select]
    if Vo == None:
        Vo = data[coupletype][1][select]
    if P1 == None:
        P1 = data[coupletype][2][select]
    if P2 == None:
        P2 = data[coupletype][3][select]
    if P3 == None:
        P3 = data[coupletype][4][select]
    if P4 == None:
        P4 = data[coupletype][5][select]
    if Q1 == None:
        Q1 = data[coupletype][6][select]
    if Q2 == None:
        Q2 = data[coupletype][7][select]
    if Q3 == None:
        Q3 = data[coupletype][8][select]
    # Calculate Temperature in Degrees C
    num = (V-Vo)*(P1+(V-Vo)*(P2+(V-Vo)*(P3+P4*(V-Vo))))
    den = 1 + (V-Vo)*(Q1+(V-Vo)*(Q2+Q3*(V-Vo)))
    temp = To + num/den
    # Return Temperature
    if fahrenheit:
        temp = (temp*9/5)+32
    temp = _np.around(temp,round)
    return(temp)

# Define RTD Calculator
def rtdtemp(RT,rtdtype="PT100",fahrenheit=False,Rref=None,Tref=None,
            a=None,round=1):
    """
    RTD Temperature Calculator
    
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
    # Define list of available builtin RTD Types
    types = {   "PT100" : [100,0.00385],
                "PT1000": [1000,0.00385],
                "CU100" : [100,0.00427],
                "NI100" : [100,0.00618],
                "NI120" : [120,0.00672],
                "NIFE"  : [604,0.00518]
            }
    # Load Variables
    if Rref==None:
        Rref = types[rtdtype][0]
    if Tref==None:
        Tref = 0
    if a==None:
        a = types[rtdtype][1]
    # Define Terms
    num = RT - Rref + Rref*a*Tref
    den = Rref*a
    temp = num/den
    # Return Temperature
    if fahrenheit:
        temp = (temp*9/5)+32
    temp = _np.around(temp,round)
    return(temp)
    
# Define Capacitor Voltage Discharge Function
def vcapdischarge(t,Vs,R,C):
    """
    Discharging Capacitor Function
    
    Function to calculate the voltage of a
    capacitor that is discharging given the time.
    
    .. math:: V_c=V_s*e^{\\frac{-t}{R*C}}
    
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
    Vc = Vs*(_np.exp(-t/(R*C)))
    return(Vc)

# Define Capacitor Voltage Charge Function
def vcapcharge(t,Vs,R,C):
    """
    Charging Capacitor Voltage
    
    Function to calculate the voltage of a
    capacitor that is charging given the time.
    
    .. math:: V_c=V_s*(1-e^{\\frac{-t}{R*C}})
    
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
    Vc = Vs*(1-_np.exp(-t/(R*C)))
    return(Vc)
    
# Define Capacitive Energy Transfer Function
def captransfer(t,Vs,R,Cs,Cd):
    """
    Capacitor Energy Transfer Function
    
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
    tau = (R*Cs*Cd) / (Cs+Cd)
    rvolt = Vs*_np.exp(-t/tau)
    vfinal = Vs*Cs/(Cs+Cd)
    return(rvolt,vfinal)
    
# Define Inductor Energy Formula
def inductorenergy(L,I):
    """
    Energy Stored in Inductor Formula
    
    Function to calculate the energy stored in an inductor
    given the inductance (in Henries) and the current.
    
    .. math:: E=\\frac{1}{2}*L*I^2
    
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
    return(1/2 * L * I**2)

# Define Inductor Charge Function
def inductorcharge(t,Vs,R,L):
    """
    Charging Inductor Formula
    
    Calculates the Voltage and Current of an inductor
    that is charging/storing energy.
    
    .. math::
       V_L = V_s*e^{\\frac{-R*t}{L}}//
       I_L = \\frac{V_s}{R}*(1-e^{\\frac{-R*t}{L}})
    
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
    Vl = Vs*_np.exp(-R*t/L)
    Il = Vs/R*(1-_np.exp(-R*t/L))
    return(Vl,Il)

# Define Capacitive Back-to-Back Switching Formula
def capbacktoback(C1,C2,Lm,VLN=None,VLL=None):
    """
    Back to Back Capacitor Transient Current Calculator
    
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
    imax = _np.sqrt(2/3)*VLL*_np.sqrt((C1*C2)/((C1+C2)*Lm))
    # Evaluate Inrush Current Frequency
    ifreq = 1/(2*_np.pi*_np.sqrt(Lm*(C1*C2)/(C1+C2)))
    return(imax,ifreq)

# Define Inductor Discharge Function
def inductordischarge(t,Io,R,L):
    """
    Discharging Inductor Formula
    
    Calculates the Voltage and Current of an inductor
    that is discharging its stored energy.
    
    .. math::
       I_L=I_0*e^{\\frac{-R*t}{L}}//
       V_L=I_0*R*(1-e^{\\frac{-R*t}{L}})
    
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
    Il = Io*_np.exp(-R*t/L)
    Vl = Io*R*(1-_np.exp(-R*t/L))
    return(Vl,Il)
    
# Define Apparent Power to Farad Conversion
def farads(VAR,V,freq=60):
    """
    Capacitance from Apparent Power Formula
    
    Function to calculate the required capacitance
    in Farads to provide the desired power rating
    (VARs).
    
    .. math:: C = \\frac{VAR}{2*\\pi*freq*V^2}
    
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
    return(VAR / (2*_np.pi*freq*V**2))

# Define Capacitor Energy Calculation
def capenergy(C,v):
    """
    Capacitor Energy Formula
    
    A simple function to calculate the stored voltage (in Joules)
    in a capacitor with a charged voltage.
    
    .. math:: E=\\frac{1}{2}*C*V^2
    
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
    energy = 1/2 * C * V**2
    return(energy)

# Define Capacitor Voltage Discharge Function
def loadedvcapdischarge(t,vo,C,P):
    """
    Loaded Capacitor Discharge Formula
    
    Returns the voltage of a discharging capacitor after time (t - 
    seconds) given initial voltage (vo - volts), capacitor size
    (cap - Farads), and load (P - Watts).
    
    .. math:: V_t=\\sqrt{v_0^2-2*P*\\frac{t}{C}}
    
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
    Vt = _np.sqrt(vo**2 - 2*P*t/C)
    return(Vt)
    
# Define Capacitor Discharge Function
def timedischarge(Vinit,Vmin,C,P,dt=1e-3,RMS=True,Eremain=False):
    """
    Capacitor Discharge Time Formula
    
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
    t = 0 # start at time t=0
    if RMS:
        vo = Vinit*_np.sqrt(2) # convert RMS to peak
    else:
        vo = Vinit
    vc = loadedvcapdischarge(t,vo,C,P) # set initial cap voltage
    while(vc >= Vmin):
        t = t+dt # increment the time
        vcp = vc # save previous voltage
        vc = loadedvcapdischarge(t,vo,C,P) # calc. new voltage
    if(Eremain):
        E = energy(C,vcp) # calc. energy
        return(t-dt,E)
    else:
        return(t-dt)


# Define Rectifier Capacitor Calculator
def rectifiercap(Iload, fswitch, dVout):
    """
    Rectifier Capacitor Formula
    
    Returns the capacitance (in Farads) for a needed capacitor in
    a rectifier configuration given the system frequency (in Hz),
    the load (in amps) and the desired voltage ripple.
    
    .. math:: C=\\frac{I_{load}}{f_{switch}*\\Delta V_{out}}
    
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
    return(C)

# Define function to find VDC setpoint
def vscdcbus(VLL,Zs,P,Q=0,mmax=0.8,debug=False):
    """
    Voltage Sourced Converter DC Bus Voltage Function
    
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
            
    Return
    ------
    VDC:    float
            The DC bus voltage.
    """
    # Determine the Load Current
    Iload = _np.conj((P+1j*Q) / (VLL*_np.sqrt(3)))
    # Evaluate the Terminal Voltage
    Vtln = abs(VLL/_np.sqrt(3) + Iload*Zs)
    # Find the Peak Terminal Voltage
    Vtpk = _np.sqrt(2)*Vtln
    # Calculate the VDC value
    VDC = 2*Vtpk / mmax
    if debug:
        print("Iload", Iload)
        print("Vtln", Vtln)
        print("Vtpk", Vtpk)
        print("VDC", VDC)
    return(VDC)

# Define kp/ki/w0L calculating function
def vscgains(Rs,Ls,tau=0.005,f=60):
    """
    Voltage Sourced Converter Gains Calculator
    
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
    f:      float, optional
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
    ki = kp*Rs/Ls
    # Calculate w0L
    w0L = 2*_np.pi*f*Ls
    return(kp,ki,w0L)

# Define Convolution Bar-Graph Function:
def convbar(h, x, outline=True):
    """
    Convolution Bar-Graph Plotter Function
    
    Generates plots of each of two input arrays as bar-graphs, then
    generates a convolved bar-graph of the two inputs to demonstrate
    and illustrate convolution, typically for an educational purpose.
    
    Parameters
    ----------
    h:      numpy.ndarray
            Impulse Response - Given as Array (Prefferably Numpy Array)
    x:      numpy.ndarray
            Input Function - Given as Array (Prefferably Numpy Array)
    """
    
    # The impulse response
    M = len(h)
    t = _np.arange(M)
    # Plot
    _plt.subplot(121)
    if(outline): _plt.plot(t,h,color='red')
    _plt.bar(t,h,color='black')
    _plt.xticks([0,5,9])
    _plt.ylabel('h')
    _plt.title('Impulse Response')
    _plt.grid()

    # The input function
    N = len(x)
    s = _np.arange(N)
    # Plot
    _plt.subplot(122)
    if(outline): _plt.plot(s,x,color='red')
    _plt.bar(s,x,color='black')
    _plt.xticks([0,10,19])
    _plt.title('Input Function')
    _plt.grid()
    _plt.ylabel('x')

    # The output
    L = M+N-1
    w = _np.arange(L)
    _plt.figure(3)
    y = _np.convolve(h,x)
    if(outline): _plt.plot(w,y,color='red')
    _plt.bar(w,y,color='black')
    _plt.ylabel('y')
    _plt.grid()
    _plt.title('Convolved Output')
    _plt.show()


# Define convolution function
def convolve(tuple):
    """
    Filter Convolution Function
    
    Given a tuple of terms, convolves all terms in tuple to
    return one tuple as a numpy array.
    
    Parameters
    ---------
    tuple:      tuple of numpy.ndarray
                Tuple of terms to be convolved.
    
    Returns
    -------
    c:          The convolved set of the individual terms.
                i.e. numpy.ndarray([ x1, x2, x3, ..., xn ])
    """
    c = sig.convolve(tuple[0],tuple[1])
    if (len(tuple) > 2):
        # Iterate starting with second element and continuing
        for i in range(2,len(tuple)):
            c = sig.convolve(c,tuple[i])
    return(c)

# Define Step function
def step(t):
    """
    Step Function [ u(t) ]
    
    Simple implimentation of numpy.heaviside function
    to provide standard step-function as specified to
    be zero at x<0, and one at x>=0.
    """
    return( _np.heaviside( t, 1) )

# RMS Calculating Function
def rms(f, T):
    """
    Function Root-Mean-Square (RMS) Evaluator
    
    Integral-based RMS calculator, evaluates the RMS value
    of a repetative signal (f) given the signal's specific
    period (T)

    Parameters
    ----------
    f:      float
            The periodic function, a callable like f(t)
    T:      float
            The period of the function f, so that f(0)==f(T)

    Returns
    -------
    RMS:    The RMS value of the function (f) over the interval ( 0, T )

    """
    fn = lambda x: f(x)**2
    integral = integrate(fn,0,T)
    RMS = _np.sqrt(1/T*integral)
    return(RMS)

# Define Gaussian Function
def gaussian(x,mu=0,sigma=1):
    """
    Gaussian Function:
    
    This function is designed to generate the gaussian
    distribution curve with configuration mu and sigma.
    
    Parameters
    ----------
    x:      float
            The input (array) x.
    mu:     float, optional
            Optional control argument, default=0
    sigma:  float, optional
            Optional control argument, default=1
    
    Returns
    -------
    Computed gaussian (numpy.ndarray) of the input x
    """
    return( 1/(sigma * _np.sqrt(2 * _np.pi)) *
            _np.exp(-(x - mu)**2 / (2 * sigma**2)) )

# Define Gaussian Distribution Function
def gausdist(x,mu=0,sigma=1):
    """
    Gaussian Distribution Function:
    
    This function is designed to calculate the generic
    distribution of a gaussian function with controls
    for mu and sigma.
    
    Parameters
    ----------
    x:      numpy.ndarray
            The input (array) x
    mu:     float, optional
            Optional control argument, default=0
    sigma:  float, optional
            Optional control argument, default=1
    
    Returns
    -------
    F:      numpy.ndarray
            Computed distribution of the gausian function at the
            points specified by (array) x
    """
    F = _np.array([])
    try:
        lx = len(x) # Find length of Input
    except:
        lx = 1 # Length 1
        x = [x] # Pack into list
    for i in range(lx):
        x_tmp = x[i]
        # Evaluate X (altered by mu and sigma)
        X = (x_tmp-mu) / sigma
        # Define Integrand
        def integrand(sq):
            return( _np.exp(-sq**2/2) )
        integral = integrate(integrand,_np.NINF,X) # Integrate
        result = 1/_np.sqrt(2*_np.pi) * integral[0] # Evaluate Result
        F = _np.append(F, result) # Append to output list
    # Return only the 0-th value if there's only 1 value available
    if(len(F)==1):
        F = F[0]
    return(F)

# Define Probability Density Function
def probdensity(func,x,x0=0,scale=True):
    """
    Probability Density Function:
    
    This function uses an integral to compute the probability
    density of a given function.
    
    Parameters
    ----------
    func:   function
            The function for which to calculate the PDF
    x:      numpy.ndarray
            The (array of) value(s) at which to calculate
            the PDF
    x0:     float, optional
            The lower-bound of the integral, starting point
            for the PDF to be calculated over, default=0
    scale:  bool, optional
            The scaling to be applied to the output,
            default=True
    
    Returns
    -------
    sumx:   numpy.ndarray
            The (array of) value(s) computed as the PDF at
            point(s) x
    """
    sumx = _np.array([])
    try:
        lx = len(x) # Find length of Input
    except:
        lx = 1 # Length 1
        x = [x] # Pack into list
    # Recursively Find Probability Density
    for i in range(lx):
        sumx = _np.append(sumx,integrate(func,x0,x[i])[0])
    # Return only the 0-th value if there's only 1 value available
    if(len(sumx)==1):
        sumx = sumx[0]
    else:
        if(scale==True):
            mx = sumx.max()
            sumx /= mx
        elif(scale!=False):
            sumx /= scale
    return(sumx)

# Define Real FFT Evaluation Function
def rfft(arr,dt=0.01,absolute=True,resample=True):
    """
    RFFT Function
    
    This function is designed to evaluat the real FFT
    of a input signal in the form of an array or list.
    
    Parameters
    ----------
    arr:        numpy.ndarray
                The input array representing the signal
    dt:         float, optional
                The time-step used for the array,
                default=0.01
    absolute:   bool, optional
                Control argument to force absolute
                values, default=True
    resample:   bool, optional
                Control argument specifying whether
                the FFT output should be resampled,
                or if it should have a specific
                resampling rate, default=True
    
    Returns
    -------
    FFT Array
    """
    # Calculate with Absolute Values
    if absolute:
        fourier = abs(_np.fft.rfft(arr))
    else:
        foruier = _np.fft.rfft(arr)
    if resample==True:
        # Evaluate the Downsampling Ratio
        dn = int(dt*len(arr))
        # Downsample to remove unnecessary points
        fixedfft = filter.dnsample(fourier,dn)
        return(fixedfft)
    elif resample==False:
        return(fourier)
    else:
        # Condition Resample Value
        resample = int(resample)
        # Downsample to remove unnecessary points
        fixedfft = filter.dnsample(fourier,resample)
        return(fixedfft)

# Define Normalized Power Spectrum Function
def wrms(func,dw=0.1,NN=100,quad=False,plot=True,
         title="Power Density Spectrum",round=3):
    """
    WRMS Function:
    
    This function is designed to calculate the RMS
    bandwidth (Wrms) using a numerical process.
    
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
    omega = _np.linspace(0,(NN-1)*del_w,NN)
    # Initialize Fraction Terms
    Stot = Sw2 = 0
    # Power Density Spectrum
    Sxx = _np.array([])
    for n in range(NN):
        # Calculate Power Density Spectrum
        Sxx = _np.append(Sxx,func(omega[n]))
        Stot = Stot + Sxx[n]
        Sw2 = Sw2 + (omega[n]**2)*Sxx[n]
    if(quad):
        def intf(w):
            return(w**2*func(w))
        num = integrate(intf,0,_np.inf)[0]
        den = integrate(func,0,_np.inf)[0]
        # Calculate W
        W = _np.sqrt(num/den)
    else:
        # Calculate W
        W = _np.sqrt(Sw2/Stot)
    Wr = _np.around(W,round)
    # Plot Upon Request
    if(plot):
        _plt.plot(omega,Sxx)
        _plt.title(title)
        # Evaluate Text Location
        x = 0.65*max(omega)
        y = 0.80*max(Sxx)
        _plt.text(x,y,"Wrms: "+str(Wr))
        _plt.show()
    # Return Calculated RMS Bandwidth
    return(W)
        
# Define Hartley's Equation for Data Capacity
def hartleydata(BW,M):
    """
    Hartley Data Function
    
    Function to calculate Hartley's Law,
    the maximum data rate achievable for
    a given noiseless channel.
    
    Parameters
    ----------
    BW:         float
                Bandwidth of the data channel.
    M:          float
                Number of signal levels.
    
    Returns:
    --------
    C:          float
                Capacity of channel (in bits per second)
    """
    C = 2*BW*_np.log2(M)
    return(C)

# Define Shannon's Equation For Data Capacity
def shannondata(BW,S,N):
    """
    Shannon Data Function
    
    Function to calculate the maximum data
    rate that may be achieved given a data
    channel and signal/noise characteristics
    using Shannon's equation.
    
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
    C = BW*_np.log2(1+S/N)
    return(C)

# Define CRC Generator (Sender Side)
def crcsender(data, key):
    """
    CRC Sender Function
    
    Function to generate a CRC-embedded
    message ready for transmission.
    
    Contributing Author Credit:
    Shaurya Uppal
    Available from: geeksforgeeks.org
    
    Parameters
    ----------
    data:       string of bits
                The bit-string to be encoded.
    key:        string of bits
                Bit-string representing key.
    
    Returns
    -------
    codeword:   string of bits
                Bit-string representation of
                encoded message.
    """
    # Define Sub-Functions
    def xor(a, b): 
        # initialize result 
        result = [] 

        # Traverse all bits, if bits are 
        # same, then XOR is 0, else 1 
        for i in range(1, len(b)): 
            if a[i] == b[i]: 
                result.append('0') 
            else: 
                result.append('1') 

        return(''.join(result))

    # Performs Modulo-2 division 
    def mod2div(divident, divisor):
        # Number of bits to be XORed at a time. 
        pick = len(divisor) 

        # Slicing the divident to appropriate 
        # length for particular step 
        tmp = divident[0 : pick] 

        while pick < len(divident): 

            if tmp[0] == '1': 

                # replace the divident by the result 
                # of XOR and pull 1 bit down 
                tmp = xor(divisor, tmp) + divident[pick] 

            else:   # If leftmost bit is '0' 

                # If the leftmost bit of the dividend (or the 
                # part used in each step) is 0, the step cannot 
                # use the regular divisor; we need to use an 
                # all-0s divisor. 
                tmp = xor('0'*pick, tmp) + divident[pick] 

            # increment pick to move further 
            pick += 1

        # For the last n bits, we have to carry it out 
        # normally as increased value of pick will cause 
        # Index Out of Bounds. 
        if tmp[0] == '1': 
            tmp = xor(divisor, tmp) 
        else: 
            tmp = xor('0'*pick, tmp) 

        checkword = tmp 
        return(checkword)
    
    # Condition data
    data = str(data)
    # Condition Key
    key = str(key)
    l_key = len(key)
   
    # Appends n-1 zeroes at end of data 
    appended_data = data + '0'*(l_key-1) 
    remainder = mod2div(appended_data, key) 
   
    # Append remainder in the original data 
    codeword = data + remainder 
    return(codeword)

# Define CRC Generator (Sender Side)
def crcremainder(data, key):
    """
    CRC Remainder Function
    
    Function to calculate the CRC
    remainder of a CRC message.
    
    Contributing Author Credit:
    Shaurya Uppal
    Available from: geeksforgeeks.org
    
    Parameters
    ----------
    data:       string of bits
                The bit-string to be decoded.
    key:        string of bits
                Bit-string representing key.
    
    Returns
    -------
    remainder: string of bits
                Bit-string representation of
                encoded message.
    """
    # Define Sub-Functions
    def xor(a, b): 
        # initialize result 
        result = [] 

        # Traverse all bits, if bits are 
        # same, then XOR is 0, else 1 
        for i in range(1, len(b)): 
            if a[i] == b[i]: 
                result.append('0') 
            else: 
                result.append('1') 

        return(''.join(result))

    # Performs Modulo-2 division 
    def mod2div(divident, divisor):
        # Number of bits to be XORed at a time. 
        pick = len(divisor) 

        # Slicing the divident to appropriate 
        # length for particular step 
        tmp = divident[0 : pick] 

        while pick < len(divident): 

            if tmp[0] == '1': 

                # replace the divident by the result 
                # of XOR and pull 1 bit down 
                tmp = xor(divisor, tmp) + divident[pick] 

            else:   # If leftmost bit is '0' 

                # If the leftmost bit of the dividend (or the 
                # part used in each step) is 0, the step cannot 
                # use the regular divisor; we need to use an 
                # all-0s divisor. 
                tmp = xor('0'*pick, tmp) + divident[pick] 

            # increment pick to move further 
            pick += 1

        # For the last n bits, we have to carry it out 
        # normally as increased value of pick will cause 
        # Index Out of Bounds. 
        if tmp[0] == '1': 
            tmp = xor(divisor, tmp) 
        else: 
            tmp = xor('0'*pick, tmp) 

        checkword = tmp 
        return(checkword)
    
    # Condition data
    data = str(data)
    # Condition Key
    key = str(key)
    l_key = len(key)
   
    # Appends n-1 zeroes at end of data 
    appended_data = data + '0'*(l_key-1) 
    remainder = mod2div(appended_data, key) 
   
    return(remainder)

# Define String to Bits Function
def string_to_bits(str):
    """
    String to Bits Converter
    
    Converts a Pythonic string to the string's
    binary representation.
    
    Parameters
    ----------
    str:        string
                The string to be converted.
    
    Returns
    -------
    data:       string
                The binary representation of the
                input string.
    """
    data = (''.join(format(ord(x), 'b') for x in str))
    return(data)
    
# Define kWh to BTU function and vice-versa
def kwh_to_btu(kWh):
    """
    Killo-Watt-Hours to BTU Function:
    
    Converts kWh (killo-Watt-hours) to BTU
    (British Thermal Units).
    
    Parameters
    ----------
    kWh:        float
                The number of killo-Watt-hours
    
    Returns
    -------
    BTU:        float
                The number of British Thermal Units
    """
    return(kWh*3412.14)
def btu_to_kwh(BTU):
    """
    BTU to Killo-Watt-Hours Function:
    
    Converts BTU (British Thermal Units) to
    kWh (killo-Watt-hours).
    
    Parameters
    ----------
    BTU:        float
                The number of British Thermal Units
    
    Returns
    -------
    kWh:        float
                The number of killo-Watt-hours
    """
    return(BTU/3412.14)

# Define Per-Unit Impedance Formula
def zpu(S,VLL=None,VLN=None):
    """
    Per-Unit Impedance Evaluator
    
    Evaluates the per-unit impedance value given the per-unit
    power and voltage bases.
    
    .. math:: Z_{pu}=\\frac{V_{LL}^2}{S}
    
    .. math:: Z_{pu}=\\frac{(\\sqrt{3}*V_{LN})^2}{S}
    
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
    if(VLL==None and VLN==None):
        raise ValueError("ERROR: One voltage must be provided.")
    if VLL!=None:
        return(VLL**2/S)
    else:
        return((_np.sqrt(3)*VLN)**2/S)

# Define Per-Unit Current Formula
def ipu(S,VLL=None,VLN=None,V1phs=None):
    """
    Per-Unit Current Evaluator
    
    Evaluates the per-unit current value given the per-unit
    power and voltage bases.
    
    .. math:: I_{pu}=\\frac{S}{\\sqrt{3}*V_{LL}}
    
    .. math:: I_{pu}=\\frac{S}{3*V_{LN}}
    
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
    if(VLL==None and VLN==None):
        raise ValueError("ERROR: One voltage must be provided.")
    if VLL!=None:
        return(S/(_np.sqrt(3)*VLL))
    elif VLN != None:
        return(S/(3*VLN))
    else:
        return(S/V1phs)

# Define Per-Unit Change of Base Function
def puchgbase(quantity, puB_old, puB_new):
    """
    Per-Unit Change of Base Function
    
    Performs a per-unit change of base operation for the given
    value constrained by the old base and new base.
    
    .. math:: Z_{pu-new}=Z_{pu-old}*\\frac{BASE_{OLD}}{BASE_{NEW}}
    
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
    pu_new = quantity*puB_old/puB_new
    return(pu_new)

# Define Recomposition Function
def zrecompose(z_pu,S3phs,VLL=None,VLN=None):
    """
    Impedance from Per-Unit System Evaluator
    
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
    zbase = zpu(S3phs,VLL,VLN)
    # Evaluate the impedance
    z = z_pu * zbase
    return(z)

# Define X/R Recomposition Function
def rxrecompose(x_pu,XoR,S3phs=None,VLL=None,VLN=None):
    """
    Resistance/Reactance from Per-Unit System Evaluator
    
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
    r_pu = x_pu/XoR
    # Compose into z
    z_pu = r_pu + 1j*x_pu
    # Recompose
    if S3phs == None:
        return(z_pu)
    else:
        z = zrecompose(z_pu,S3phs,VLL,VLN)
        return(z)

# Define Generator Internal Voltage Calculator
def geninternalv(I,Zs,Vt,Vgn=None,Zm=None,Ip=None,Ipp=None):
    """
    Generator Internal Voltage Evaluator
    
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
    if Zmp == Zmpp == Ip == Ipp != None :
        if Vgn == None:
            Vgn = 0
        Ea = Zs*I + Zmp*Ip + Zmpp*Ipp + Vt + Vgn
    # Select Parameters Provided
    elif Vgn == Zm == Ip == Ipp == None :
        Ea = Zs*I + Vt
    # Invalid Parameter Set
    else:
        raise ValueError("Invalid Parameter Set")
    return(Ea)

# Define Sequence Component Conversion Function
def abc_to_seq(Mabc,reference='A'):
    """
    Phase-System to Sequence-System Conversion
    
    Converts phase-based values to sequence
    components.
    
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
    return(M.dot(Mabc))
# Define Second Name for abc_to_seq
phs_to_seq = abc_to_seq

# Define Phase Component Conversion Function
def seq_to_abc(M012,reference='A'):
    """
    Sequence-System to Phase-System Conversion
    
    Converts sequence-based values to phase
    components.
    
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
    return(M)
# Define Second Name for seq_to_abc
seq_to_phs = seq_to_abc

# Define Sequence Impedance Calculator
def sequencez(Zabc,reference='A',round=3):
    """
    Sequence Impedance Calculator
    
    Accepts the phase (ABC-domain) impedances for a
    system and calculates the sequence (012-domain)
    impedances for the same system.
    
    Parameters
    ----------
    Zabc:       numpy.ndarray of complex
                2-D (3x3) matrix of complex values
                representing the phasor impedances
                in the ABC-domain.
    reference:  {'A', 'B', 'C'}
                Single character denoting the reference,
                default='A'
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
    """
    # Condition Reference
    reference = reference.upper()
    rollrate = {'A':0,'B':1,'C':2}
    # Test Validity
    if reference not in rollrate:
        raise ValueError("Invalad Phase Reference")
    # Determine Roll Factor
    roll = rollrate[ reference ]
    # Evaluate Matricies
    M012 = _np.roll(A012,roll,0)
    Minv = _np.linalg.inv(M012)
    # Compute Sequence Impedances
    Z012 = Minv.dot( Zabc.dot(M012) )
    return(_np.around(Z012,round))

# FFT Coefficient Calculator Function
def funcfft(func, minfreq=60, maxmult=15, complex=False):
    """
    Function FFT Evaluator
    
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
    T = 1/minfreq
    # Generate time range to apply for FFT
    t, dt = _np.linspace(0, T, NN, endpoint=False, retstep=True)
    # Evaluate FFT
    y = _np.fft.rfft(func(t)) / t.size
    # Return Complex Values
    if complex:
       return(y)
    # Split out useful values
    else:
       y *= 2
       return(y[0].real, y[1:-1].real, -y[1:-1].imag)

def sampfft(data,dt,minfreq=60.0,complex=False):
    """
    Sampled Dataset FFT Evaluator
    
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
    FR = 1/(dt*len(data))
    NN = 1//(dt*minfreq)
    # Test for Invalid System
    if FR > minfreq:
        raise ValueError("Too few data samples to evaluate FFT at specified minimum frequency.")
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
       return(y)
    # Split out useful values
    else:
       y *= 2
       return(y[0].real, y[1:-1].real, -y[1:-1].imag)

# Define FFT Plotting Function
def fftplot(dc, real, imag=None, title="Fourier Coefficients"):
    """
    FFT System Plotter
    
    Plotting function for FFT (harmonic) values,
    plots the DC, Real, and Imaginary components.
    
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
    """
    # Define Range values for plots
    rng = range(1,len(real)+1,1)
    xtic = range(0,len(real)+1,1)
    # Set up Arguments
    a0x = [0,0]
    a0y = [0,dc/2]
    # Plot
    _plt.title(title)
    _plt.plot(a0x,a0y,'g',label="DC-Term")
    _plt.stem(rng,real,'r','ro',label="Real-Terms",use_line_collection=True)
    if imag != None:
        _plt.stem(rng,imag,'b','bo',label="Imaginary-Terms",use_line_collection=True)
    _plt.xlabel("Harmonics (Multiple of Fundamental)")
    _plt.ylabel("Harmonic Magnitude")
    _plt.axhline(0.0,color='k')
    _plt.legend()
    if(len(xtic) < 50):
        _plt.xticks(xtic)
    _plt.show()

# Define FFT Composition Plotting Function
def fftsumplot(dc,real,imag=None,freq=60,xrange=None,npts=1000,
               plotall=False,title="Fourier Series Summation"):
    """
    FFT Summation Plotter
    
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
    """
    # Determine the number (N) of terms
    N = len(real)
    # Determine the system period (T)
    T = 1/freq
    # Generate Domain Array
    if xrange == None:
        x = _np.linspace(0,T,npts)
    else:
        x = _np.linspace(xrange[0],xrange[1],npts)
    # Initialize output with DC term
    yout = _np.ones(len(x))*dc
    # Plot each iteration of the Fourier Series
    for k in range(1,N):
        if plotall:
            _plt.plot(x,yout)
        yout += real[k-1]*_np.cos(k*2*_np.pi*x/T)
        if imag != None:
            yout += imag[k-1]*_np.sin(k*2*_np.pi*x/T)
    _plt.plot(x,yout)
    _plt.title(title)
    _plt.xlabel("Time (seconds)")
    _plt.ylabel("Magnitude")
    _plt.show()

# Define harmonic system generation function
def harmonics(real,imag=None,dc=0,freq=60,domain=None):
    """
    Harmonic Function Generator
    
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
    if not isinstance(real,(list,_np.ndarray)):
        raise ValueError("Argument *real* must be array-like.")
    if imag != None and not isinstance(imag,(list,_np.ndarray)):
        raise ValueError("Argument *imag* must be array-like.")
    # Calculate Omega
    w = 2*_np.pi*freq
    def _harmonic_(t):
        out = dc
        for k in range(len(real)):
            # Evaluate Current Coefficient
            A = real[k]
            if imag != None:
                B = imag[k]
            else:
                B = 0
            m = k + 1
            # Calculate Output
            out += A*_np.cos(m*w*t) + B*_np.sin(m*w*t)
        # Return Value
        return(out)
    if domain is None:
        system = _harmonic_
    else:
        system = _harmonic_(domain)
    return(system)

# Define Single Phase Motor Startup Capacitor Formula
def motorstartcap(V,I,freq=60):
    """
    Single Phase Motor Starting Capacitor Function
    
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
    C = I / (2*_np.pi*freq*V)
    return(C)

# Define Power Factor Correction Function
def pfcorrection(S,PFold,PFnew,VLL=None,VLN=None,V=None,freq=60):
    """
    Power Factor Correction Function
    
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
    Pold = S*PFold
    Qold = _np.sqrt(S**2 - Pold**2)
    # Evaluate Reactive Power Requirements
    Scorrected = Pold/PFnew
    Qcorrected = _np.sqrt(Scorrected**2 - Pold**2)
    Qc = Qold - Qcorrected
    # Evaluate Capacitance Based on Voltage Input
    if VLL == VLN == V == None:
        raise ValueError("One voltage must be specified.")
    elif VLN != None:
        C = Qc / (2*_np.pi*freq*3*VLN**2)
    else:
        if VLL != None:
            V = VLL
        C = Qc / (2*_np.pi*freq*V**2)
    # Return Value
    return(C,Qc)

# Define Apparent Power / Voltage / Current Relation Function
def acpiv(S=None,I=None,VLL=None,VLN=None,V=None):
    """
    AC Power-Voltage-Current Relation Function
    
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
    """
    # Validate Inputs
    if S == I == None:
        raise ValueError("To few arguments.")
    # Solve Single-Phase
    if V != None:
        if S == None:   # Solve for Apparent Power
            S = V * _np.conj( I )
            return(S)
        else:           # Solve for Current
            I = _np.conj( S/V )
            return(I)
    # Solve Line-to-Line
    elif VLL != None:
        if S == None:   # Solve for Apparent Power
            S = _np.sqrt(3) * VLL * _np.conj( I )
            return(S)
        else:           # Solve for Current
            I = _np.conj( S/(_np.sqrt(3) * VLL) )
            return(I)
    # Solve Line-to-Neutral
    elif VLN != None:
        if S == None:   # Solve for Apparent Power
            S = 3 * VLN * _np.conj( I )
            return(S)
        else:           # Solve for Current
            I = _np.conj( S/(3*VLN) )
            return(I)
    # Solve for Voltages
    else:
        V = S/_np.conj( I )
        VLL = S/(_np.sqrt(3) * _np.conj( I ))
        VLN = S/(3 * _np.conj( I ))
        return(VLL,VLN,V)

# Define Primary Ratio Function
def primary(val, Np, Ns=1, invert=False):
    """
    Transformer Primary Evaluator
    
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
        return( val * Ns/Np )
    return( val * Np/Ns )

# Define Secondary Ratio Function
def secondary(val, Np, Ns=1,invert=False):
    """
    Transformer Secondary Evaluator
    
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
        return( val * Np/Ns )
    return( val * Ns/Np )

# Define Natural Frequency/Resonant Frequency Calculator
def natfreq(C,L,Hz=True):
    """
    Natural Frequency Evaluator
    
    Evaluates the natural frequency (resonant frequency)
    of a circuit given the circuit's C and L values. Defaults
    to returning values in Hz, but may also return in rad/sec.
    
    .. math:: freq=\\frac{1}{\\sqrt{L*C}*(2*\\pi)}
    
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
    freq = 1/_np.sqrt(L*C)
    # Convert to Hz as requested
    if Hz:
        freq = freq / (2*_np.pi)
    return(freq)

# Define Voltage/Current Unbalance Equation
def unbalance(A,B,C,all=False):
    """
    Voltage/Current Unbalance Function
    
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
    avg = (A + B + C)/3
    # Determine Variance
    dA = abs(A-avg)
    dB = abs(B-avg)
    dC = abs(C-avg)
    # Gather Maximum Variation
    mx = max(dA,dB,dC)
    # Calculate Maximum Variation
    unbalance = mx/avg
    # Return Results
    if all:
        return(dA/avg,dB/avg,dC/avg)
    else:
        return(unbalance)

# Define Cosine Filter Function
def cosfilt(arr,Srate,domain=False):
    """
    Cosine Filter Function
    
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
    ind = _np.arange(Srate-1, len(arr)-1)
    # Define Cosine Coefficient Function
    def cos(k,Srate):
        return(_np.cos(2*_np.pi*k/Srate))
    # Calculate Constant
    const = 2/Srate
    # Iteratively Calculate
    cosf = 0
    for k in range(0,Srate-1):
        slc = (ind-(Srate-1))+k
        cosf += cos(k,Srate) * arr[slc]
    # Scale
    cosf = const * cosf
    # Return Cosine-Filtered Array
    if domain:
        xarray = _np.linspace(Srate+Srate/4-1,len(arr)-1,len(cosf))
        xarray = xarray / Srate
        return(cosf,xarray)
    return(cosf)

# Define Sine Filter Function
def sinfilt(arr,Srate,domain=False):
    """
    Sine Filter Function
    
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
    ind = _np.arange(Srate-1, len(arr)-1)
    # Define Cosine Coefficient Function
    def sin(k,Srate):
        return(_np.sin(2*_np.pi*k/Srate))
    # Calculate Constant
    const = 2/Srate
    # Iteratively Calculate
    sinf = 0
    for k in range(0,Srate-1):
        slc = (ind-(Srate-1))+k
        sinf += sin(k,Srate) * arr[slc]
    # Scale
    sinf = const * sinf
    # Return Cosine-Filtered Array
    if domain:
        xarray = _np.linspace(Srate+Srate/4-1,len(arr)-1,len(sinf))
        xarray = xarray / Srate
        return(sinf,xarray)
    return(sinf)


# Define Characteristic Impedance Calculator
def characterz(R,G,L,C,freq=60):
    """
    Characteristic Impedance Calculator
    
    Function to evaluate the characteristic 
    impedance of a system with specefied
    line parameters as defined. System uses
    the standard characteristic impedance
    equation :eq:`Zc`.
    
    .. math:: Z_c = \\sqrt{\\frac{R+j\\omega L}{G+j\\omega C}}
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
    w = 2*_np.pi*freq
    # Evaluate Zc
    Zc = _np.sqrt((R+1j*w*L)/(G+1j*w*C))
    return(Zc)



# END OF FILE