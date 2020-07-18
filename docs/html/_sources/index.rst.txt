.. _electricpy_home:

==========
ElectricPy
==========


*Electrical-Engineering-for-Python*

Python Libraries with functions and constants related to electrical engineering.

The functions and constants that make up these modules represent a library of
material compiled with the intent of being used primarily for research,
development, education, and exploration in the realm of electrical engineering.

The base module for the `electricpy` package, electricpy.py may be leveraged
in any Python script or program by using the *import* command similar to that
shown below.

>>> import electricpy as ep

Filled with calculators, evaluators, and plotting functions, this package will
provide a wide array of capabilities to any electrical engineer.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.

.. toctree::
   :maxdepth: 1
   
   electricpy
   bode
   fault
   sim

-------------------------
ADDITIONAL PROJECT LINKS:
-------------------------
PyPI: https://pypi.org/project/electricpy/

GitHub: https://github.com/engineerjoe440/electricpy


-------------
Dependencies:
-------------
* NUMPY
* MATPLOTLIB
* SCIPY
* SYMPY
* NUMDIFFTOOLS


-------------
INSTALLATION:
-------------

#. Install required dependencies (NumPy, SciPy, SymPy, NUMDIFFTOOLS, and Matplotlib)

   .. code-block:: bash
   
      $> pip install numpy
      $> pip install scipy
      $> pip install matplotlib
      $> pip install sympy
      $> pip install numdifftools
  
#. Install ElectricPy

   .. code-block:: bash
   
      $> pip install electricpy
  
#. Check installation success in Python environment

   .. code-block:: python
   
      import electricpy
      electricpy._version_

----------
To Do List
----------

* Port Remaining Functions from ELECTRICALPYTHON

* Add Heat Sink Solver

* DC/DC Converters

* DC/AC Converters

* Stationary and Synchronous Reference Frame conversion Matricies/Constants


---------------------
ADDITIONAL RESOURCES:
---------------------
Generic; Data Science:

    * NumPy: https://numpy.org/

    * SciPy: https://scipy.org/

    * Matplotlib: https://matplotlib.org/

    * SymPy: https://www.sympy.org/en/index.html

Electrical Engineering Focus:

    * Python COMTRADE File Interpreter: https://github.com/dparrini/python-comtrade
    
    * PandaPower: https://www.pandapower.org/start/
    
    * PyPSA: https://github.com/PyPSA/PyPSA
    
    * PyPower (no longer supported): https://pypi.org/project/PYPOWER/


-------
Contact
-------

For more information regarding this resource, please contact Joe Stanley

* <stan3926@vandals.uidaho.edu>

* <joe_stanley@selinc.com>





.. note::
   **Special thanks to:**
   
   * Stephen Weeks | Student | University of Idaho
   * Jeremy Perhac | Student | University of Idaho
   * Daniel Allen | Student | Universtiy of Idaho
   * Dr. Dennis Sullivan | Proffessor | University of Idaho
   * Dr. Brian Johnson | Proffessor | University of Idaho
   * Dr. Joe Law | Proffessor | University of Idaho
   * StackOverflow user gg349
   * Shaurya Uppal | Online Code Contributor
   * Paul Ortman | Power Quality Engineer | Idaho Power | Instructor | University of Idaho





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
 - Sinusoid Peak-to-RMS Converter:          rms
 - Sinusoid RMS-to-Peak Converter:          peak
 - Arbitrary Waveform RMS Calculator:       funcrms
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
 - Transformer Phase Shift Calculator:      xfmphs
 - Hertz to Radians Converter:              hz_to_rad
 - Radians to Hertz Converter:              rad_to_hz
 - Induction Machine Vth Calculator:        indmachvth
 - Induction Machine Zth Calculator:        indmachzth
 - Induction Machine Pem Calculator:        indmachpem
 - Induction Machine Tem Calculator:        indmachtem
 - Induction Machine Peak Slip Calculator:  indmachpkslip
 - Induction Machine Peak Torque Calc.:     indmachpktorq
 - Induction Machine Rotor Current:         indmachiar
 - Induction Machine Starting Torque:       indmachstarttorq
 - Stator Power for Induction Machine:      pstator
 - Rotor Power for Induction Machine:       protor
 - De Calculator:                           de_calc
 - Z Per Length Calculator:                 zperlength
 - Induction Machine FOC Rating Calculator: indmachfocratings
 - Induction Machine FOC Control Calc.:     imfoc_control
 - Synchronous Machine Internal Voltage:    synmach_Eq
 - Voltage / Current / PF Relation:         vipf
 - Radians/Second to RPM Converter:         rad_to_rpm
 - RPM to Radians/Second Converter:         rpm_to_rad
 - Hertz to RPM Converter:                  hz_to_rpm
 - RPM to Hertz Converter:                  rpm_to_hz
 - Synchronous Speed Calculator:            syncspeed
 - Machine Slip Calculator:                 machslip

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
 - Single Pole Open:                     poleopen1
 - Double Pole Open:                     poleopen2
 - Simple MVA Calculator:                scMVA
 - Three-Phase MVA Calculator:           phs3mvasc
 - Single-Phase MVA Calculator:          phs1mvasc
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
 - Synchronous Machine Symm. Current:    synmach_Isym
 - Synchronous Machine Asymm. Current:   synmach_Iasym
 - Induction Machine Eigenvalue Calc.:   indmacheigenvalues
 - Induction Machine 3-Phase-SC Calc.:   indmachphs3sc
 - Induction Machine 3-Phs-Torq. Calc.:  indmachphs3torq

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