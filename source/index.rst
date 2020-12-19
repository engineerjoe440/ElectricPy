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

- Add Heat Sink Solver
- DC/DC Converters
- DC/AC Converters
- Stationary and Synchronous Reference Frame conversion Matrices/Constants
- Induction Machine slip finder
- Add arc-flash calculators suggested
- Add Pi Attenuator (https://www.basictables.com/electronics/resistor/pi-attenuator) Formula
- Add T Attenuator (https://www.basictables.com/electronics/resistor/t-attenuator) Formula
- Add Simple decibel (https://www.basictables.com/electronics/decibel-dbw) Formulas
- Add Simple wire resistance (https://www.basictables.com/electronics/resistor/wire-resistance) Formula(s)
- Add Simple Wheatstone Bridge (https://www.basictables.com/electronics/resistor/wheatstone-bridge) Formulas
- Add 555 Timer (https://www.basictables.com/electronics/integrated-circuit/555-timer) Formulas
- Add Inductive Voltage Divider (https://www.basictables.com/electronics/inductor/inductive-voltage-divider) Formula
- Add Slew Rate (https://www.basictables.com/electronics/slew-rate) Formula(s)
- Add Simple Battery Discharge Rate (https://www.basictables.com/electronics/battery/battery-discharge-rate) Formula
- Add Simple Air Core Inductor (https://www.basictables.com/electronics/inductor/air-core-inductor) Formula(s)
- Add Simple Zener Diode (https://www.basictables.com/electronics/diode/zener-diode) Formulas
- *Develop Testing for All Functions*


---------------------
ADDITIONAL RESOURCES:
---------------------
Generic; Data Science:

    * NumPy: https://numpy.org/

    * SciPy: https://scipy.org/

    * Matplotlib: https://matplotlib.org/

    * SymPy: https://www.sympy.org/en/index.html
    
    * Pyomo: https://www.pyomo.org/
    
    * numdifftools: https://numdifftools.readthedocs.io/en/latest/

Electrical Engineering Focus:

    * Python COMTRADE File Interpreter: https://github.com/dparrini/python-comtrade
    
    * Python COMTRADE Writer: https://github.com/relihanl/comtradehandlers
    
    * PandaPower: https://www.pandapower.org/start/
    
    * PyPSA: https://github.com/PyPSA/PyPSA
    
    * PyPower (no longer supported): https://pypi.org/project/PYPOWER/
    
    * minpower: http://adamgreenhall.github.io/minpower/index.html
    
    * oemof (Open Energy MOdeling Framework): https://oemof.org/
    
    * PowerGAMA: https://bitbucket.org/harald_g_svendsen/powergama/wiki/Home


-------
Contact
-------

For more information regarding this resource, please contact Joe Stanley

* <stan3926@alumni.uidaho.edu>

* <joe_stanley@selinc.com>





.. note::
   
   A great many individuals contributed knowledge, formulas, resources and more!
   This project certainly would not have been possible without their contributions.
   
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


