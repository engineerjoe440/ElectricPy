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


