.. _electricpy_home:

==========
ELECTRICPY
==========


*Electrical-Engineering-for-Python*

Python Libraries with functions and constants related to electrical engineering.

The functions and constants that make up these modules represent a library of material compiled with the intent of being used primarily
for research, development, education, and exploration in the realm of electrical engineering.

.. note::
   **Special thanks to:**
   
   * Stephen Weeks | Student | University of Idaho
   * Jeremy Perhac | Student | University of Idaho
   * Daniel Allen | Student | Universtiy of Idaho
   * Dr. Dennis Sullivan | Proffessor | University of Idaho
   * Dr. Brian Johnson | Proffessor | University of Idaho
   * StackOverflow user gg349
   * Shaurya Uppal | Online Code Contributor
   * Paul Ortman | Power Quality Engineer | Idaho Power | Instructor | University of Idaho


---------
CONTENTS:
---------

.. toctree::
   :maxdepth: 2
   
   electricpy
   constants
   fault
   bode

---------------------
ADDITIONAL RESOURCES:
---------------------
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

#. Install required dependencies (NUMPY, SCIPY, SYMPY, NUMDIFFTOOLS, and MATPLOTLIB)

   .. code-block:: bash
   
      $> pip install numpy
      $> pip install scipy
      $> pip install matplotlib
      $> pip install sympy
      $> pip install numdifftools
  
#. Install `electricpy`

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


-------
Contact
-------

For more information regarding this resource, please contact Joe Stanley

* <stan3926@vandals.uidaho.edu>

* <joe_stanley@selinc.com>


.. automodule:: electricpy