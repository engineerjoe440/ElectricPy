ElectricPy API
================================================================================

.. _electricpyapi.py:

Python functions and constants related to electrical engineering.

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

.. rubric:: Modules

.. autosummary::
   :recursive:
   :toctree: api
   :template: module.rst

   electricpy
   electricpy.bode
   electricpy.conversions
   electricpy.fault
   electricpy.latex
   electricpy.math
   electricpy.passive
   electricpy.phasor
   electricpy.sim
   electricpy.visu
