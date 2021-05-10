# ElectricPy <img src="https://raw.githubusercontent.com/engineerjoe440/ElectricPy/master/logo/ElectricpyLogo.svg" width="100" alt="logo" align="right">
*Electrical-Engineering-for-Python*

[![](https://img.shields.io/pypi/pyversions/electricpy.svg?color=3776AB&logo=python&logoColor=white)](https://www.python.org/)
![Tox Import Test](https://github.com/engineerjoe440/ElectricPy/workflows/Tox%20Tests/badge.svg)
![ElectricPy Tests (pytest)](https://github.com/engineerjoe440/ElectricPy/workflows/pytest/badge.svg)

[![](https://img.shields.io/pypi/v/electricpy.svg?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/electricpy/)
[![](https://pepy.tech/badge/electricpy)](https://pepy.tech/project/electricpy)
[![](https://img.shields.io/github/stars/engineerjoe440/electricpy?logo=github)](https://github.com/engineerjoe440/electricpy/)
[![](https://img.shields.io/pypi/l/electricpy.svg?color=blue)](https://github.com/engineerjoe440/electricpy/blob/master/LICENSE.txt)
[<img align="right" src="https://cdn.buymeacoffee.com/buttons/default-orange.png" width="217px" height="51x">](https://www.buymeacoffee.com/engineerjoe440)

Python Libraries with functions and constants related to electrical engineering.

The functions and constants that make up these modules represent a library of material compiled with the intent of being used primarily
for research, development, education, and exploration in the realm of electrical engineering.

Check out our full documentation: https://engineerjoe440.github.io/ElectricPy/

### Special thanks to:
- Stephen Weeks | Student - University of Idaho
- Jeremy Perhac | Student - University of Idaho
- Daniel Allen | Student - Universtiy of Idaho
- Dr. Dennis Sullivan | Proffessor - University of Idaho
- Dr. Brian Johnson | Proffessor - University of Idaho
- Dr. Joe Law | Proffessor - University of Idaho
- StackOverflow user gg349
- Shaurya Uppal | Online Code Contributor
- Paul Ortman | Power Quality Engineer - Idaho Power | Instructor - University of Idaho


### Dependencies:
- NUMPY
- MATPLOTLIB
- SCIPY
- SYMPY
- NUMDIFFTOOLS


## INSTALLATION:

### 1) Install Required Dependencies:
 1. Install Dependencies
    - `pip install numpy`
    - `pip install scipy`
    - `pip install matplotlib`
    - `pip install sympy`
    - `pip install numdifftools`

### 2) (option a) Install ElectricPy with Python's Own `pip`
 2. Install *electricpy*
    - `pip install electricpy`

### 2) (option b) Install ElectricPy from Source
[Python Documentation](https://docs.python.org/3/install/index.html)
  
 2. Collect Repository and Install
    
    1. Clone/Download Source Code from GitHub [Repository](https://github.com/engineerjoe440/ElectricPy)
  
    2. Open Terminal and Navigate to Folder with `cd` Commands:
        - `cd <path\to\containing\folder>\electricpy`

    3. Use Python to Install Module from `setup.py`:
        - `python setup.py install`
  
### 3) Verify Installation
 3. Check installation success in Python environment

   ```python
   import electricpy
   electricpy._version_
   ```

## To Do List:
- Add Heat Sink Solver
- DC/DC Converters
- DC/AC Converters
- Stationary and Synchronous Reference Frame conversion Matrices/Constants
- Induction Machine slip finder
- Add arc-flash calculators suggested
- Add [Pi Attenuator](https://www.basictables.com/electronics/resistor/pi-attenuator) Formula
- Add [T Attenuator](https://www.basictables.com/electronics/resistor/t-attenuator) Formula
- Add Simple [decibel](https://www.basictables.com/electronics/decibel-dbw) Formulas
- Add Simple [wire resistance](https://www.basictables.com/electronics/resistor/wire-resistance) Formula(s)
Formulas
- Add [Inductive Voltage Divider](https://www.basictables.com/electronics/inductor/inductive-voltage-divider) Formula
- Add [Slew Rate](https://www.basictables.com/electronics/slew-rate) Formula(s)
- Add Simple [Battery Discharge Rate](https://www.basictables.com/electronics/battery/battery-discharge-rate) Formula
- Add Simple [Air Core Inductor](https://www.basictables.com/electronics/inductor/air-core-inductor) Formula(s)
- Add Simple [Zener Diode](https://www.basictables.com/electronics/diode/zener-diode) Formulas
- *Develop Testing for All Functions*

## Get Involved / Contribute
If you're interested in contributing, we'd love to see your support in a number of ways!
 1. **Write Tests** - We're really lacking in this area. We've recently added simple GitHub actions to test installation, but that's about it. We hope that someday we can test all functions in this module for verification.
 2. **Contribute New Electrical Engineering Functions** - If you've got a new function related to electrical engineering that you'd like to see added, we'd love to throw it into this module. Our goal is that this module can become the comprehensive electrical engineering toolkit in Python. Drop us a note, or create a pull request!
 3. **Report Issues** - We don't want issues to go unnoticed. Please help us track bugs and resolve them!
 4. **Get the Word Out** - This project is still in its infancy, so please share it with your friends and colleagues. We want to make sure that everyone has the opportunity to take advantage of this project.
    

## Contact:
For more information regarding this resource, please contact Joe Stanley
- <stan3926@almuni.uidaho.edu>
- <joe_stanley@selinc.com>

## License and Usage:
ElectricPy is licensed under the standard MIT license, and as such, you are permitted
to use this resource as you see fit. Please feel free to ask questions, suggest edits
and report bugs or other issues.
