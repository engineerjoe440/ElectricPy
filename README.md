# ElectricPy

<a href="https://electricpy.readthedocs.io/en/latest/">
  <img src="https://raw.githubusercontent.com/engineerjoe440/ElectricPy/master/logo/ElectricpyLogo.svg" width="200" alt="logo" align="right">
</a>

> *Electrical-Engineering-for-Python*

<!-- Documentation -->
[![sphinx](https://github.com/engineerjoe440/ElectricPy/actions/workflows/sphinx-build.yml/badge.svg?branch=master)](https://github.com/engineerjoe440/ElectricPy/actions/workflows/sphinx-build.yml)
[![Documentation Status](https://readthedocs.org/projects/electricpy/badge/?version=latest)](https://electricpy.readthedocs.io/en/latest/?badge=latest)

<!-- Testing -->
[![pytest](https://github.com/engineerjoe440/ElectricPy/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/engineerjoe440/ElectricPy/actions/workflows/pytest.yml)
[![pydocstyle](https://github.com/engineerjoe440/ElectricPy/actions/workflows/pydocstyle.yml/badge.svg?branch=master)](https://github.com/engineerjoe440/ElectricPy/actions/workflows/pydocstyle.yml)
![Coverage](https://raw.githubusercontent.com/engineerjoe440/ElectricPy/gh-pages/coverage.svg)

<!-- General About -->
[![PyPI Link](https://img.shields.io/pypi/v/electricpy.svg?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/electricpy/)
[![download count](https://pepy.tech/badge/electricpy)](https://pepy.tech/project/electricpy)
[![github stars count](https://img.shields.io/github/stars/engineerjoe440/electricpy?logo=github)](https://github.com/engineerjoe440/electricpy/)
[![license](https://img.shields.io/pypi/l/electricpy.svg?color=blue)](https://github.com/engineerjoe440/electricpy/blob/master/LICENSE.txt)

<!-- Python Support -->
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)

<!-- Social -->
[![Matrix](https://img.shields.io/matrix/electricpy:stanleysolutionsn.com?label=%23electricpy:stanleysolutionsnw.com&logo=matrix&server_fqdn=matrix.stanleysolutionsnw.com&style=for-the-badge)](https://matrix.to/#/#electricpy:stanleysolutionsnw.com)

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/engineerjoe440)

Python Libraries with functions and constants related to electrical engineering.

The functions and constants that make up these modules represent a library of
material compiled with the intent of being used primarily for research,
development, education, and exploration in the realm of electrical engineering.

Check out our full documentation: [https://electricpy.readthedocs.io/en/latest/](https://electricpy.readthedocs.io/en/latest/)

## Features

* Extensive set of common functions and formulas for electrical engineering and
electronics.
* Support for LaTeX math generation (use this in conjunction with your Jupyter
notebooks!)
* Generate focussed and simple plots, diagrams, and figures.

### Samples Generated with ElectricPy

| Phasor Plot | Power Triangle | Induction Motor Circle |
| ----------- | -------------- | ---------------------- |
| ![phasor plot](https://raw.githubusercontent.com/engineerjoe440/ElectricPy/gh-pages/_images/PhasorPlot.png) | ![power triangle](https://raw.githubusercontent.com/engineerjoe440/ElectricPy/gh-pages/_images/PowerTriangle.png) | ![induction motor circle](https://raw.githubusercontent.com/engineerjoe440/ElectricPy/gh-pages/_images/InductionMotorCircleExample.png) |

| RLC Frequency Response | Placeholder | Receiving Power Circle |
| --- | --- | --- |
| ![series rlc response](https://raw.githubusercontent.com/engineerjoe440/ElectricPy/gh-pages/_images/series-rlc-r5-l0.4.png) | - | ![receiving power circle](https://raw.githubusercontent.com/engineerjoe440/ElectricPy/gh-pages/_images/ReceivingPowerCircleExample.png) |

## Installing / Getting Started

1. ElectricPy has a few basic installation options for use with `pip`. For most
   common users, use the following command to install ElectricPy with `pip`

   ```bash
   pip install electricpy[full]
   ```

2. Check installation success in Python environment:

   ```python
   import electricpy
   electricpy._version_
   ```

3. Start using the electrical engineering formulas

   ```python
   >>> import electricpy as ep
   >>> voltage = ep.phasor(67, 120) # 67 volts at angle 120 degrees
   >>> voltage
   (-33.499999999999986+58.02370205355739j)
   >>> ep.cprint(voltage)
   67.0 ∠ 120.0°
   ```

### Installing from Source

If you're looking to get the "latest and greatest" from electricpy, you'll want
to install directly from GitHub, you can do that one of two ways, the easiest of
which is to simply issue the following command for `pip`

```bash
pip install git+https://github.com/engineerjoe440/ElectricPy.git
```

Alternatively, you can do it the "old fashioned way" by cloning the repository
and installing locally.

1. Clone/Download Source Code from [GitHub Repository](https://github.com/engineerjoe440/ElectricPy)
2. Open Terminal and Navigate to Folder with `cd` Commands:
   * `cd <path\to\containing\folder>\electricpy`
3. Use Python to Install Module from `setup.py`:
   * `pip install .`

### Dependencies

* [NumPy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [SciPy](https://scipy.org/)
* [SymPy](https://www.sympy.org/en/index.html)

#### Optional Dependencies

For numerical analysis (install with `pip install electricpy[numerical]`):

* [numdifftools](https://numdifftools.readthedocs.io/en/latest/)

For fault analysis (install with `pip install electricpy[fault]`)

* [arcflash](https://github.com/LiaungYip/arcflash)

## Get Involved / Contribute

If you're interested in contributing, we'd love to see your support in a number
of ways!

1. **Write Tests** - We're really lacking in this area. We've recently added
simple GitHub actions to test installation, but that's about it. We hope that
someday we can test all functions in this module for verification.
2. **Contribute New Electrical Engineering Functions** - If you've got a new
function related to electrical engineering that you'd like to see added, we'd
love to throw it into this module. Our goal is that this module can become the
comprehensive electrical engineering toolkit in Python. Drop us a note, or
create a [pull request](https://github.com/engineerjoe440/ElectricPy/pulls)!
3. **Report Issues** - We don't want issues to go unnoticed. Please help us
track bugs in [our issues](https://github.com/engineerjoe440/ElectricPy/issues)
and resolve them!
4. **Get the Word Out** - This project is still in its infancy, so please share
it with your friends and colleagues. We want to make sure that everyone has the
opportunity to take advantage of this project.

**Check out the [contribution guide](https://github.com/engineerjoe440/ElectricPy/blob/master/CONTRIBUTING.md)**

**Come [chat about ElectricPy](https://matrix.to/#/#electricpy:stanleysolutionsnw.com)**

### Special Thanks

* Stephen Weeks | Student - U of Idaho
* Jeremy Perhac | Student - U of Idaho
* Daniel Allen | Student - Universtiy of Idaho
* Dr. Dennis Sullivan | Proffessor - U of Idaho
* Dr. Brian Johnson | Proffessor - U of Idaho
* Dr. Joe Law | Proffessor - U of Idaho
* StackOverflow user gg349
* Shaurya Uppal | Online Code Contributor
* Paul Ortman | Power Quality Engineer - Idaho Power | Instructor - U of Idaho

> *and*

<a href="https://github.com/engineerjoe440/electricpy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=engineerjoe440/electricpy" alt="contributors">
</a>

## Contact

For more information regarding this resource, please contact Joe Stanley

* <engineerjoe440@yahoo.com>

## License and Usage

ElectricPy is licensed under the standard MIT license, and as such, you are
permitted to use this resource as you see fit. Please feel free to ask
questions, suggest edits and report bugs or other issues.
