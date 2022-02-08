################################################################################
"""
`electricpy` Package - `latex` Module.

>>> from electricpy import latex

This module is specifically designed to help create strings that represent LaTeX
formatted formulas, functions, and more for easy printing in tools such as
Jupyter Notebooks.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

# Import Required Packages
import numpy as _np
import cmath as _c



# Define Complex LaTeX Generator
def clatex(val, round=3, polar=True, predollar=True, postdollar=True,
           double=False):
    """
    Complex Value Latex Generator.

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
    def polarstring(val, round):
        mag, ang_r = _c.polar(val)  # Convert to polar form
        ang = _np.degrees(ang_r)  # Convert to degrees
        mag = _np.around(mag, round)  # Round
        ang = _np.around(ang, round)  # Round
        latex = str(mag) + '∠' + str(ang) + '°'
        return (latex)

    def rectstring(val, round):
        real = _np.around(val.real, round)  # Round
        imag = _np.around(val.imag, round)  # Round
        if imag > 0:
            latex = str(real) + "+j" + str(imag)
        else:
            latex = str(real) + "-j" + str(abs(imag))
        return (latex)

    # Interpret as numpy array if simple list
    if isinstance(val, list):
        val = _np.asarray(val)  # Ensure that input is array
    # Find length of the input array
    if isinstance(val, _np.ndarray):
        shp = val.shape
        try:
            row, col = shp  # Interpret Shape of Object
        except:
            row = shp[0]
            col = 1
        _ = val.size
        # Open Matrix
        latex = r'\begin{bmatrix}'
        # Iteratively Process Each Item in Array
        for ri in range(row):
            if ri != 0:  # Insert Row Separator
                latex += r'\\'
            if col > 1:
                for ci in range(col):
                    if ci != 0:  # Insert Column Separator
                        latex += r' & '
                    # Add Complex Represetation of Value
                    if polar:
                        latex += polarstring(val[ri][ci], round)
                    else:
                        latex += rectstring(val[ri][ci], round)
            else:
                # Add Complex Represetation of Value
                if polar:
                    latex += polarstring(val[ri], round)
                else:
                    latex += rectstring(val[ri], round)
        # Close Matrix
        latex += r'\end{bmatrix}'
    elif isinstance(val, complex):
        # Treat as Polar When Directed
        if polar:
            latex = polarstring(val, round)
        else:
            latex = rectstring(val, round)
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
    return latex


# Define Transfer Function LaTeX Generator
def tflatex(sys, sysp=None, var='s', predollar=True,
            postdollar=True, double=False, tolerance=1e-8):
    r"""
    Transfer Function LaTeX String Generator.

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
                term (i.e. 's' or 'j\omega'). default='s'
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
    if isinstance(sysp, (list, tuple, _np.ndarray)):
        num = sys
        den = sysp
    else:
        num, den = sys

    # Generate String Function
    def genstring(val):
        length = len(val)
        strg = ''
        for i, v in enumerate(val):
            # Add Each Term to String
            if abs(v) > tolerance:
                # Add '+' Symbol After Each Term
                if i != 0:
                    strg += r'+'
                strg += str(v)
                # Determine Exponent
                xpnt = length - i - 1
                if xpnt == 1:
                    strg += var
                elif xpnt == 0:
                    pass  # Don't Do Anything
                else:
                    strg += var + r'^{' + str(xpnt) + r'}'
        return (strg)

    # Generate Total TF String
    latex = r'\frac{' + genstring(num) + r'}{'
    latex += genstring(den) + r'}'
    # Add Dollar Sign pre-post
    if double:
        dollar = r'$$'
    else:
        dollar = r'$'
    if predollar:
        latex = dollar + latex
    if postdollar:
        latex = latex + dollar
    return (latex)