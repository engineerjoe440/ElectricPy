################################################################################
"""
LaTeX Support Module to help Generate LaTeX formatted Math Symbols and Formulas.

>>> from electricpy import latex

This module is specifically designed to help create strings that represent LaTeX
formatted formulas, functions, and more for easy printing in tools such as
Jupyter Notebooks.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################

import cmath as _c

import numpy as _np


def _is_complex_scalar(x):
    return isinstance(x, (complex, _np.complexfloating))


def _fmt_number(x, ndigits=3):
    """
    Format real/complex numbers for LaTeX without adding dollar signs.
    Keeps output compact and avoids '+-' artifacts.
    """
    if _is_complex_scalar(x):
        xr = float(_np.real(x))
        xi = float(_np.imag(x))
        xr_r = _np.around(xr, ndigits)
        xi_r = _np.around(xi, ndigits)

        if _np.isclose(xi_r, 0.0, atol=10 ** (-(ndigits + 1))):
            # Pure real
            return str(xr_r)
        if _np.isclose(xr_r, 0.0, atol=10 ** (-(ndigits + 1))):
            # Pure imaginary
            if xi_r >= 0:
                return r'\mathrm{j}' + str(abs(xi_r))
            return r'-\mathrm{j}' + str(abs(xi_r))

        # Full complex
        if xi_r >= 0:
            return str(xr_r) + r'+\mathrm{j}' + str(abs(xi_r))
        return str(xr_r) + r'-\mathrm{j}' + str(abs(xi_r))

    # Real numeric
    try:
        xr = float(x)
    except Exception:
        return str(x)

    xr_r = _np.around(xr, ndigits)
    return str(xr_r)


def _fmt_poly_term(coeff, power, var='s', ndigits=3):
    """
    Format a single polynomial term (no sign), e.g. "3s^{2}", "s", "5".
    coeff is assumed positive magnitude here.
    """
    # Decide if coefficient should be shown
    show_coeff = True
    if power > 0:
        if _is_complex_scalar(coeff):
            # If complex, always show it (wrapped)
            pass
        else:
            try:
                if _np.isclose(float(coeff), 1.0, atol=10 ** (-(ndigits + 1))):
                    show_coeff = False
            except Exception:
                pass

    coeff_str = _fmt_number(coeff, ndigits)

    # Wrap complex coefficients for readability when multiplied by variable
    if _is_complex_scalar(coeff) and power > 0:
        coeff_str = r'\left(' + coeff_str + r'\right)'

    if power == 0:
        return coeff_str

    if power == 1:
        if show_coeff:
            return coeff_str + var
        return var

    if show_coeff:
        return coeff_str + var + r'^{' + str(power) + r'}'
    return var + r'^{' + str(power) + r'}'


# Define Complex LaTeX Generator
def clatex(val, round=3, polar=True, predollar=True, postdollar=True,
           double=False):
    """
    Complex Value Latex Generator.

    Function to generate a LaTeX string of complex value(s)
    in either polar or rectangular form. May generate both dollar
    signs.
    """
    # Define Interpretation Functions
    def polarstring(val, round):
        mag, ang_r = _c.polar(val)  # Convert to polar form
        ang = _np.degrees(ang_r)  # Convert to degrees
        mag = _np.around(mag, round)  # Round
        ang = _np.around(ang, round)  # Round
        latex = str(mag) + r'\angle' + str(ang) + r'^{\circ}'
        return latex

    def rectstring(val, round):
        real = _np.around(_np.real(val), round)  # Round
        imag = _np.around(_np.imag(val), round)  # Round
        if imag > 0:
            latex = str(real) + r"+\mathrm{j}" + str(imag)
        else:
            latex = str(real) + r"-\mathrm{j}" + str(abs(imag))
        return latex

    # Interpret as numpy array if simple list
    if isinstance(val, list):
        val = _np.asarray(val)  # Ensure that input is array
    # Find length of the input array
    if isinstance(val, _np.ndarray):
        shp = val.shape
        try:
            row, col = shp  # Interpret Shape of Object
        except ValueError:
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
    elif _is_complex_scalar(val):
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
    """
    # Collect Numerator and Denominator Terms
    if isinstance(sysp, (list, tuple, _np.ndarray)):
        num = sys
        den = sysp
    else:
        num, den = sys

    # Generate String Function
    def genstring(val):
        val = list(val)
        length = len(val)
        terms = []
        for i, v in enumerate(val):
            if abs(v) > tolerance:
                power = length - i - 1

                # Determine sign and magnitude
                sign = '+'
                v_mag = v
                if _is_complex_scalar(v):
                    # Use real part for sign only when imag ~ 0, otherwise keep as-is (no sign extraction)
                    vr = _np.real(v)
                    vi = _np.imag(v)
                    if _np.isclose(vi, 0.0, atol=tolerance) and (vr < 0):
                        sign = '-'
                        v_mag = -v
                    else:
                        # complex (or imag not ~0): do not force sign splitting; treat as one coefficient
                        sign = '+'
                        v_mag = v
                else:
                    try:
                        if float(v) < 0:
                            sign = '-'
                            v_mag = -v
                    except Exception:
                        sign = '+'
                        v_mag = v

                term = _fmt_poly_term(v_mag, power, var=var, ndigits=3)
                terms.append((sign, term))

        if not terms:
            return '0'

        # Build string with correct leading sign handling
        out = ''
        for idx, (sgn, term) in enumerate(terms):
            if idx == 0:
                if sgn == '-':
                    out += r'-' + term
                else:
                    out += term
            else:
                out += (r'+' if sgn == '+' else r'-') + term
        return out

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
    return latex
