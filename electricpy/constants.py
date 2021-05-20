################################################################################
"""
`electricpy.constants` - Electrical Engineering Constants.

Defenition of all required constants and matricies for
*electricpy* module.
"""
################################################################################

# Import Supporting Dependencies
import numpy as _np
import cmath as _c

# Define Electrical Engineering Constants
pi = _np.pi #: PI Constant 3.14159...
a = _c.rect(1,_np.radians(120)) #: 'A' Operator for Symmetrical Components
p = 1e-12 #: Pico Multiple      (10^-12)
n = 1e-9 #: Nano Multiple       (10^-9)
u = 1e-6 #: Micro (mu) Multiple (10^-6)
m = 1e-3 #: Mili Multiple       (10^-3)
k = 1e+3 #: Kili Multiple       (10^3)
M = 1e+6 #: Mega Multiple       (10^6)
G = 1e+9 #: Giga Multiple       (10^9)
u0 = 4*_np.pi*10**(-7) #: µ0 (mu-not)       4πE-7
e0 = 8.8541878128e-12  #: ε0 (epsilon-not)  8.854E-12
carson_r = 9.869e-7 #: Carson's Ristance Constant  8.869E-7
De0 = 2160 #: De Constant for Use with Transmission Impedance Calculations      2160
NAN = float('nan')
VLLcVLN = _c.rect(_np.sqrt(3),_np.radians(30)) # Conversion Operator
ILcIP = _c.rect(_np.sqrt(3),_np.radians(30)) # Conversion Operator

# Define Symmetrical Component Matricies
Aabc = 1/3 * _np.array([[ 1, 1, 1    ],  # Convert ABC to 012
                       [ 1, a, a**2 ],  # (i.e. phase to sequence)
                       [ 1, a**2, a ]])
A012 = _np.array([[ 1, 1, 1    ],        # Convert 012 to ABC
                 [ 1, a**2, a ],        # (i.e. sequence to phase)
                 [ 1, a, a**2 ]])
# Define Clarke Component Matricies
Cabc = _np.sqrt(2/3) * _np.array([[ 1, -1/2, -1/2],         # Convert ABC to alpha/beta/gamma
                                  [ 0, _np.sqrt(3)/2, -_np.sqrt(3)/2],
                                  [ 1/_np.sqrt(2), 1/_np.sqrt(2), 1/_np.sqrt(2)]])
Cxyz = _np.array([[ 2/_np.sqrt(6), 0, 1/_np.sqrt(3)],       # Convert alpha/beta/gamma to ABC
                  [ -1/_np.sqrt(6), 1/_np.sqrt(2), 1/_np.sqrt(3)],
                  [ -1/_np.sqrt(6), -1/_np.sqrt(2), 1/_np.sqrt(3)]])
# Define Park Components Matricies
_rad = lambda th: _np.radians( th )
Pdq0_im = lambda th: _np.sqrt(2/3)*_np.array([[ _np.cos(_rad(th)), _np.cos(_rad(th)-2*pi/3), _np.cos(_rad(th)+2*pi/3)],
                                           [-_np.sin(_rad(th)),-_np.sin(_rad(th)-2*pi/3),-_np.sin(_rad(th)+2*pi/3)],
                                           [ _np.sqrt(2)/2,     _np.sqrt(2)/2,            _np.sqrt(2)/2]])
Pabc_im = lambda th: _np.sqrt(2/3)*_np.array([[ _np.cos(_rad(th)),      -_np.sin(_rad(th)),        _np.sqrt(2)/2],
                                           [_np.cos(_rad(th)-2*pi/3),-_np.sin(_rad(th)-2*pi/3), _np.sqrt(2)/2],
                                           [_np.cos(_rad(th)+2*pi/3),-_np.sin(_rad(th)+2*pi/3), _np.sqrt(2)/2]])
Pdq0 = 2/3 * _np.array([[0,-_np.sqrt(3/2),_np.sqrt(3/2)],
                        [1,-1/2,-1/2],
                        [1/2, 1/2, 1/2]])
Pqd0 = 2/3 * _np.array([[1,-1/2,-1/2],
                        [0,-_np.sqrt(3/2),_np.sqrt(3/2)],
                        [1/2, 1/2, 1/2]])
                 
# Define Transformer Shift Correction Matricies
XFMY0 = _np.array([[1,0,0],[0,1,0],[0,0,1]])
XFMD1 = 1/_np.sqrt(3) * _np.array([[1,-1,0],[0,1,-1],[-1,0,1]])
XFMD11 = 1/_np.sqrt(3) * _np.array([[1,0,-1],[-1,1,0],[0,-1,1]])
XFM12 = 1/3 * _np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])

# Define Complex Angle Terms
e30 = _c.rect(1,_np.radians(30)) #: 30° Phase Operator
en30 = _c.rect(1,_np.radians(-30)) #: -30° Phase Operator
e60 = _c.rect(1,_np.radians(60)) #: 60° Phase Operator
en60 = _c.rect(1,_np.radians(-60)) #: -60° Phase Operator
e90 = _c.rect(1,_np.radians(90)) #: 90° Phase Operator
en90 = _c.rect(1,_np.radians(-90)) #: -90° Phase Operator
e45 = _c.rect(1,_np.radians(45)) #: 45° Phase Operator
en45 = _c.rect(1,_np.radians(-45)) #: -45° Phase Operator

# Define Material Resistivity (Rho)
resistivity_rho = {
    'silver':       15.9,
    'copper':       16.8,
    'aluminium':    6.5,
    'tungsten':     56,
    'iron':         97.1,
    'platinum':     106,
    'manganin':     482,
    'lead':         220,
    'mercury':      980,
    'nichrome':     1000,
    'constantan':   490,
}


# END OF FILE