###################################################################
#   CONSTANTS.PY
#
#   Defenition of all required constants and matricies for
#   *electricpy* module.
###################################################################

# Import Supporting Libraries
import numpy as _np
import cmath as _c

# Define Electrical Engineering Constants
a = _c.rect(1,_np.radians(120)) #: 'A' Operator for Symmetrical Components
p = 1e-12 #: Pico Multiple      (10^-12)
n = 1e-9 #: Nano Multiple       (10^-9)
u = 1e-6 #: Micro (mu) Multiple (10^-6)
m = 1e-3 #: Mili Multiple       (10^-3)
k = 1e+3 #: Kili Multiple       (10^3)
M = 1e+6 #: Mega Multiple       (10^6)
G = 1e+9 #: Giga Multiple       (10^9)
NAN = float('nan')
VLLcVLN = _c.rect(_np.sqrt(3),_np.radians(30)) # Conversion Operator
ILcIP = _c.rect(_np.sqrt(3),_np.radians(-30)) # Conversion Operator

# Define Symmetrical Component Matricies
Aabc = 1/3 * _np.array([[ 1, 1, 1    ],  # Convert ABC to 012
                       [ 1, a, a**2 ],  # (i.e. phase to sequence)
                       [ 1, a**2, a ]])
A012 = _np.array([[ 1, 1, 1    ],        # Convert 012 to ABC
                 [ 1, a**2, a ],        # (i.e. sequence to phase)
                 [ 1, a, a**2 ]])
# Define Clark Component Matricies
Cabc = _np.sqrt(2/3) * _np.array([[ 1, -1/2, -1/2],         # Convert ABC to alpha/beta/gamma
                                  [ 0, _np.sqrt(3)/2, -_np.sqrt(3)/2],
                                  [ 1/_np.sqrt(2), 1/_np.sqrt(2), 1/_np.sqrt(2)]])
Cxyz = _np.array([[ 2/_np.sqrt(6), 0, 1/_np.sqrt(3)],       # Convert alpha/beta/gamma to ABC
                  [ -1/_np.sqrt(6), 1/_np.sqrt(2), 1/_np.sqrt(3)],
                  [ -1/_np.sqrt(6), -1/_np.sqrt(2), 1/_np.sqrt(3)]])
# Define Park Components Matricies
_rad = lambda th: _np.radians( th )
Pxyz = lambda th: _np.array([[ _np.cos(_rad(th)), _np.sin(_rad(th)), 0],
                             [ -_np.sin(_rad(th)),_np.cos(_rad(th)), 0],
                             [0, 0, 1]])
                 
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