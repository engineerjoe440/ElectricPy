###################################################################
#   CONSTANTS.PY
#
#   Defenition of all required constants and matricies for
#   *electricpy* module.
###################################################################

# Import Supporting Libraries
import numpy as np
import cmath as c

# Define Electrical Engineering Constants
a = c.rect(1,np.radians(120)) #: 'A' Operator for Symmetrical Components
p = 1e-12 #: Pico Multiple      (10^-12)
n = 1e-9 #: Nano Multiple       (10^-9)
u = 1e-6 #: Micro (mu) Multiple (10^-6)
m = 1e-3 #: Mili Multiple       (10^-3)
k = 1e+3 #: Kili Multiple       (10^3)
M = 1e+6 #: Mega Multiple       (10^6)
NAN = float('nan')
VLLcVLN = c.rect(np.sqrt(3),np.radians(30)) # Conversion Operator
ILcIP = c.rect(np.sqrt(3),np.radians(-30)) # Conversion Operator

# Define Electrical Engineering Matricies
Aabc = 1/3 * np.array([[ 1, 1, 1    ],  # Convert ABC to 012
                       [ 1, a, a**2 ],  # (i.e. phase to sequence)
                       [ 1, a**2, a ]])
A012 = np.array([[ 1, 1, 1    ],        # Convert 012 to ABC
                 [ 1, a**2, a ],        # (i.e. sequence to phase)
                 [ 1, a, a**2 ]])
                 
# Define Transformer Shift Correction Matricies
XFMY0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
XFMD1 = 1/np.sqrt(3) * np.array([[1,-1,0],[0,1,-1],[-1,0,1]])
XFMD11 = 1/np.sqrt(3) * np.array([[1,0,-1],[-1,1,0],[0,-1,1]])
XFM12 = 1/3 * np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])

# Define Complex Angle Terms
e30 = c.rect(1,np.radians(30)) #: 30° Phase Operator
en30 = c.rect(1,np.radians(-30)) #: -30° Phase Operator
e60 = c.rect(1,np.radians(60)) #: 60° Phase Operator
en60 = c.rect(1,np.radians(-60)) #: -60° Phase Operator
e90 = c.rect(1,np.radians(90)) #: 90° Phase Operator
en90 = c.rect(1,np.radians(-90)) #: -90° Phase Operator
e45 = c.rect(1,np.radians(45)) #: 45° Phase Operator
en45 = c.rect(1,np.radians(-45)) #: -45° Phase Operator