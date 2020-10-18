###################################################################
"""
`phasor` Module

Represent complex numbers as phasors with magnitude and angle
more appropriate for electrical calculations.

THIS IS NOT COMPLETE!!!
"""
###################################################################


class phasor(object):

    def __init__(self,mag,ang=0):
        self.mag = mag
        self.ang = ang
    
    def __add__(self,addition):
        self.mag += addition
        return self

    def __str__(self):
        return f'{self.mag}/_{self.ang}'

    def __complex__(self):
        return self.mag + 1j * self.ang

    def __repr__(self):
        return f'{self.__class__.__name__}({self.mag},{self.ang})'