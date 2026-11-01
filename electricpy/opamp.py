################################################################################
"""
Functions for Operational Amplifiers (OpAmps).

>>> from electricpy import opamp
"""

################################################################################
# TODO: Add gain equations?, update docstring for inverting_summing, replace repeated operations with variables

def non_inverting(Vin=None, Vout=None, Rg=None, Rf=None):
    """
    Non-Inverting Op-Amp Solver

    This function will accept three inputs for the common operational amplifier
    formulas related to a non-inverting operational amplifier (op-amp) and will
    solve for the fourth value. All inputs default to `None`, to solve for any
    value, do not declare it as an argument and set all other inputs.

    .. image:: https://www.basictables.com/media/non-inverting-opamp-circuit.png

    Examples
    --------
    >>> from electricpy.opamp import non_inverting
    >>> non_inverting(Vin=10, Rg=10, Rf=10) # Solve for Vout
    {...}
    >>> non_inverting(Vin=10, Vout=10, Rg=10) # Solve for Rf
    {...}

    Parameters
    ----------
    Vin:    [int, float]
            Input voltage, in volts. Default is None.
    Vout:   [int, float]
            Output voltage, in volts. Default is None.
    Rg:     [int, float]
            Ground resistor value, in ohms. Default is None.
    Rf:     [int, float]
            Feedback resistor value, in ohms. Default is None.

    Returns
    -------
    int, float:  This function will return the unknown variable from the 
                 previously described set of variables.   
    """

    # Check if exactly one input is None
    if [Vin, Vout, Rg, Rf].count(None) != 1:
        raise ValueError("Exactly one input should be None")

    if Rg != None and Rg < 0:
        raise ValueError("Rg must be positive")
    
    if Rf != None and Rf < 0:
        raise ValueError("Rf must be positive")

    if Vin == None:
        try: 
            Vin = (Vout*Rg)/(Rg+Rf)
        except ZeroDivisionError:
            raise ValueError("Rg+Rf should be non-zero")
        return Vin

    if Vout == None:
        try: 
            Vout = Vin*(Rg+Rf)/Rg
        except ZeroDivisionError:
            raise ValueError("Rg should be non-zero")
        return Vout

    if Rg == None:
        try: 
            Rg = (Vin*Rf)/(Vout-Vin)
        except ZeroDivisionError:
            raise ValueError("Vout should not equal Vin")
        return Rg

    if Rf == None:
        try: 
            Rf = ((Vout*Rg)/Vin)-Rg
        except ZeroDivisionError:
            raise ValueError("Vin should be non-zero")
        return Rf


def inverting(Vin=None, Vout=None, Rin=None, Rf=None):
    """
    Inverting Op-Amp Solver

    This function will accept three inputs for the common operational amplifier
    formulas related to an inverting operational amplifier (op-amp) and will
    solve for the fourth value. All inputs default to `None`, to solve for any
    value, do not declare it as an argument and set all other inputs.

    .. image:: https://www.basictables.com/media/inverting-opamp-circuit.png

    Examples
    --------
    >>> from electricpy.opamp import inverting
    >>> inverting(Vin=10, Rin=10, Rf=10) # Solve for Vout
    {...}
    >>> inverting(Vin=10, Vout=10, Rin=10) # Solve for Rf
    {...}

    Parameters
    ----------
    Vin:    [int, float]
            Input voltage, in volts. Default is None.
    Vout:   [int, float]
            Output voltage, in volts. Default is None.
    Rin:    [int, float]
            Input resistor value, in ohms. Default is None.
    Rf:     [int, float]
            Feedback resistor value, in ohms. Default is None.

    Returns
    -------
    int, float:  This function will return the unknown variable from the 
                 previously described set of variables.   
    """

    # Check if exactly one input is None
    if [Vin, Vout, Rin, Rf].count(None) != 1:
        raise ValueError("Exactly one input should be None")

    if Rin != None and Rin < 0:
        raise ValueError("Rin must be positive")
    
    if Rf != None and Rf < 0:
        raise ValueError("Rf must be positive")

    if Vin == None:
        try: 
            Vin = -(Vout*Rin)/Rf
        except ZeroDivisionError:
            raise ValueError("Rf should be non-zero")
        return Vin

    if Vout == None:
        try: 
            Vout = -(Vin*Rf)/Rin
        except ZeroDivisionError:
            raise ValueError("Rin should be non-zero")
        return Vout

    if Rin == None:
        try: 
            Rin = -(Vin*Rf)/Vout
        except ZeroDivisionError:
            raise ValueError("Vout should be non-zero")
        return Rin

    if Rf == None:
        try: 
            Rf = -(Vout*Rin)/Vin
        except ZeroDivisionError:
            raise ValueError("Vin should be non-zero")
        return Rf

def differential(V1=None, V2=None, Vout=None, R1=None, R2=None, R3=None, R4=None, ):
    """
    Differential Op-Amp Solver

    This function will accept six inputs for the common operational amplifier
    formulas related to a differential operational amplifier (op-amp) and will
    solve for the seventh value. All inputs default to `None`, to solve for any
    value, do not declare it as an argument and set all other inputs.

    .. image:: https://www.basictables.com/media/differential-opamp-circuit.png

    Examples
    --------
    >>> from electricpy.opamp import differential
    >>> differential(V1=10, V2=15, R1=10, R2=10, R3=20 , R4=3) # Solve for Vout
    {...}
    >>> differential(V1=10,z V2=20, Vout=12, R1=10, R2=10, R4=3) # Solve for R3
    {...}

    Parameters
    ----------
    V1:     [int, float]
            Input voltage 1, in volts. Default is None.
    V2:     [int, float]
            Input voltage 2, in volts. Default is None.
    Vout:   [int, float]
            Output voltage, in volts. Default is None.
    R1:     [int, float]
            V1 Input resistor value, in ohms. Default is None.
    R2:     [int, float]
            V2 input resistor value, in ohms. Default is None.
    R3:     [int, float]
            Feedback resistor value, in ohms. Default is None.
    R4:     [int, float]
            Ground resistor value, in ohms. Default is None.

    Returns
    -------
    int, float:  This function will return the unknown variable from the 
                 previously described set of variables.   
    """

    # Check if exactly one input is None
    if [V1, V2, Vout, R1, R2, R3, R4].count(None) != 1:
        raise ValueError("Exactly one input should be None")

    if R1 != None and R1 < 0:
        raise ValueError("R1 must be positive")
    
    if R2 != None and R2 < 0:
        raise ValueError("R2 must be positive")
    
    if R3 != None and R3 < 0:
        raise ValueError("R3 must be positive")

    if R4 != None and R4 < 0:
        raise ValueError("R4 must be positive")

    if V1 == None:
        try: 
            V1 = -Vout*(R1/R3)+V2*(R4/(R2+R4))*((R1+R3)/R3)
        except ZeroDivisionError:
            raise ValueError("R2, R3, R4 should be non-zero")
        return V1

    if V2 == None:
        try: 
            V2 = (Vout+V1*(R3/R1))*(((R2+R4)/R4)*(R1/(R1+R3)))
        except ZeroDivisionError:
            raise ValueError("Rf should be non-zero")
        return V2

    if Vout == None:
        try: 
            Vout = -V1*(R3/R1)+V2*(R4/(R2+R4))*((R1+R3)/R1)
        except ZeroDivisionError:
            raise ValueError("R1, R3, R4 should be non-zero")
        return Vout

    if R1 == None:
        try: 
            R1 = (R3*((V2*R4)/(R2+R4)-V1))/(Vout-(V2*R4)/(R2+R4))
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return R1

    if R2 == None:
        try: 
            R2 = (V2*R4*(R1+R3))/(R1*(Vout+V1*(R3/R1)))-R4
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return R2

    if R3 == None:
        try: 
            R3 = (R1*(Vout-(R4*V2)/(R2+R4)))/((R4*V2)/(R2+R4)-V1)
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return R3

    if R4 == None:
        try: 
            R4 = (R2*(R1*Vout+V1*R3))/(V2*(R1+R3)-Vout*R1-V1*R3)
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return R4

def inverting_summing(Vout=None, Rf=None, *inputs):
    """
    Inverting Summing Op-Amp Solver

    This function will accept  for the common operational amplifier
    formulas related to an inverting operational amplifier (op-amp) and will
    solve for the fourth value. All inputs default to `None`, to solve for any
    value, do not declare it as an argument and set all other inputs.

    .. image:: https://www.basictables.com/media/inverting-summing-opamp-circuit.png
    Examples
    --------
    >>> from electricpy.opamp import inverting
    >>> inverting(Vin=10, Rin=10, Rf=10) # Solve for Vout
    {...}
    >>> inverting(Vin=10, Vout=10, Rin=10) # Solve for Rf
    {...}

    Parameters
    ----------
    Vin:    [int, float]
            Input voltage, in volts. Default is None.
    Rf:     [int, float]
            Feedback resistor value, in ohms. Default is None.
    *inputs [int, float]
            Input voltages and resistances, given as pairs of unnamed arguments, V1, R1, V2, R2, ..., Vn, Rn. Voltage in volts, resistance in ohms.

    Returns
    -------
    int, float:  This function will return the unknown variable from the 
                 previously described set of variables.   
    """
    
    if len(inputs) < 2:
        raise ValueError("At least one input voltage and one input resistance must be provided")

    #Check if exactly one input is None
    if ([Vout,Rf] + list(inputs)).count(None) != 1:
        raise ValueError("Exactly one input should be None")

    #Check that each input has 2 values
    if len(inputs) % 2 != 0:
        raise ValueError("Each input must have a voltage and a resistance")

    if Rf != None and Rf < 0:
        raise ValueError("Rf must be positive")
    
    #Gets all odd *inputs (resistances) with list(inputs[0::2]), removes any None values with list comprehension, then checks for negatives with any() function.
    if any(x < 0 for x in [x for x in list(inputs[0::2]) if x is not None]):
        raise ValueError("Input resistances must be positive")

    if Vout == None:
        try: 
            Vout = 0
            for i in range(0,len(list(inputs)),2):
                Vout += inputs[i]/inputs[i+1]
            Vout = Vout*-Rf
        except ZeroDivisionError:
            raise ValueError("Input resistances should be non-zero")
        return Vout

    if Rf == None:
        try: 
            Rf = 0
            for i in range(0,len(list(inputs)),2):
                Rf += inputs[i]/inputs[i+1]
            Rf = -Vout/Rf
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return Rf

    if list(inputs).index(None) % 2 == 0:
        none_index = list(inputs).index(None)
        Rn = inputs[list(inputs).index(None)+1]
        try:
            #Vn_over_Rn refers to all Vn/Rn except the pair that has a None value
            Vn_over_Rn = 0
            for i in range(0,len(list(inputs)),2):
                if i != none_index and i != none_index+1:
                    Vn_over_Rn += list(inputs)[i]/list(inputs)[i+1]
                    print()
            Vn = Rn*(-(Vout/Rf)-Vn_over_Rn)
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return Vn

    if list(inputs).index(None) % 2 == 1:
        none_index = list(inputs).index(None)
        #Vn is the corresponding voltage of the missing resistance
        Vn = inputs[list(inputs).index(None)-1]
        try:
            #Vn_over_Rn refers to all Vn/Rn except the pair that has a None value
            Vn_over_Rn = 0
            for i in range(0,len(list(inputs)),2):
                if i != none_index and i != none_index-1:
                    Vn_over_Rn += list(inputs)[i]/list(inputs)[i+1]
            Rn = (-(Vout/Rf)-(Vn_over_Rn))/Vn
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return Rn




# print(inverting_summing(None,1,6,1,6,1,6,1))
# print(inverting_summing(-18,None,6,1,6,1,6,1))
# print(inverting_summing(-18,1,6,1,6,1,None,1))
print(inverting_summing(-18,1,6,1,6,1,6,None))
print(inverting_summing(-18,1,1,None))
# inverting_summing(10,300,None,100,12,10)
# print(non_inverting(3,5, 5,None))
# print(inverting(5,None,10,12))
# print("Balanced vout: ",  differential(3,3,None,10,10,10,10))
# print("R1 should be 10: ",  differential(3,3,0,None,10,10,10))
# print("R2 should be 10: ",  differential(3,3,0,10,None,10,10))
# print("R3 should be 10: ",  differential(3,3,0,10,10,None,10))
# print("R4 should be 10: ",  differential(3,3,0,10,10,10,None))
# print("vout: ",  differential(5,8,None,12,15,17,18))
# print("R1 should be 12: ",  differential(5,8,3.46212,None,15,17,18))
# print("R2 should be 15: ",  differential(5,8,3.46212,12,None,17,18))
# print("R3 should be 17: ",  differential(5,8,3.46212,12,15,None,18))
# print("R4 should be 18: ",  differential(5,8,3.46212,12,15,17,None))
