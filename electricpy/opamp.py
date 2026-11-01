################################################################################
"""
Functions for Operational Amplifiers (OpAmps).

>>> from electricpy import opamp
"""

################################################################################
# TODO: Add gain equations?

def non_inverting(Vin=None, Vout=None, Rg=None, Rf=None):
    """
    Non-Inverting Op-Amp Solver

    This function will accept three variables for the non-inverting operational
    amplifier (op-amp) circuit output voltage formula and will solve for the
    fourth value. All inputs default to `None`, to solve for any value, do not
    declare it as an argument and set all other inputs.

    .. math:: V_{out} = \\frac{V_{in}(R_g+R_f)}{Rg}

    .. image:: https://www.basictables.com/media/non-inverting-opamp-circuit.png

    Examples
    --------
    >>> from electricpy.opamp import non_inverting
    >>> non_inverting(Vin=5, Rg=15, Rf=10) # Solve for Vout
    8.333
    >>> non_inverting(Vin=5, Vout=8.333, Rg=15) # Solve for Rf
    10

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
    float:  This function will return the unknown variable from the previously described set of variables.   
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

    This function will accept three variables for the inverting operational
    amplifier (op-amp) circuit output voltage formula and will solve for the
    fourth value. All inputs default to `None`, to solve for any value, do not
    declare it as an argument and set all other inputs.

    .. math:: V_{out} = -\\frac{V_{in}\\cdot R_f}{R_{in}}

    .. image:: https://www.basictables.com/media/inverting-opamp-circuit.png

    Examples
    --------
    >>> from electricpy.opamp import inverting
    >>> inverting(Vin=5, Rin=10, Rf=15) # Solve for Vout
    -7.5
    >>> inverting(Vin=5, Vout=-7.5, Rin=10) # Solve for Rf
    15.0

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
    float:  This function will return the unknown variable from the previously described set of variables.   
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

    This function will accept six variables for the differential operational
    amplifier (op-amp) circuit output voltage formula and will solve for the
    seventh value. All inputs default to `None`, to solve for any value, do not
    declare it as an argument and set all other inputs.

    .. math:: V_{out} =-V_1 \\frac{R_3}{R_1}+V_2\\frac{R_4}{R_2+R_4}\\cdot \\frac{R_1+R_3}{R_1}

    .. image:: https://www.basictables.com/media/differential-opamp-circuit.png

    Examples
    --------
    >>> from electricpy.opamp import differential
    >>> differential(V1=10, V2=15, R1=10, R2=10, R3=20 , R4=3) # Solve for Vout
    -9.615
    >>> differential(V1=10,z V2=15, Vout=-9.615, R1=10, R2=10, R4=3) # Solve for R3
    19.9994

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
    float:  This function will return the unknown variable from the previously described set of variables.   
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

def inverting_summing(*inputs, Vout=None, Rf=None):
    """
    Inverting Summing Op-Amp Solver

    This function will accept any even number of variables for the differential operational
    amplifier (op-amp) circuit output voltage formula and will solve for the
    'None' value. Vout and Rf default to `None`, to solve for either value, do not
    declare it as an argument and set all other inputs. To solve for any input
    voltage or resistance, explicitly set it to 'None'. Input voltages and
    resistance arguments are positional.

    .. math:: V_{out}=-R_f(\\frac{V_1}{R_1}+\\frac{V_2}{R_2}\\ldots \\frac{V_n}{R_n})

    .. image:: https://www.basictables.com/media/inverting-summing-opamp-circuit.png
    Examples
    --------
    >>> from electricpy.opamp import inverting
    >>> inverting_summing(10, 6.3, 12, 33, 20, 63, Rf=10) # Solve for Vout
    -22.68398
    >>> inverting_summing(10, 6.3, 12, None , 20, 63, Vout=-22.68398, Rf=10) # Solve for R2
    33.0
    >>> inverting_summing(10, 6.3, None, 33, 20, 63, Vout=-22.68398, Rf=10) #Solve for V2
    11.9999

    Parameters
    ----------
    Vout:    [int, float]
            Output voltage, in volts. Default is None.
    Rf:     [int, float]
            Feedback resistor value, in ohms. Default is None.
    *inputs [int, float]
            Input voltages and resistances, given as pairs of unnamed arguments, V1, R1, V2, R2, ..., Vn, Rn. Voltage in volts, resistance in ohms.

    Returns
    -------
    float:  This function will return the unknown variable from the previously described set of variables.   
    """

    if len(inputs) < 2:
        raise ValueError("At least one input voltage and one input resistance must be provided")
    input_list = list(inputs)

    #Check if exactly one input is None
    if ([Vout,Rf] + input_list).count(None) != 1:
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
            for i in range(0,len(input_list),2):
                Vout += inputs[i]/inputs[i+1]
            Vout = Vout*-Rf
        except ZeroDivisionError:
            raise ValueError("Input resistances should be non-zero")
        return Vout

    if Rf == None:
        try: 
            Rf = 0
            for i in range(0,len(input_list),2):
                Rf += inputs[i]/inputs[i+1]
            Rf = -Vout/Rf
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return Rf

    none_index = input_list.index(None)

    if none_index % 2 == 0:
        #R1 is the corresponding resistance of the missing voltage
        R1 = inputs[none_index+1]
        try:
            #Vn_over_Rn refers to all Vn/Rn except the pair that has a None value
            Vn_over_Rn = 0
            for i in range(0,len(input_list),2):
                if i != none_index and i != none_index+1:
                    Vn_over_Rn += input_list[i]/input_list[i+1]
            V1 = R1*(-(Vout/Rf)-Vn_over_Rn)
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return V1

    if none_index % 2 == 1:
        #V1 is the corresponding voltage of the missing resistance
        V1 = inputs[none_index-1]
        try:
            #Vn_over_Rn refers to all Vn/Rn except the pair that has a None value
            Vn_over_Rn = 0
            for i in range(0,len(input_list),2):
                if i != none_index and i != none_index-1:
                    Vn_over_Rn += input_list[i]/input_list[i+1]
            R1 = V1/((-Vout/Rf)-Vn_over_Rn)
        except ZeroDivisionError:
            raise ValueError("Invalid arguments, results in division by zero error")
        return R1

def main():

    print(non_inverting(Vin=None,Vout=10,Rg=5,Rf=15)) #2.5
    print(non_inverting(Vin=2.5,Vout=None,Rg=5,Rf=15)) #10
    print(non_inverting(Vin=2.5,Vout=10,Rg=None,Rf=15)) #5
    print(non_inverting(Vin=2.5,Vout=10,Rg=5,Rf=None)) #15
    print()
    print(inverting(Vin=None,Vout=10,Rin=5,Rf=15)) #-3.33
    print(inverting(Vin=-3.33,Vout=None,Rin=5,Rf=15)) #9.99
    print(inverting(Vin=-3.33,Vout=10,Rin=None,Rf=15)) #5
    print(inverting(Vin=-3.33,Vout=10,Rin=5,Rf=None)) #15
    print()
    print(differential(V1=None,V2=1,Vout=-10,R1=1,R2=1,R3=10,R4=10)) #2.0
    print(differential(V1=2,V2=None,Vout=-10,R1=1,R2=1,R3=10,R4=10)) #1.0
    print(differential(V1=2,V2=1,Vout=None,R1=1,R2=1,R3=10,R4=10)) #-10.0
    print(differential(V1=2,V2=1,Vout=-10,R1=None,R2=1,R3=10,R4=10)) #1.0
    print(differential(V1=2,V2=1,Vout=-10,R1=1,R2=None,R3=10,R4=10)) #1.0
    print(differential(V1=2,V2=1,Vout=-10,R1=1,R2=1,R3=None,R4=10)) #10.0
    print(differential(V1=2,V2=1,Vout=-10,R1=1,R2=1,R3=10,R4=None)) #10.0
    print()
    print(inverting_summing(None,10,3,10,1,10,Vout=-6,Rf=10)) #2.0
    print(inverting_summing(2,None,3,10,1,10,Vout=-6,Rf=10)) #10.0
    print(inverting_summing(2,10,None,10,1,10,Vout=-6,Rf=10)) #3.0
    print(inverting_summing(2,10,3,None,1,10,Vout=-6,Rf=10)) #10.0
    print(inverting_summing(2,10,3,10,None,10,Vout=-6,Rf=10)) #1.0
    print(inverting_summing(2,10,3,10,1,None,Vout=-6,Rf=10)) #10.0
    print(inverting_summing(2,10,3,10,1,10,Vout=None,Rf=10)) #-6.0
    print(inverting_summing(2,10,3,10,1,10,Vout=-6,Rf=None)) #10.0

if __name__ == '__main__':
    main()
