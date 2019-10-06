###################################################################
"""
---------------------
`electricpy.fault.py`
---------------------

Included Functions
------------------
- Digital Filter Simulator:             digifiltersim
- Step Response Filter Simulator:       step_response
- Ramp Response Filter Simulator:       ramp_response
- Parabolic Response Filter Simulator:  parabolic_response
- State-Space System Simulator:         statespace
"""
###################################################################

# Import Required Libraries
import numpy as _np
import matplotlib.pyplot as _plt
from warnings import warn as _warn


# Define Digital Filter Simulator Function
def digifiltersim(fin,filter,freqs,NN=1000,dt=0.01,title="",
               legend=True,xlim=False,xmxscale=None,figsize=None):
    """
    Digital Filter Simulator
    
    Given an input function and filter parameters (specified in
    the z-domain) this function will plot the input function over
    NN time-steps of an unspecified size (the step-size must be
    specified outside the scope of this function). This function
    will also plot the resultant (output) function over the NN
    time-steps after having been filtered by that filter which
    is specified.
    
    The applied filter should be of the form:
    
    .. math:: \\frac{b_0+b_1z^{-1}+b_2z^{-2}}{1-a_1z^{-1}-a_2z^{-2}}
    
    Where each row corresponds to a 1- or 2-pole filter.
    
    Parameters
    ----------
    fin:        function
                The input function, must be callable with specified
                step-size.
    filter:     array_like
                The filter parameter set as shown here:
                
                .. code-block:: python
                   [[ a11, a12, b10, b11, b12],
                   [ a21, a22, b20, b21, b22],
                   [           ...          ],
                   [ an1, an2, bn0, bn1, bn2]]
                
    freqs:      list of float
                The set of frequencies to plot the input and output for.
    NN:         int, optional
                The number of time-steps to be plotted; default=1000
    dt:         float, optional
                The time-step size; default=0.01
    title:      str, optional
                The title presented on each plot; default=""
    xlim:       list, optional
                Limit in x-axis for graph plot. Accepts tuple of: (xmin, xmax).
                default=False.
    xmxscale:   float, optional
                Scaling limit of the x-axis, will set the maximum of the
                x-axis to xmxscale/(dt*freq) where freq is the current
                frequency being plotted.
    legend:     str, optional
                An argument to control whether the legend is shown,
                default=True.
    figsize:    tuple, optional
                The figure dimensions for each subplot, default=None
    """
    if(figsize!=None): _plt.figure(figsize=figsize)
    flen = len(freqs)
    for i in range(flen):
        # Gather frequency
        freq = freqs[i]
        
        # Start with arrays set to zero
        x = _np.zeros(NN)
        y = _np.zeros(NN)
        
        # ----- The input  -----
        for k in range(NN):
            x[k] = fin(k*dt,freq)
        
        # Identify how many rows were provided
        sz = filter.size
        if(sz < 5):
            raise ValueError("ERROR: Too few filter arguments provided. "+
                             "Refer to documentation for proper format.")
        elif(sz == 5):
            rows = 1
        else:
            rows, cols = filter.shape
        # Operate with each individual filter set
        x_tmp = _np.copy( x )
        nsteps = NN - 4
        for row_n in range(rows):
            row = filter[row_n] # Capture individual row
            A1 = row[0]
            A2 = row[1]
            B0 = row[2]
            B1 = row[3]
            B2 = row[4]
            T = 3
            for _ in range(nsteps):
                T = T + 1
                # Apply Filtering Specified by Individual Row
                y[T] = (A1*y[T-1] + A2*y[T-2] +
                        B0*x_tmp[T] + B1*x_tmp[T-1] +  B2*x_tmp[T-2])
            # Copy New output into temporary input
            x_tmp = _np.copy( y )
        # Copy finalized output into *ytime* for plotting
        ytime = _np.copy( x_tmp )
        # Plot Filtered Output
        if(flen%2==0): _plt.subplot(flen,2,i+1)
        else: _plt.subplot(flen,1,i+1)
        _plt.plot(x,'k--',label="Input")
        _plt.plot(ytime,'k',label="Output")
        _plt.title(title)
        _plt.grid(which='both')
        if legend: _plt.legend(title="Frequency = "+str(freq)+"Hz")
        if xlim!=False: _plt.xlim(xlim)
        elif xmxscale!=None: _plt.xlim((0,xmxscale/(freq*dt)))
        
    _plt.tight_layout()
    _plt.show()

# Define Step Response Simulator Function
def step_response(system,npts=1000,dt=0.01,combine=True,xlim=False,
                  title="Step Response",errtitle="Step Response Error",
                  resplabel="Step Response",funclabel="Step Function",
                  errlabel="Error",filename=None):
    """
    Step Function Response Plotter Function
    
    Given a transfer function, plots the response against step input
    and plots the error for the function.
    
    Parameters
    ----------
    system:     array_like
                The Transfer Function; can be provided as the following:
                * 1 (instance of lti)
                * 2 (num, den)
                * 3 (zeros, poles, gain)
                * 4 (A, B, C, D)
    npts:       int, optional
                Number of steps to calculate over; default is 1000.
    dt:         float, optional
                Difference between each data point, default is 0.01.
    combine:    bool, optional
                If combination of numerator and denominator is needed.
                This value should be set to "True" if the parts should be
                combined to show the complete system with feedback.
                default=True.
    title:      str, optional
                Additional string to be added to plot titles;
                default=""
    xlim:       list, optional
                Limit in x-axis for graph plot. Accepts tuple of: (xmin, xmax).
                default=False.
    filename:   bool, optional
                Control argument to specify whether the plotted
                figures should be saved. default=False
    """
    # Define Time Axis
    TT = _np.arange(0,npts*dt,dt)
    
    # Condition system input to ensure proper execution
    system = _sys_condition(system,combine)    
    
    # Allocate space for all outputs
    step = _np.zeros(npts)
    errS = _np.zeros(npts)
    
    # Generate Inputs
    for i in range(npts):
        step[i] = 1.0
    
    # Simulate Response for each input (step, ramp, parabola)
    # All 'x' values are variables that are considered don't-care
    x, y1, x = sig.lsim((system),step,TT)
    
    # Calculate error over all points
    for k in range(npts):
        errS[k] = step[k] - y1[k]
    
    # Plot Step Response
    _plt.figure()
    _plt.subplot(121)
    _plt.title(title)
    _plt.plot(TT,y1,'k--', label=resplabel)
    _plt.plot(TT,step,'k', label=funclabel)
    _plt.grid()
    _plt.legend()
    _plt.xlabel("Time (seconds)")
    if xlim != False:
        _plt.xlim(xlim)
    _plt.subplot(122)
    _plt.title(errtitle)
    _plt.plot(TT,errS,'k', label=errlabel)
    _plt.grid()
    _plt.legend()
    _plt.xlabel("Time (seconds)")
    if xlim != False:
        _plt.xlim(xlim)
    _plt.subplots_adjust(wspace=0.3)
    if filename!=None:
        _plt.savefig(filename)
    _plt.show()

# Define Ramp Response Simulator Function
def ramp_response(system,npts=1000,dt=0.01,combine=True,xlim=False,
                  title="Ramp Response",errtitle="Ramp Response Error",
                  resplabel="Ramp Response",funclabel="Ramp Function",
                  errlabel="Error",filename=None):
    """
    Ramp Function Response Plotter Function
    
    Given a transfer function, plots the response against step input
    and plots the error for the function.
    
    Parameters
    ----------
    system:     array_like
                The Transfer Function; can be provided as the following:
                * 1 (instance of lti)
                * 2 (num, den)
                * 3 (zeros, poles, gain)
                * 4 (A, B, C, D)
    npts:       int, optional
                Number of steps to calculate over; default is 1000.
    dt:         float, optional
                Difference between each data point, default is 0.01.
    combine:    bool, optional
                If combination of numerator and denominator is needed.
                This value should be set to "True" if the parts should be
                combined to show the complete system with feedback.
                default=True.
    title:      str, optional
                Additional string to be added to plot titles;
                default=""
    xlim:       list, optional
                Limit in x-axis for graph plot. Accepts tuple of: (xmin, xmax).
                default=False.
    filename:   str, optional
                File directory/name with which the plotted figures 
                should be saved. default=None
    """
    # Define Time Axis
    TT = _np.arange(0,npts*dt,dt)
    
    # Condition system input to ensure proper execution
    system = _sys_condition(system,combine)
    
    # Allocate space for all outputs
    ramp = _np.zeros(npts)
    errR = _np.zeros(npts)
    
    # Generate Inputs
    for i in range(npts):
        ramp[i] = (dt*i)
    
    # Simulate Response for each input (step, ramp, parabola)
    # All 'x' values are variables that are considered don't-care
    x, y2, x = sig.lsim((system),ramp,TT)
    
    # Calculate error over all points
    for k in range(npts):
        errR[k] = ramp[k] - y2[k]
    
    # Plot Ramp Response
    _plt.figure()
    _plt.subplot(121)
    _plt.title(title)
    _plt.plot(TT,y2,'k--', label=resplabel)
    _plt.plot(TT,ramp,'k', label=funclabel)
    _plt.grid()
    _plt.legend()
    _plt.xlabel("Time (seconds)")
    if xlim != False:
        _plt.xlim(xlim)
    _plt.subplot(122)
    _plt.title(errtitle)
    _plt.plot(TT,errR,'k', label=errlabel)
    _plt.grid()
    _plt.legend()
    _plt.xlabel("Time (seconds)")
    if xlim != False:
        _plt.xlim(xlim)
    _plt.subplots_adjust(wspace=0.3)
    if filename!=None:
        _plt.savefig(filename)
    _plt.show()
    
# Define Parabolic Response Simulator Function
def parabolic_response(system,npts=1000,dt=0.01,combine=True,xlim=False,
                  title="Parabolic Response",errtitle="Parabolic Response Error",
                  resplabel="Parabolic Response",funclabel="Parabolic Function",
                  errlabel="Error",filename=None):
    """
    Parabolic Function Response Plotter Function
    
    Given a transfer function, plots the response against step input
    and plots the error for the function.
    
    Parameters
    ----------
    system:     array_like
                The Transfer Function; can be provided as the following:
                * 1 (instance of lti)
                * 2 (num, den)
                * 3 (zeros, poles, gain)
                * 4 (A, B, C, D)
    npts:       int, optional
                Number of steps to calculate over; default is 1000.
    dt:         float, optional
                Difference between each data point, default is 0.01.
    combine:    bool, optional
                If combination of numerator and denominator is needed.
                This value should be set to "True" if the parts should be
                combined to show the complete system with feedback.
                default=True.
    title:      str, optional
                Additional string to be added to plot titles;
                default=""
    xlim:       list, optional
                Limit in x-axis for graph plot. Accepts tuple of: (xmin, xmax).
                default=False.
    filename:   bool, optional
                Control argument to specify whether the plotted
                figures should be saved. default=False
    """
    # Define Time Axis
    TT = _np.arange(0,npts*dt,dt)
    
    # Condition system input to ensure proper execution
    system = _sys_condition(system,combine)
    
    # Allocate space for all outputs
    parabola = _np.zeros(npts)
    errP = _np.zeros(npts)
    
    # Generate Inputs
    for i in range(npts):
        parabola[i] = (dt*i)**(2)
    
    # Simulate Response for each input (step, ramp, parabola)
    # All 'x' values are variables that are considered don't-care
    x, y3, x = sig.lsim((system),parabola,TT)
    
    # Calculate error over all points
    for k in range(npts):
        errP[k] = parabola[k] - y3[k]
    
    # Plot Parabolic Response
    _plt.figure()
    _plt.subplot(121)
    _plt.title(title)
    _plt.plot(TT,y3,'k--', label=resplabel)
    _plt.plot(TT,parabola,'k', label=funclabel)
    _plt.grid()
    _plt.legend()
    _plt.xlabel("Time (seconds)")
    if xlim != False:
        _plt.xlim(xlim)
    _plt.subplot(122)
    _plt.title(errtitle)
    _plt.plot(TT,errP,'k', label=errlabel)
    _plt.grid()
    _plt.legend()
    _plt.xlabel("Time (seconds)")
    if xlim != False:
        _plt.xlim(xlim)
    _plt.subplots_adjust(wspace=0.3)
    if filename!=None:
        _plt.savefig(filename)
    _plt.show()

# Define State Space Simulator
def statespace(A,B,x=None,func=None,C=None,D=None,simpts=9999,NN=10000,dt=0.01,
               xlim=False,ylim=False,title="",ret=False,plotstate=True,
               plotforcing=None,plotresult=None,filename=None):
    """
    State-Space Simulation Plotter

    Parameters
    ----------
    A:          array_like
                Matrix A of State-Space Formulation
    B:          array_like
                Matrix B of State-Space Formulation
    x:          array_like, optional
                Initial Condition Matrix, if not provided, will assume
                initial conditions of zero.
    f:          function, optional
                State-Space Forcing Function; callable function that
                will return any/all forcing function Arguments needed as
                array-like object.
                Forcing function(s) can be provided as tuple of function
                handles, system will automatically concatenate their output
                to a matrix that can be handled.
    simpts:     int, optional
                Changes the range of simulation; defualt=9999
    NN:         int, optional
                Number of descrete points; default=10,000
    dt:         float, optional
                Time-step-size; default=0.01
    xlim:       list of float, optional
                Limit in x-axis for graph plot.
    ylim:       list of float, optional
                Limit in y-axis for graph plot.
    title:      str, optional
                Additional String for Plot Title
    ret:        bool, optional
                Control value to specify whether the state space terms should
                be returned.
    plot:       bool, optional
                Control value to enable/disable all plotting capabilities.
    plotforcing:bool, optional
                Control value to enable plotting of the forcing function(s)
    plotresult: bool, optional
                Control value to enable plotting of the final (combined) result.

    Figures:
    --------
    Forcing Functions:        The plot of forcing functions, only provided if plotforcing is true.
    State Variables:        The plot of state variables, always provided if plot is true.
    Combined Output:        The plot of the combined terms in the output, provided if C and D are not False.

    """
    # Define Initial Condition for Solution
    solution = 3
    #0=zero-input    ( No Forcing Function )
    #1=zero-state    ( No Initial Conditions )
    #2=total         ( Both Initial Conditions and Forcing Function )
    #3=total, output ( Both ICs and FFs, also plot combined output )
    
    # Tuple to Matrix Converter
    def tuple_to_matrix(x,yx):
        n = yx(x) # Evaluate function at specified point
        n = _np.asmatrix(n) # Convert tuple output to matrix
        n = n.T # Transpose matrix
        return(n)

    # Numpy Array to Matrix Converter
    def nparr_to_matrix(x,yx):
        n = yx(x) # Evaluate function at specified point
        n = _np.asmatrix(n) # Convert _np.arr output to matrix
        if n.shape[1] != 1: # If there is more than 1 column
            n = _np.matrix.reshape(n,(n.size,1)) # Reshape
        return(n)
    
    # Define Function Concatinator Class
    class c_func_concat:
        def __init__(self,funcs): # Initialize class with tupple of functions
            self.nfuncs = len(funcs) # Determine how many functions are in tuple
            self.func_reg = {} # Create empty keyed list of function handles
            for key in range(self.nfuncs): # Iterate adding to key
                self.func_reg[key] = funcs[key] # Fill keyed list with functions

        def func_c(self,x): # Concatenated Function
            rets = _np.array([]) # Create blank numpy array to store function outputs
            for i in range(self.nfuncs):
                y = self.func_reg[i](x) # Calculate each function at value x
                rets = _np.append(rets, y) # Add value to return array
            rets = _np.asmatrix(rets).T # Convert array to matrix, then transpose
            return(rets)
    
    # Condition Inputs
    A = _np.asmatrix(A)
    B = _np.asmatrix(B)
    
    # Define Tuple of Types For Testing
    typetest = (_np.matrixlib.defmatrix.matrix,_np.ndarray,tuple,list)
    
    # Test for NN and simpts
    if (simpts >= NN):
        _warn("WARNING: NN must be greater than simpts; NN="+str(NN)+
             "simpts="+str(simpts)," Autocorrecting simpts to be NN-1.")
        simpts = NN-1

    # Test for C and D matricies
    if isinstance(C,typetest) and isinstance(D,typetest):
        solution = 3 # Set to solve and plot complete output
    elif isinstance(C,typetest) and not isinstance(D,typetest):
        if (D==None):
            _warn("WARNING: D matrix not provided; D now assumed to be 0.")
            D = _np.matrix('0')
            solution = 3 # Set to solve and plot complete output
    else:
        C = _np.matrix('0')
        D = _np.matrix('0')
        solution = 2
    
    # Condition C/D Matrices
    C = _np.asmatrix(C)
    D = _np.asmatrix(D)


    # Create values for input testing
    if isinstance(func,function): # if f is a function, test as one
        mF = f(1) # f should return: int, float, tuple, _np.arr, _np.matrix
    elif isinstance(func,(tuple,list)): # if f is tupple of arguments
        if isinstance(func[0], function): #if first argument is a function
            c_funcs = c_func_concat(func) # concatinate functions into one
            mF = "MultiFunctions" # label as multiple concatenated functions
        else:
            mF = "NA" # Can't handle function type
    else:
        mF = "NA" # Can't handle function type

    # Test for x input
    if not isinstance(x,typetest):
        if x==None: # No specified initial conditions
            rA = A.shape[0]
            x = _np.asmatrix(_np.zeros(rA)).T
    # Condition x
    x = _np.asmatrix(x)

    # Gather dimensions of inputs
    rA, cA = A.shape
    rB, cB = B.shape
    rx, cx = x.shape
    rC, cC = C.shape
    rD, cD = D.shape
    rF, cF = 1, 1 # Defualt for a function returning one value

    if isinstance(mF,tuple): # If function returns tuple
        fn = lambda x: tuple_to_matrix(x, func) # Use conversion function
        rF, cF = fn(1).shape # Prepare for further testing
    elif isinstance(mF,_np.ndarray): # If function returns numpy array
        fn = lambda x: nparr_to_matrix(x, func) # Use conversion function
        rF, cF = fn(1).shape # Prepare for further testing
    elif isinstance(mF,(int,float,_np.float64)): # If function returns int or float or numpy float
        fn = f # Pass function handle
    elif isinstance(mF,_np.matrixlib.defmatrix.matrix): # If function returns matrix
        fn = f # Pass function handle
        rF, cF = fn(1).shape # Prepare for further testing
    elif (mF=="MultiFunctions"): # There are multiple functions in one argument
        fn = c_funcs.func_c # Gather function handle from function concatenation class
        rF, cF = fn(1).shape # Prepare for further testing
    elif (mF=="NA"): # Function doesn't meet requirements
        raise ValueError("Forcing function does not meet requirements."+
                        "\nFunction doesn't return data type: int, float, numpy.ndarray"+
                        "\n or numpy.matrixlib.defmatrix.matrix. Nor does function "+
                        "\ncontain tuple of function handles. Please review function.")

    # Test for size correlation between matricies
    if (cA != rA): # A isn't nxn matrix
        raise ValueError("Matrix 'A' is not NxN matrix.")
    elif (rA != rB): # A and B matricies don't have same number of rows
        if (B.size % rA) == 0: # Elements in B divisible by rows in A
            _warn("WARNING: Reshaping 'B' matrix to match 'A' matrix.")
            B = _np.matrix.reshape(B,(rA,int(B.size/rA))) # Reshape Matrix
        else:
            raise ValueError("'A' matrix dimensions don't match 'B' matrix dimensions.")
    elif (rA != rx): # A and x matricies don't have same number of rows
        if (x.size % rA) == 0: # Elements in x divisible by rows in A
            _warn("WARNING: Reshaping 'x' matrix to match 'A' matrix.")
            x = _np.matrix.reshape(x,(rA,1)) # Reshape Matrix
        else:
            raise ValueError("'A' matrix dimensions don't match 'B' matrix dimensions.")
    elif (cB != rF) or (cF != 1): # Forcing Function matrix doesn't match B matrix
        raise ValueError("'B' matrix dimensions don't match forcing function dimensions.")
    elif (solution==3) and (cC != cA) or (rC != 1): # Number of elements in C don't meet requirements
        raise ValueError("'C' matrix dimensions don't match state-space variable dimensions.")
    elif (solution==3) and ((cD != rF) or (rD != 1)): # Number of elements in D don't meet requirements
        if (cD == rD) and (cD == 1) and (D[0] == 0): # D matrix is set to [0]
            D = _np.asmatrix(_np.zeros(rF)) # Re-create D to meet requirements
            _warn("WARNING: Autogenerating 'D' matrix of zeros to match forcing functions.")
        else:
            raise ValueError("'D' matrix dimensions don't match forcing function dimensions.")

    # Test for forcing function
    if (f==None) and (solution!=0):
        solution = 0 # Change to Zero-Input calculation

    # Start by defining Constants
    T = 0
    TT = _np.arange(0,(dt*(NN)),dt)
    yout = 0

    # Define list of strings for plot output
    soltype = ["(Zero-Input)","(Zero-State)","(Complete Simulation)","(Complete Sim., Combined Output)"]

    # Create a dictionary of state-space variables
    xtim = {}
    xtim_len = rA # Number of Rows in A matrix
    for n in range(xtim_len):
        key = n #Each key should be the iterative variable
        xtim_init = _np.zeros(NN) #Define the initial array
        xtim[key] = xtim_init #Create each xtim

    # Create a dictionary of function outputs
    if (mF!=tint) and (mF!=tfloat):
        fn_arr = {}
        for n in range(rF):
            key = n #Each key should be the iterative variable
            fn_init = _np.zeros(NN) #Define the initial array
            fn_arr[key] = fn_init #Create each fn_arr
            fnc = rF
    else:
        fn_arr = _np.zeros(NN) #Create the fn_arr
        fnc = 1

    # When asked to find zero-state, set all ICs to zero
    if solution == 1:
        for n in range(xtim_len):
            x[n] = 0 #Set each value to zero

    # Finite-Difference Simulation
    for i in range(0,simpts):
        for n in range(xtim_len):
            xtim[n][i] = x[n] #xtim[state-variable][domain] = x[state-variable]
        # Create Forcing Function output

        if fnc > 1: # More than one forcing function
            for n in range(fnc):
                fn_arr[n][i] = _np.asarray(fn(T))[n][0]
        else: # only one forcing function
            fn_arr[i] = fn(T)

        if solution == 0: #Zero-input, no added function input
            x = x + dt*A*x
        else: #Zero-state or Total, add function input
            x = x + dt*A*x + dt*B*fn(T)
            if solution==3:
                yout = yout + dt*D*fn(T)

        T = T+dt #Add discrete increment to T

    # Plot Forcing Functions
    if (plotforcing):
        fffig = _plt.figure("Forcing Functions")
        if fnc > 1:
            for x in range(fnc):
                _plt.plot(TT,fn_arr[x],label="f"+str(x+1))
        else:
            _plt.plot(TT,fn_arr,label="f1")
        if xlim!=False:
            _plt.xlim(xlim)
        if ylim!=False:
            _plt.ylim(ylim)
        _plt.title("Forcing Functions "+title)
        _plt.xlabel("Time (seconds)")
        _plt.legend(title="Forcing Functions")
        _plt.grid()
        if filename!=None:
            _plt.savefig('Simulation Forcing Functions.png')
        if plot:
            _plt.show()

    # Plot each state-variable over time
    stvfig = _plt.figure("State Variables")
    for x in range(xtim_len):
        _plt.plot(TT,xtim[x],label="x"+str(x+1))
    if xlim!=False:
            _plt.xlim(xlim)
    if ylim!=False:
        _plt.ylim(ylim)
    _plt.title("Simulated Output Terms "+soltype[solution]+title)
    _plt.xlabel("Time (seconds)")
    _plt.legend(title="State Variable")
    _plt.grid()
    if filename!=None:
        _plt.savefig('Simulation Terms.png')
    if plot:
        _plt.show()

    # Plot combined output
    if (plotresult and solution==3):
        cofig = _plt.figure("Combined Output")
        C = _np.asarray(C) # convert back to array for operation
        for i in range(cC):
            yout = yout + xtim[i]*C[0][i] # Sum all st-space var mult. by their coeff
        yout = _np.asarray(yout) # convert output to array for plotting purposes
        _plt.plot(TT,yout[0])
        if xlim!=False:
            _plt.xlim(xlim)
        if ylim!=False:
            _plt.ylim(ylim)
        _plt.title("Combined Output "+title)
        _plt.xlabel("Time (seconds)")
        _plt.grid()
        if filename!=None:
            _plt.savefig('Simulation Combined Output.png')
        if plot:
            _plt.show()

    # Return Variables if asked to
    if ret:
        return(TT, xtim)

# END OF FILE