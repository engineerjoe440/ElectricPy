import pytest
import numpy as np
import cmath
import math
from numpy.testing import assert_almost_equal

def test_phasor():
    from electricpy.phasor import phasor
    magnitude = 10
    # basic angles test case 0
    z1 = phasor(magnitude, 0)
    z2 = phasor(magnitude, 30)
    z3 = phasor(magnitude, 45)
    z4 = phasor(magnitude, 60)
    z5 = phasor(magnitude, 90)

    assert_almost_equal(z1, complex(magnitude, 0))
    assert_almost_equal(z2, complex(magnitude * np.sqrt(3) / 2, magnitude / 2))
    assert_almost_equal(z3, complex(magnitude / np.sqrt(2), magnitude / np.sqrt(2)))
    assert_almost_equal(z4, complex(magnitude / 2, magnitude * np.sqrt(3) / 2))
    assert_almost_equal(z5, complex(0, magnitude))

    # z(theta) = z(theta+360) test case 1
    theta = np.random.randint(360)
    assert_almost_equal(phasor(magnitude, theta), phasor(magnitude, theta + 360))

    # z(-theta)*z(theta) == abs(z)^2 test case 2.
    z0 = phasor(magnitude, theta)
    z1 = phasor(magnitude, -theta)
    assert_almost_equal(z0 * z1, np.power(abs(z0), 2))

    # z(theta+180) = -1*Z(theta)
    z0 = phasor(magnitude, theta)
    z1 = phasor(magnitude, 180 + theta)
    assert_almost_equal(z0, -1 * z1)

def test_bridge_impedance():
    # Perfectly Balanced Wheat Stone Bridge
    from electricpy import bridge_impedance

    z1 = complex(1, 0)
    z2 = complex(1, 0)
    z3 = complex(1, 0)
    z4 = complex(1, 0)
    z5 = complex(np.random.random(), np.random.random())

    zeq = bridge_impedance(z1, z2, z3, z4, z5)

    zactual = complex(1, 0)

    assert zeq == zactual

    # Balanced Wheat Stone Bridge
    z1 = complex(2, 0)
    z2 = complex(4, 0)
    z3 = complex(8, 0)
    z4 = complex(4, 0)
    z5 = complex(np.random.random(), np.random.random())

    zeq = bridge_impedance(z1, z2, z3, z4, z5)

    zactual = complex(4, 0)

    assert zeq == zactual

    # Base Case
    z1 = complex(10, 0)
    z2 = complex(20, 0)
    z3 = complex(30, 0)
    z4 = complex(40, 0)
    z5 = complex(50, 0)

    zeq = bridge_impedance(z1, z2, z3, z4, z5)

    zactual = complex(4 + (50 / 3), 0)

    assert_almost_equal(zeq, zactual)

def test_dynetz():

    from electricpy import dynetz
    z1 = complex(3, 3)
    z2 = complex(3, 3)
    z3 = complex(3, 3)

    za, zb, zc = dynetz(delta=(z1, z2, z3))

    assert (za, zb, zc) == (z1 / 3, z2 / 3, z3 / 3)

    za, zb, zc = dynetz(wye=(z1, z2, z3))

    assert (za, zb, zc) == (3 * z1, 3 * z2, 3 * z3)

def test_powerset():

    from electricpy import powerset

    # Test case 0
    P = 10
    PF = 0.8
    _, Q, S, _ = powerset(P=P, PF=PF)

    assert_almost_equal(S, P / PF)
    assert_almost_equal(Q, S * np.sqrt(1 - PF ** 2))

    # Test case 1
    Q = 8
    P = 6

    _, _, S, PF = powerset(P=P, Q=Q)

    assert_almost_equal(S, 10)
    assert_almost_equal(PF, 0.6)

    #Test case 2 Failed

    # P = 0
    # Q = 0

    # _,_,S,PF = powerset(P = P, Q = Q)

    # assert_almost_equal(S,0)
    # assert_almost_equal(PF,0)

    #Test case 3 input validation is required
    # P = 10
    # Q = 10
    # PF = 0.6

    # _,_,S,_ = powerset(P = P, Q = Q, PF = PF)
    # assert_almost_equal(S,10*np.sqrt(2))
    # assert_almost_equal(PF,1/(np.sqrt(2)))

def test_voltdiv():
    from electricpy import voltdiv
    from electricpy.phasor import phasor

    # Test case 0 R1 == R2 == Rload
    Vin = 10
    R1 = 10
    R2 = 10
    Rload = 10

    Vout = voltdiv(Vin, R1, R2, Rload=Rload)

    assert Vout == R1 / 3

    # Test case 1 Vin, R1 and  R2 are in complex form
    Vin = phasor(220, 30)

    R1 = complex(10, 0)
    R2 = complex(10, 10)
    Rload = complex(10, -10)
    Vout = voltdiv(Vin, R1, R2, Rload=Rload)

    Vout_actual = phasor(110, 30)

    assert_almost_equal(Vout, Vout_actual)

def test_suspension_insulators():
    """Electric Power Systems by C.L Wadhwa Overhead Line Insulator example."""
    from electricpy import suspension_insulators

    number_capacitors = 5

    capacitance_ration = 5

    Voltage = 66

    _, string_efficiency = suspension_insulators(number_capacitors,
                                                capacitance_ration, 
                                                Voltage)

    string_efficiency_actual = 54.16
    assert_almost_equal(string_efficiency, string_efficiency_actual, decimal=2)

def test_propagation_constants():

    from electricpy import propagation_constants
    z = complex(0.5, 0.9)
    y = complex(0, 6e-6)
    params_dict = propagation_constants(z, y, 520)

    alpha_cal = 0.622 * (10 ** -3)
    beta_cal = 2.4 * (10 ** -3)

    assert_almost_equal(params_dict['alpha'], alpha_cal, decimal = 4)
    assert_almost_equal(params_dict['beta'], beta_cal, decimal = 4)

def test_funcrms():

    from electricpy.math import funcrms

    f = lambda x:np.sin(x)

    assert_almost_equal(funcrms(f,np.pi), 1/np.sqrt(2))

def test_convolve():

    from electricpy.math import convolve
    A = (1,1,1)
    B = (1,1,1)

    assert ([1,2,3,2,1] == convolve((A,B))).all()

def test_ic_555_astable():

    from electricpy import ic_555_astable

    # Astable configuration
    R = [10, 10]
    C = 1e-6
    result = ic_555_astable(R, C)

    for key,value in result.items():
        result[key] = np.round(value, decimals=6)

    assert_almost_equal(result['duty_cycle'], 66.666666667, decimal = 3)
    assert_almost_equal(result['t_low'], 6.931*10**-6, decimal = 3)
    assert_almost_equal(result['t_high'], 1.386*10**-5, decimal = 3)

    #test the other way around
def test_powerflow():
    from electricpy import powerflow
    from electricpy.phasor import phasor
    
    def test_0():
        Vsend = phasor(1.01, 30)
        Vrecv = phasor(1, 0)
        Xline = 0.2

        ans = powerflow(Vsend, Vrecv, Xline)
        assert_almost_equal(ans, 2.525)

    def test_1():
        Vsend = 1.01
        Vrecv = 1
        Xline = 0.2

        ans = powerflow(Vsend, Vrecv, Xline)
        assert ans == 0

    for i in range(2):
        exec(f"test_{i}()")

def test_slew_rate():

    from electricpy import slew_rate

    SR = slew_rate(V=1, freq=1, find='SR')
    V = slew_rate(SR=1, freq=1, find='V')
    freq = slew_rate(V=1, SR=1, find='freq')

    assert_almost_equal(np.pi*2, SR)
    assert_almost_equal(1/(np.pi*2), V)
    assert_almost_equal(1/(np.pi*2), freq)

def test_phs():
    from electricpy.phasor import phs
    def test_0():
        

        inputs = [0, 90, 180, 270, 360]

        outputs = [phs(x) for x in inputs]
        actual_outputs = [1, 1j, -1, -1j, 1]

        for x,y in zip(outputs, actual_outputs):
            assert_almost_equal(x, y)

    def test_1():
        inputs = [30, 45, 60, 135]

        outputs = [phs(x) for x in inputs]
        actual_outputs = [0.866025+0.5j, 0.707106+0.707106j, 0.5+0.866025j, -0.707106+0.707106j]

        for x,y in zip(outputs, actual_outputs):
            assert_almost_equal(x, y, decimal = 3)

    test_0()
    test_1()

def test_induction_motor_circle():
    from electricpy.visu import InductionMotorCircle

    open_circuit_test_data = {'V0': 400, 'I0': 9, 'W0': 1310}
    blocked_rotor_test_data = {'Vsc': 200, 'Isc': 50, 'Wsc': 7100}
    ratio = 1  # stator copper loss/ rotor copper loss
    output_power = 15000
    # InductionMotorCircle(open_circuit_test_data, blocked_rotor_test_data, output_power, torque_ration=ratio)
    #
    MotorCircle = InductionMotorCircle(open_circuit_test_data, blocked_rotor_test_data,
                        output_power, torque_ration=ratio, frequency=50, poles=4)

    assert_almost_equal(MotorCircle()['no_load_loss'], open_circuit_test_data['W0'])

def test_t_attenuator():
    Adb = 1
    Z0 = 1

    from electricpy import t_attenuator

    R1, R2 = t_attenuator(Adb, Z0)

    assert_almost_equal(R1,  0.0575, decimal = 3)
    assert_almost_equal(R2, 8.6673, decimal = 3)

def test_pi_attenuator():
    Adb = 1
    Z0 = 1

    from electricpy import pi_attenuator

    R1, R2 = pi_attenuator(Adb, Z0)
    assert_almost_equal(R1, 17.39036, decimal = 3)
    assert_almost_equal(R2, 0.11538, decimal = 3)

def test_inductor_voltdiv():

    from electricpy.passive import inductive_voltdiv

    params = {
        'Vin':1,
        'L1':1,
        'L2':1
    }
    Vout = inductive_voltdiv(**params, find='Vout')
    assert (Vout == params['Vin']/2)

    params = {
        'Vout':1,
        'L1':1,
        'L2':1
    }

    Vin = inductive_voltdiv(**params, find = 'Vin')
    assert (Vin == params['Vout']*2)

    params = {
        'Vout':1,
        'Vin':2,
        'L2':1
    }

    L1 = inductive_voltdiv(**params, find='L1')
    assert(L1 == 1)

    params = {
        'Vout':1,
        'Vin':2,
        'L1':1
    }
    L1 = inductive_voltdiv(**params, find='L1')
    assert(L1 == 1)

def test_induction_machine_slip():
    from electricpy import induction_machine_slip

    Nr = 1200
    freq = 50
    p = 4

    # Test case 1
    assert induction_machine_slip(0) == 1
    assert induction_machine_slip(1800) == 0

    # Test case 2
    assert induction_machine_slip(1000) == 4/9
    assert induction_machine_slip(1200) == 1/3

    # Test case 3
    assert induction_machine_slip(Nr, freq=freq, poles=p) == 0.2
    assert induction_machine_slip(1500, freq=freq, poles=p) == 0
    assert induction_machine_slip(0, freq=freq, poles=p) == 1

def test_abc_to_seq():

    from electricpy.conversions import abc_to_seq
    a = cmath.rect(1, np.radians(120))

    def test_0():
        np.testing.assert_array_almost_equal(abc_to_seq([1, 1, 1]), [1+0j, 0j, 0j])
        np.testing.assert_array_almost_equal(abc_to_seq([1, 0, 0]), [1/3+0j, 1/3+0j, 1/3+0j])
        np.testing.assert_array_almost_equal(abc_to_seq([0, 1, 0]), [1/3+0j, a/3+0j, a*a/3+0j])
        np.testing.assert_array_almost_equal(abc_to_seq([0, 0, 1]), [1/3+0j, a*a/3, a/3])

    test_0()

def test_seq_to_abc():
    from electricpy.conversions import seq_to_abc
    a = cmath.rect(1, np.radians(120))    

    def test_0():
        np.testing.assert_array_almost_equal(seq_to_abc([1, 1, 1]), [3+0j, 0j, 0j])
        np.testing.assert_array_almost_equal(seq_to_abc([1, 0, 0]), [1+0j, 1+0j, 1+0j])
        np.testing.assert_array_almost_equal(seq_to_abc([0, 1, 0]), [1+0j, a*a+0j, a+0j])
        np.testing.assert_array_almost_equal(seq_to_abc([0, 0, 1]), [1+0j, a, a*a])

    test_0()

def test_vectarray():

    from electricpy.phasor import vectarray

    def test_0():
        
        A = [2+3j, 4+5j, 6+7j, 8+9j]
        B = vectarray(A)

        B_test = [[np.abs(x), np.degrees(np.angle(x))] for x in A]

        np.testing.assert_array_almost_equal(B, B_test)

    def test_1():

        A = np.random.random(size = 16)
        B = vectarray(A)

        B_test = [[np.abs(x), 0] for x in A]
        np.testing.assert_array_almost_equal(B, B_test)

        A = np.random.random(size = 16)*1j
        B = vectarray(A)

        B_test = [[np.abs(x), 90] for x in A]
        np.testing.assert_array_almost_equal(B, B_test)

    test_0()
    test_1()

def test_parallel_plate_capacitance():
    from electricpy import parallel_plate_capacitance

    # Test 1: In the free space (by default e=e0=8.8542E-12)

    A1 = 100e-4
    d1 = 8.8542e-2
    C1 = 1e-12

    # Test capacitance given area and distance
    assert_almost_equal(parallel_plate_capacitance(A=A1, d=d1), C1)
    # Test area given capacitance and distance
    assert_almost_equal(parallel_plate_capacitance(C=C1, d=d1), A1)
    # Test distance given capacitance and area
    assert_almost_equal(parallel_plate_capacitance(C=C1, A=A1), d1)

    # Test 2: Not in the free space (e≠8.8542E-12)

    A2 = 100e-4
    d2 = 8.8542e-2
    e2 = 17.7084e-12
    C2 = 2e-12

    # Test capacitance given area, distance and permitivity
    assert_almost_equal(parallel_plate_capacitance(A=A2, d=d2, e=e2), C2)
    # Test area given capacitance, distance and permitivity
    assert_almost_equal(parallel_plate_capacitance(C=C2, d=d2, e=e2), A2)
    # Test distance given capacitance, area and permitivity
    assert_almost_equal(parallel_plate_capacitance(C=C2, A=A2, e=e2), d2)

def test_solenoid_inductance():
    from electricpy import solenoid_inductance

    # Test 1: In the free space (by default u=u0=4πE-7)

    A1 = 2e-3
    N1 = 550
    l1 = 20e-2*np.pi
    L1 = 1.21e-3

    # Test inductance given area, number of turns and length
    assert_almost_equal(solenoid_inductance(A=A1, N=N1, l=l1), L1)
    # Test area given inductance, number of turns and length
    assert_almost_equal(solenoid_inductance(L=L1, N=N1, l=l1), A1)
    # Test number of turns given inductance, area and length
    assert_almost_equal(solenoid_inductance(L=L1, A=A1, l=l1), N1)
    # Test length given inductance, area and number of turns
    assert_almost_equal(solenoid_inductance(L=L1, A=A1, N=N1), l1)

    # Test 2: Not the free space (u≠4πE-7)

    A2 = 2e-3
    N2 = 550
    l2 = 20e-2*np.pi
    u2 = 100*4e-7*np.pi # Iron permeability
    L2 = 0.121

    # Test inductance given area, number of turns, length and permeability
    assert_almost_equal(solenoid_inductance(A=A2, N=N2, l=l2, u=u2), L2)
    # Test area given inductance, number of turns, length and permeability
    assert_almost_equal(solenoid_inductance(L=L2, N=N2, l=l2, u=u2), A2)
    # Test number of turns given inductance, area, length and permeability
    assert_almost_equal(solenoid_inductance(L=L2, A=A2, l=l2, u=u2), N2)
    # Test length given inductance, area, number of turns and permeability
    assert_almost_equal(solenoid_inductance(L=L2, A=A2, N=N2, u=u2), l2)

class Test_air_core_inductor:

    def invoke(test_case):
        def wrapper(self):
            from electricpy.passive import air_core_inductor
            expected_result = test_case(self)
            computed_result = air_core_inductor(self.coil_diameter, self.coil_length, self.turn)            
            assert_almost_equal(computed_result, expected_result, decimal = 3)
        return wrapper

    @invoke
    def test_0(self):
        
        self.coil_diameter = 1e-3
        self.coil_length = 1e-3
        self.turn = 1000

        return 0.678640

    @invoke
    def test_1(self):

        self.coil_diameter = 1e-2
        self.coil_length = 1e-2
        self.turn = 251

        return 0.42755

class Test_visualization:

    def test_induction_motor_circle(self):
        from electricpy.visu import InductionMotorCircle

        open_circuit_test_data = {'V0': 400, 'I0': 9, 'W0': 1310}
        blocked_rotor_test_data = {'Vsc': 200, 'Isc': 50, 'Wsc': 7100}
        ratio = 1  # stator copper loss/ rotor copper loss
        output_power = 15000
        # InductionMotorCircle(open_circuit_test_data, blocked_rotor_test_data, output_power, torque_ration=ratio)
        #
        MotorCircle = InductionMotorCircle(open_circuit_test_data, blocked_rotor_test_data,
                            output_power, torque_ration=ratio, frequency=50, poles=4)

        assert_almost_equal(MotorCircle()['no_load_loss'], open_circuit_test_data['W0'])

    def test_power_circle(self):
        from electricpy.visu import receiving_end_power_circle
        data = {
            "A" : cmath.rect(0.895, math.radians(1.4)),
            "B" : cmath.rect(182.5, math.radians(78.6)),
            "Vr" : cmath.rect(215, 0),
            "Pr": 50,
            "power_factor": -0.9,
        }

        power_circle = receiving_end_power_circle(**data)

        assert_almost_equal(abs(power_circle()['Vs']), 224.909, decimal = 3)