import pytest
import numpy as np

def assertAlmostEqual(a,b,abs=1e-6):

    if np.isnan(a) or np.isnan(b):
        raise ValueError("nan objects can not be compared")

    a = pytest.approx(a, abs=abs)
    b = pytest.approx(b, abs=abs)

    assert a==b

def test_phasor():
    from electricpy import phasor
    magnitude = 10
    # basic angles test case 0
    z1 = phasor(magnitude, 0)
    z2 = phasor(magnitude, 30)
    z3 = phasor(magnitude, 45)
    z4 = phasor(magnitude, 60)
    z5 = phasor(magnitude, 90)

    assertAlmostEqual(z1, complex(magnitude, 0))
    assertAlmostEqual(z2, complex(magnitude * np.sqrt(3) / 2, magnitude / 2))
    assertAlmostEqual(z3, complex(magnitude / np.sqrt(2), magnitude / np.sqrt(2)))
    assertAlmostEqual(z4, complex(magnitude / 2, magnitude * np.sqrt(3) / 2))
    assertAlmostEqual(z5, complex(0, magnitude))

    # z(theta) = z(theta+360) test case 1
    theta = np.random.randint(360)
    assertAlmostEqual(phasor(magnitude, theta), phasor(magnitude, theta + 360))

    # z(-theta)*z(theta) == abs(z)^2 test case 2.
    z0 = phasor(magnitude, theta)
    z1 = phasor(magnitude, -theta)
    assertAlmostEqual(z0 * z1, np.power(abs(z0), 2))

    # z(theta+180) = -1*Z(theta)
    z0 = phasor(magnitude, theta)
    z1 = phasor(magnitude, 180 + theta)
    assertAlmostEqual(z0, -1 * z1)

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

    assertAlmostEqual(zeq, zactual)

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

    assertAlmostEqual(S, P / PF)
    assertAlmostEqual(Q, S * np.sqrt(1 - PF ** 2))

    # Test case 1
    Q = 8
    P = 6

    _, _, S, PF = powerset(P=P, Q=Q)

    assertAlmostEqual(S, 10)
    assertAlmostEqual(PF, 0.6)

    #Test case 2 Failed

    # P = 0
    # Q = 0

    # _,_,S,PF = powerset(P = P, Q = Q)

    # assertAlmostEqual(S,0)
    # assertAlmostEqual(PF,0)

    #Test case 3 input validation is required
    # P = 10
    # Q = 10
    # PF = 0.6

    # _,_,S,_ = powerset(P = P, Q = Q, PF = PF)
    # assertAlmostEqual(S,10*np.sqrt(2))
    # assertAlmostEqual(PF,1/(np.sqrt(2)))

def test_voltdiv():
    from electricpy import voltdiv
    from electricpy import phasor

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

    assertAlmostEqual(Vout, Vout_actual)

def test_suspension_insulators():
    """
        Electric Power Systems by C.L Wadhwa Overhead Line Insulator example
    """

    from electricpy import suspension_insulators

    number_capacitors = 5

    capacitance_ration = 5

    Voltage = 66

    _, string_efficiency = suspension_insulators(number_capacitors,
                                                capacitance_ration, 
                                                Voltage)

    string_efficiency_actual = 54.16
    assertAlmostEqual(string_efficiency, string_efficiency_actual, abs=1e-2)

def test_propagation_constants():

    from electricpy import propagation_constants
    z = complex(0.5, 0.9)
    y = complex(0, 6e-6)
    params_dict = propagation_constants(z, y, 520)

    alpha_cal = 0.622 * (10 ** -3)
    beta_cal = 2.4 * (10 ** -3)

    assertAlmostEqual(params_dict['alpha'], alpha_cal, abs=1e-4)
    assertAlmostEqual(params_dict['beta'], beta_cal, abs=1e-4)

def test_funcrms():

    from electricpy import funcrms

    f = lambda x:np.sin(x)

    assertAlmostEqual(funcrms(f,np.pi), 1/np.sqrt(2))

def test_convolve():

    from electricpy import convolve
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

    assertAlmostEqual(result['duty_cycle'], 66.666666667, abs=1e-3)
    assertAlmostEqual(result['t_low'], 6.931*10**-6, abs=1e-3)
    assertAlmostEqual(result['t_high'], 1.386*10**-5, abs=1e-3)

    #test the other way around

def test_slew_rate():

    from electricpy import slew_rate

    SR = slew_rate(V=1, freq=1, find='SR')
    V = slew_rate(SR=1, freq=1, find='V')
    freq = slew_rate(V=1, SR=1, find='freq')

    assertAlmostEqual(np.pi*2, SR)
    assertAlmostEqual(1/(np.pi*2), V)
    assertAlmostEqual(1/(np.pi*2), freq)


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

    assertAlmostEqual(MotorCircle()['no_load_loss'], open_circuit_test_data['W0'])

def test_t_attenuator():
    Adb = 1
    Z0 = 1

    from electricpy import t_attenuator

    R1, R2 = t_attenuator(Adb, Z0)

    assertAlmostEqual(R1,  0.0575, abs=1e-3)
    assertAlmostEqual(R2, 8.6673, abs=1e-3)

def test_pi_attenuator():
    Adb = 1
    Z0 = 1

    from electricpy import pi_attenuator

    R1, R2 = pi_attenuator(Adb, Z0)
    assertAlmostEqual(R1, 17.39036, abs=1e-3)
    assertAlmostEqual(R2, 0.11538, abs=1e-3)

def test_inductor_voltdiv():

    from electricpy import inductive_voltdiv

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

    assert induction_machine_slip(Nr, freq=freq, poles=p) == 0.2
    assert induction_machine_slip(1500, freq=freq, poles=p) == 0
    assert induction_machine_slip(0, freq=freq, poles=p) == 1