import math
import cmath
from numpy.testing import assert_almost_equal

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