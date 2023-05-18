import math
import cmath
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal

class Test_visualization:

    def test_induction_motor_circle(self):
        from electricpy.visu import InductionMotorCircle

        open_circuit_test_data = {'V0': 400, 'I0': 9, 'W0': 1310}
        blocked_rotor_test_data = {'Vsc': 200, 'Isc': 50, 'Wsc': 7100}
        ratio = 1  # stator copper loss/ rotor copper loss
        output_power = 15000
        MotorCircle = InductionMotorCircle(
            open_circuit_test_data,
            blocked_rotor_test_data,
            output_power,
            torque_ration=ratio,
            frequency=50,
            poles=4
        )

        assert_almost_equal(
            MotorCircle()['no_load_loss'],
            open_circuit_test_data['W0']
        )

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

    def test_rlc_frequency_response(self):
        # import RLC from electricpy.visu
        from electricpy.visu import SeriesRLC

        # test RLC
        rlc_obj1 = SeriesRLC(
            resistance=5, inductance=0.4, capacitance=25.3e-6, frequency=50
        )

        # gh1 = rlc_obj1.graph(lower_frequency_cut=0.1, upper_frequency_cut=100, samples=1000)

        rlc_obj2 = SeriesRLC(
            resistance=10, inductance=0.5, capacitance=25.3e-6, frequency=50
        )

        # gh2 = rlc_obj2.graph(lower_frequency_cut=0.1, upper_frequency_cut=100, samples=1000)

        # plt.gca().add_artist(gh1.legend(rlc_obj1.legend(), title=f"(R, L, C) => (5, 0.4, 25.3e-6)", loc='upper right'))
        # plt.gca().add_artist(gh2.legend(rlc_obj2.legend(), title=f"(R, L, C) => (10, 0.5 25.3e-6)", loc='upper left'))

        # plt.show()


