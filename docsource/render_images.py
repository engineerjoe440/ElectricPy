################################################################################
"""Render images as needed using the various ElectricPy functions for docs."""
################################################################################

import electricpy as ep
from electricpy import visu

def render_power_triangle():
    """Render the Power Triangle Function."""
    plt = ep.powertriangle(400, 200)
    plt.savefig("docsource/static/PowerTriangle.png")

def render_phasor_plot():
    """Render the Phasor Plot Function."""
    phasors = ep.phasor.phasorlist([
        [67,0],
        [45,-120],
        [52,120]
    ])
    plt = ep.phasor.phasorplot(phasor=phasors, colors=["red", "green", "blue"])
    plt.savefig("docsource/static/PhasorPlot.png")

def render_motor_circle():
    """Render the Induction Motor Circle and Draw."""
    open_circuit_test_data = {'V0': 400, 'I0': 9, 'W0': 1310}
    blocked_rotor_test_data = {'Vsc': 200, 'Isc': 50, 'Wsc': 7100}
    ratio = 1  # stator copper loss/ rotor copper loss
    output_power = 15000
    plt = visu.InductionMotorCircle(
        no_load_data=open_circuit_test_data,
        blocked_rotor_data=blocked_rotor_test_data,
        output_power=output_power,
        torque_ration=ratio,
        frequency=50, 
        poles=4
    ).plot()
    plt.savefig("docsource/static/InductionMotorCircleExample.png")

def render_receiving_end_power_circle():
    """Render the Receiving End Power Circle Plot."""
    import math, cmath
    plt = visu.receiving_end_power_circle(
        A=cmath.rect(0.895, math.radians(1.4)),
        B=cmath.rect(182.5, math.radians(78.6)),
        Vr=cmath.rect(215, 0),
        Pr=50,
        power_factor=-0.9
    ).plot()
    plt.savefig("docsource/static/ReceivingEndPowerCircleExample.png")


# Entrypoint
if __name__ == '__main__':
    # Add function calls here as new rendering functions are added
    render_power_triangle()
    render_phasor_plot()
    render_motor_circle()

# END
