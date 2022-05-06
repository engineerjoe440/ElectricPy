################################################################################
"""
Render images as needed using the various ElectricPy functions for docs.
"""
################################################################################

import electricpy as ep

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
    plt = ep.phasor.phasorplot(phasor=phasors)
    plt.savefig("docsource/static/PhasorPlot.png")



# Entrypoint
if __name__ == '__main__':
    # Add function calls here as new rendering functions are added
    render_power_triangle()
    render_phasor_plot()

# END
