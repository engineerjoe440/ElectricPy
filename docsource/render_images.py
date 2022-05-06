################################################################################
"""
Render images as needed using the various ElectricPy functions for docs.
"""
################################################################################

import electricpy as ep

def render_power_triangle():
    """Render the Power Triangle Function."""
    plt = ep.powertriangle(400, 200)
    plt.savefig("./static/PowerTriangle.png")



# Entrypoint
if __name__ == '__main__':
    # Add function calls here as new rendering functions are added
    render_power_triangle()

