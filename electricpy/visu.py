################################################################################
"""
Visualizations Specifically for Electrical Engineering.

Filled with plotting functions and visualization tools for electrical engineers,
this module is designed to assist engineers visualize their designs.
"""
################################################################################

import cmath as _c

import numpy as _np
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt

from electricpy import powerset, geometry
from electricpy.geometry import Point
from electricpy.geometry.circle import Circle


# Define Power Triangle Function
def powertriangle(
    P=None, Q=None, S=None, PF=None, color="red", text="Power Triangle", printval=False
):
    """
    Power Triangle Plotting Function.

    This function is designed to draw a power triangle given
    values for the complex power system.

    .. image:: /static/PowerTriangle.png

    Parameters
    ----------
    P:          float
                Real Power, unitless, default=None
    Q:          float
                Reactive Power, unitless, default=None
    S:          float
                Apparent Power, unitless, default=None
    PF:         float
                Power Factor, unitless, provided as a decimal value, lagging is
                positive, leading is negative; default=None
    color:      string, optional
                The color of the power triangle lines, default="red"
    text:       string, optional
                The title of the power triangle plot, default="Power Triangle"
    printval:   bool, optional
                Control argument to allow the numeric values to be printed on
                the plot, default="False"

    Returns
    -------
    matplotlib.pyplot:  Plotting object to be used for additional configuration
                        or plotting.
    """
    # Calculate all values if not all are provided
    if P is None or Q is None or S is None or PF is None:
        P, Q, S, PF = powerset(P, Q, S, PF)

    # Generate Lines
    p_line_x = [0, P]
    p_line_y = [0, 0]
    q_line_x = [P, P]
    q_line_y = [0, Q]
    s_line_x = [0, P]
    s_line_y = [0, Q]

    # Plot Power Triangle
    _plt.figure(1)
    _plt.title(text)
    _plt.plot(p_line_x, p_line_y, color=color)
    _plt.plot(q_line_x, q_line_y, color=color)
    _plt.plot(s_line_x, s_line_y, color=color)
    _plt.xlabel("Real Power (W)")
    _plt.ylabel("Reactive Power (VAR)")
    maximum = max(abs(P), abs(Q))

    if P > 0:
        _plt.xlim(0, maximum * 1.1)
        x = maximum
    else:
        _plt.xlim(-maximum * 1.1, 0)
        x = -maximum
    if Q > 0:
        _plt.ylim(0, maximum * 1.1)
        y = maximum
    else:
        _plt.ylim(-maximum * 1.1, 0)
        y = -maximum
    if PF > 0:
        power_factor_text = "Lagging"
    else:
        power_factor_text = "Leading"
    # Print all values if asked to
    if printval:
        _plt.text(
            x / 20,
            y * 4 / 5,
            (
                f"P:   {P} W\n"
                f"Q:   {Q} VAR\n"
                f"S:   {S} VA\n"
                f"PF:  {abs(PF)} {power_factor_text}\n"
                f"ΘPF: {_np.degrees(_np.arccos(PF))}° {power_factor_text}"
            ),
            color=color,
        )
    return _plt


# Define Convolution Bar-Graph Function:
def convbar(h, x, outline=True):
    """
    Convolution Bar-Graph Plotter Function.

    Generates plots of each of two input arrays as bar-graphs, then
    generates a convolved bar-graph of the two inputs to demonstrate
    and illustrate convolution, typically for an educational purpose.

    Examples
    --------
    >>> import numpy as np
    >>> import electricpy.visu as visu
    >>> h = np.array([0, 1, 1, 1, 0])
    >>> x = np.array([0, 1, 1, 1, 0])
    >>> visu.convbar(h, x)

    .. image:: /static/convbar-example.png

    Parameters
    ----------
    h:      numpy.ndarray
            Impulse Response - Given as Array (Prefferably Numpy Array)
    x:      numpy.ndarray
            Input Function - Given as Array (Prefferably Numpy Array)
    """
    # The impulse response
    M = len(h)
    t = _np.arange(M)
    # Plot
    _plt.subplot(121)
    if outline:
        _plt.plot(t, h, color="red")
    _plt.bar(t, h, color="black")
    _plt.xticks([0, 5, 9])
    _plt.ylabel("h")
    _plt.title("Impulse Response")
    _plt.grid()

    # The input function
    N = len(x)
    s = _np.arange(N)
    # Plot
    _plt.subplot(122)
    if outline:
        _plt.plot(s, x, color="red")
    _plt.bar(s, x, color="black")
    _plt.xticks([0, 10, 19])
    _plt.title("Input Function")
    _plt.grid()
    _plt.ylabel("x")

    # The output
    L = M + N - 1
    w = _np.arange(L)
    _plt.figure(3)
    y = _np.convolve(h, x)
    if outline:
        _plt.plot(w, y, color="red")
    _plt.bar(w, y, color="black")
    _plt.ylabel("y")
    _plt.grid()
    _plt.title("Convolved Output")
    return _plt


# Define Phasor Plot Generator
def phasorplot(
    phasors,
    title="Phasor Diagram",
    legend=False,
    bg=None,
    colors=None,
    radius=None,
    linewidth=None,
    size=None,
    label=False,
    labels=False,
    tolerance=None,
):
    """
    Phasor Plotting Function.

    This function is designed to plot a phasor-diagram with angles in degrees
    for up to 12 phasor sets (more may be used if additional colors are set).
    Phasors must be passed as a complex number set, (e.g.
    [ m+ja, m+ja, m+ja, ... , m+ja ] ).

    Examples
    --------
    >>> import numpy as np
    >>> from electricpy import phasors
    >>> from electricpy import visu
    >>> voltages = np.array([
    ...     [67,0],
    ...     [45,-120],
    ...     [52,120]
    ... ])
    >>> phasors = phasors.phasorlist(voltages)
    >>> plt = visu.phasorplot(phasors, colors=["red", "green", "blue"])
    >>> plt.show()

    .. image:: /static/PhasorPlot.png

    Parameters
    ----------
    phasors:    list of complex
                The set of phasors to be plotted.
    title:      string, optional
                The Plot Title, default="Phasor Diagram"
    legend:     bool, optional
                Control argument to enable displaying the legend, must be passed
                as an array or list of strings. `label` and `labels` are mimic-
                arguments and will perform similar operation, default=False
    bg:         string, optional
                Background-Color control, default="#d5de9c"
    radius:     float, optional
                The diagram radius, unless specified, automatically scales
    colors:     list of str, optional
                List of hexidecimal color strings denoting the line colors to
                use.
    size:       float, optional
                Control argument for figure size. default=None
    linewidth:  float, optional
                Control argument to declare the line thickness. default=None
    tolerance:  float, optional
                Minimum magnitude to plot, anything less than tolerance will be
                plotted as a single point at the origin, by default, the
                tolerance is scaled to be 1/25-th the maximum radius. To disable
                the tolerance, simply provide either False or -1.

    Returns
    -------
    matplotlib.pyplot:  Plotting object to be used for additional configuration
                        or plotting.
    """
    # Load Complex Values if Necessary
    try:
        len(phasors)
    except TypeError:
        phasors = [phasors]
    # Manage Colors
    if colors is None:
        colors = [
            "#FF0000",
            "#800000",
            "#FFFF00",
            "#808000",
            "#00ff00",
            "#008000",
            "#00ffff",
            "#008080",
            "#0000ff",
            "#000080",
            "#ff00ff",
            "#800080",
        ]
    # Scale Radius
    if radius is None:
        radius = _np.abs(phasors).max()
    # Set Tolerance
    if tolerance is None:
        tolerance = radius / 25
    elif tolerance is False:
        tolerance = -1
    # Set Background Color
    if bg is None:
        bg = "#FFFFFF"
    # Load labels if handled in other argument
    if label:
        legend = label
    if labels:
        legend = labels
    # Check for more phasors than colors
    if len(phasors) > len(colors):
        raise ValueError("ERROR: Too many phasors provided. Specify more line colors.")

    if size is None:
        # Force square figure and square axes
        width, height = _matplotlib.rcParams["figure.figsize"]
        size = min(width, height)
    # Make a square figure
    fig = _plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, facecolor=bg)
    _plt.grid(True)

    # Plot the diagram
    _plt.title(title + "\n")
    arrows = []
    for i, phasor in enumerate(phasors):
        mag, ang_r = _c.polar(phasor)
        # Plot with labels
        if legend:
            if mag > tolerance:
                arrows.append(
                    _plt.arrow(
                        0,
                        0,
                        ang_r,
                        mag,
                        color=colors[i],
                        label=legend[i],
                        linewidth=linewidth,
                    )
                )
            else:
                arrows.append(
                    _plt.plot(
                        0,
                        0,
                        "o",
                        markersize=linewidth * 3,
                        label=legend[i],
                        color=colors[i],
                    )
                )
        # Plot without labels
        else:
            _plt.arrow(0, 0, ang_r, mag, color=colors[i], linewidth=linewidth)
    if legend:
        _plt.legend(arrows, legend)
    # Set Minimum and Maximum Radius Terms
    ax.set_rmax(radius)
    ax.set_rmin(0)
    return _plt


class InductionMotorCircle:
    """
    Plot Induction Motor Circle Diagram.

    This class is designed to plot induction motor circle diagram
    and plot circle diagram to obtain various parameters of induction motor.

    Examples
    --------
    >>> from electricpy.visu import InductionMotorCircle
    >>> open_circuit_test_data = {'V0': 400, 'I0': 9, 'W0': 1310}
    >>> blocked_rotor_test_data = {'Vsc': 200, 'Isc': 50, 'Wsc': 7100}
    >>> ratio = 1  # stator copper loss/ rotor copper loss
    >>> output_power = 15000
    >>> InductionMotorCircle(
    ...     no_load_data=open_circuit_test_data,
    ...     blocked_rotor_data=blocked_rotor_test_data,
    ...     output_power=output_power,
    ...     torque_ration=ratio,
    ...     frequency=50,
    ...     poles=4
    ... )

    .. image:: /static/InductionMotorCircleExample.png

    Parameters
    ----------
    no_load_data:       dict {'V0', 'I0', 'W0'}
                        V0: no load test voltage
                        I0: no load current in rotor
                        W0: No load power(in Watts)
    blocked_rotor_data: dict {'Vsc','Isc','Wsc'}
                        Vsc: blocked rotor terminal voltage
                        Isc: blocked rotor current in rotor
                        Wsc: Power consumed in blocked rotor test
    output_power:       int
                        Desired power output from the induction motor
    torque_ration:      float
                        Ration between rotor resistance to stator resistance
                        (i.e., R2/R1)
    frequency:          int
                        AC supply frequency
    poles:              int
                        Pole count of induction Motor
    """

    def __init__(
        self,
        no_load_data,
        blocked_rotor_data,
        output_power,
        torque_ration=1,
        frequency=50,
        poles=4,
    ):
        """Primary Entrypoint."""
        self.no_load_data = no_load_data
        self.blocked_rotor_data = blocked_rotor_data
        self.frequency = frequency
        self.operating_power = output_power
        self.torque_ratio = torque_ration
        self.poles = poles
        self.sync_speed = 120 * frequency / poles  # rpm

        v0 = no_load_data["V0"]
        i0 = no_load_data["I0"]
        w0 = no_load_data["W0"]

        self.no_load_pf = w0 / (_np.sqrt(3) * v0 * i0)
        theta0 = _np.arccos(self.no_load_pf)

        # get short circuit power factor and Current at slip=1
        vsc = blocked_rotor_data["Vsc"]
        isc = blocked_rotor_data["Isc"]
        wsc = blocked_rotor_data["Wsc"]

        self.blocked_rotor_pf = wsc / (_np.sqrt(3) * vsc * isc)
        theta_sc = _np.arccos(self.blocked_rotor_pf)

        # because V is on Y axis
        theta0 = _np.pi / 2 - theta0
        theta_sc = _np.pi / 2 - theta_sc

        # isc is the current at reduced voltage
        # calculate current at rated voltage
        isc = v0 * isc / vsc
        self.no_load_line = [[0, i0 * _np.cos(theta0)], [0, i0 * _np.sin(theta0)]]
        self.full_load_line = [
            [0, isc * _np.cos(theta_sc)],
            [0, isc * _np.sin(theta_sc)],
        ]

        # secondary current line
        self.secondary_current_line = [
            [i0 * _np.cos(theta0), isc * _np.cos(theta_sc)],
            [i0 * _np.sin(theta0), isc * _np.sin(theta_sc)],
        ]

        [[x1, x2], [y1, y2]] = self.secondary_current_line
        self.theta = _np.arctan((y2 - y1) / (x2 - x1))

        # get the induction motor circle
        self.power_scale = w0 / (i0 * _np.sin(theta0))
        self.center, self.radius = self.compute_circle_params()
        [self.center_x, self.center_y] = self.center
        self.p_max = self.radius * _np.cos(self.theta) - (
            self.radius - self.radius * _np.sin(self.theta)
        ) * _np.tan(self.theta)
        self.torque_line, self.torque_point = self.get_torque_line()
        self.torque_max, self.torque_max_x, self.torque_max_y = self.get_torque_max()
        # Take low slip point
        _, [self.power_x, self.power_y] = self.get_output_power()
        self.data = self.compute_efficiency()

    def __call__(self):
        # noqa: D102
        __doc__ = self.__doc__
        return self.data

    def plot(self):
        """Plot the Induction Motor Circle Diagram."""
        [circle_x, circle_y] = InductionMotorCircle.__get_circle(
            self.center, self.radius, semi=True
        )
        _plt.plot(circle_x, circle_y)

        InductionMotorCircle.__plot_line(self.no_load_line)
        InductionMotorCircle.__plot_line(self.secondary_current_line)
        InductionMotorCircle.__plot_line(self.full_load_line, ls="-.")
        InductionMotorCircle.__plot_line(self.torque_line, ls="-.")

        # Full load output
        _plt.plot(
            [self.secondary_current_line[0][1], self.secondary_current_line[0][1]],
            [self.secondary_current_line[1][1], self.center_y],
        )
        # Diameter of the circle
        _plt.plot(
            [self.center_x - self.radius, self.center_x + self.radius],
            [self.center_y, self.center_y],
            ls="-.",
        )
        # Max torque line
        _plt.plot(
            [self.center_x, self.torque_max_x],
            [self.center_y, self.torque_max_y],
            ls="-.",
        )
        # Max Output Power line
        _plt.plot(
            [self.center_x, self.center_x - self.radius * _np.sin(self.theta)],
            [self.center_y, self.center_y + self.radius * _np.cos(self.theta)],
            ls="-.",
        )
        # Operating Point
        _plt.plot([0, self.power_x], [0, self.power_y], c="black")

        _plt.scatter(
            self.power_x, self.power_y, marker="X", c="red", label="_nolegend_"
        )
        # mark the center of the circle
        _plt.scatter(
            self.center_x, self.center_y, marker="*", c="blue", label="_nolegend_"
        )
        _plt.scatter(
            self.center_x - self.radius * _np.sin(self.theta),
            self.center_y + self.radius * _np.cos(self.theta),
            linewidths=3,
            c="black",
            marker="*",
            label="_nolegend_",
        )
        _plt.scatter(
            self.torque_max_x,
            self.torque_max_y,
            linewidths=3,
            c="black",
            marker="*",
            label="_nolegend_",
        )

        _plt.title("Induction Motor Circle Diagram")
        _plt.grid()
        _plt.legend(
            [
                "I2 locus",
                "No Load Current",
                "Output Line",
                "Blocked Rotor Current",
                "Torque line",
                "Full Load Losses",
                "Diameter",
                "Maximum Torque",
                "Maximum Output Power",
                f"Operating Power {self.operating_power}",
            ]
        )
        return _plt

    def compute_efficiency(self):
        """Compute the output efficiency of induction motor."""
        [[_, no_load_x], [_, no_load_y]] = self.no_load_line
        no_load_losses = no_load_y * self.power_scale

        compute_slope = InductionMotorCircle.compute_slope

        torque_slope = compute_slope(self.torque_line)
        stator_cu_loss = (self.power_x - no_load_x) * torque_slope * self.power_scale

        rotor_current_slope = compute_slope(self.secondary_current_line)
        total_cu_loss = (
            (self.power_x - no_load_x) * rotor_current_slope * self.power_scale
        )

        rotor_cu_loss = total_cu_loss - stator_cu_loss

        rotor_output = self.power_y * self.power_scale - (
            rotor_cu_loss + stator_cu_loss + no_load_losses
        )

        slip = rotor_cu_loss / rotor_output

        self.rotor_speed = self.sync_speed * (1 - slip)

        data = {
            "no_load_loss": no_load_losses,
            "rotor_copper_loss": rotor_cu_loss,
            "stator_copper_loss": stator_cu_loss,
            "rotor_output": rotor_output,
            "slip": slip,
            "stator_rmf_speed (RPM)": self.sync_speed,
            "rotor_speed (RMP)": self.rotor_speed,
            "power_factor": (
                self.power_y / _np.sqrt(self.power_x**2 + self.power_y**2)
            ),
            "efficiency": f"{rotor_output * 100 / (self.power_y * self.power_scale)} %",
        }
        return data

    @staticmethod
    def __get_circle(center, radius, semi=False):
        """
        Determine parametric equation of circle.

        Parameters
        ----------
        center: list[float, float] [x0, y0]
        radius: float

        Returns
        -------
        (x, y): tuple
                parametric equation of circle
                (x = x0 + r*cos(theta) ; y = y0 + r*sin(theta))
        """
        [x0, y0] = center

        if semi:
            theta = _np.arange(0, _np.pi, 1e-4)
        else:
            theta = _np.arange(0, _np.pi * 2, 1e-4)

        x = x0 + radius * _np.cos(theta)
        y = y0 + radius * _np.sin(theta)
        return x, y

    @staticmethod
    def __plot_line(line, mark_start=True, mark_end=True, ls="-", marker=None):
        """Supporting function to plot a line."""
        [x, y] = line
        [x1, x2] = x
        [y1, y2] = y
        _plt.plot(x, y, ls=ls)
        if mark_start:
            _plt.scatter(x1, y1, marker=marker, label="_nolegend_")
        if mark_end:
            _plt.scatter(x2, y2, marker=marker, label="_nolegend_")

    def compute_circle_params(self):
        """Compute the parameters of induction motor circle."""
        [[x1, x2], [y1, y2]] = self.secondary_current_line
        theta = _np.arctan((y2 - y1) / (x2 - x1))
        length = _np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        radius = length / (2 * _np.cos(theta))
        center = [radius + x1, y1]

        return center, radius

    def get_torque_line(self):
        """Obtain the torque line of the induction motor."""
        [[x1, x2], [y1, y2]] = self.secondary_current_line
        y = (self.torque_ratio * y2 + y1) / (self.torque_ratio + 1)
        torque_point = [x2, y]
        torque_line = [[x1, x2], [y1, y]]
        return torque_line, torque_point

    def get_torque_max(self):
        """Compute max torque for given Induction Motor parameters."""
        [x, y] = self.torque_line
        [x1, x2] = x
        [y1, y2] = y
        alpha = _np.arctan((y2 - y1) / (x2 - x1))
        torque_max = self.radius * _np.cos(alpha) - (
            self.radius - self.radius * _np.sin(alpha)
        ) * _np.tan(alpha)
        torque_max_x = self.center_x - self.radius * _np.sin(alpha)
        torque_max_y = self.center_y + self.radius * _np.cos(alpha)
        return torque_max, torque_max_x, torque_max_y

    @staticmethod
    def compute_slope(line):
        """
        Compute slope of the line.

        Parameters
        ----------
        line: list[float, float]

        Returns
        -------
        slope: float
        """
        [[x1, x2], [y1, y2]] = line
        return (y2 - y1) / (x2 - x1)

    def get_output_power(self):
        """
        Determine induction motor circle desired output power point.

        Obtain the point on the induction motor circle diagram which
        corresponds to the desired output power
        """
        [[x1, x2], [y1, y2]] = self.secondary_current_line
        alpha = _np.arctan((y2 - y1) / (x2 - x1))
        [center_x, center_y] = self.center

        [[_, no_load_x], [_, _]] = self.no_load_line
        beta = _np.arcsin(
            (
                self.operating_power / self.power_scale
                + (center_x - no_load_x) * _np.tan(alpha)
            )
            * _np.cos(alpha)
            / self.radius
        )
        beta_0 = alpha + beta
        beta_1 = -alpha + beta
        # high slip
        p_x_1 = center_x + self.radius * _np.cos(beta_0)
        p_y_1 = center_y + self.radius * _np.sin(beta_0)
        # low slip
        p_x_2 = center_x - self.radius * _np.cos(beta_1)
        p_y_2 = center_y + self.radius * _np.sin(beta_1)
        return [p_x_1, p_y_1], [p_x_2, p_y_2]


class PowerCircle:
    r"""
    Plot Power Circle Diagram of Transmission System.

    This class is designed to plot the power circle diagram of a transmission
    system both sending and receiving ends.

    Examples
    --------
    >>> import math, cmath
    >>> from electricpy import visu
    >>> visu.PowerCircle(
    ...     power_circle_type="receiving",
    ...     A=cmath.rect(0.895, math.radians(1.4)),
    ...     B=cmath.rect(182.5, math.radians(78.6)),
    ...     Vr=cmath.rect(215, 0),
    ...     Pr=50,
    ...     power_factor=-0.9
    ... )

    .. image:: /static/ReceivingPowerCircleExample.png

    Parameters
    ----------
    power_circle_type:  ["sending", "receiving"]
                        Type of power circle diagram to plot.
    Vr:                 complex
                        Transmission Line Receiving End Voltage (phasor complex
                        value)
    Vs:                 complex
                        Transmission Line Sending End Voltage (phasor complex
                        value)
    power_factor:       float
                        Power Factor of the transmission system, default = None
    Pr:                 float
                        Receiving End Real Power, default = None
    Qr:                 float
                        Receiving End Reactive Power, default = None
    Sr:                 complex
                        Receiving End Total Complex Power, default = None
    Ps:                 float
                        Sending End Real Power, default = None
    Qs:                 float
                        Sending End Reactive Power, default = None
    Ss:                 complex
                        Sending End Total Complex Power, default = None
    A:                  float
                        Transmission System ABCD Parameters, A, default = None
    B:                  float
                        Transmission System ABCD Parameters, B, default = None
    C:                  float
                        Transmission System ABCD Parameters, C, default = None
    D:                  float
                        Transmission System ABCD Parameters, D, default = None
    """

    def __init__(
        self,
        power_circle_type: str,
        power_factor: float = None,
        Vr: complex = None,
        Vs: complex = None,
        Pr: float = None,
        Qr: float = None,
        Sr: complex = None,
        Ps: float = None,
        Qs: float = None,
        Ss: complex = None,
        A: complex = None,
        B: complex = None,
        C: complex = None,
        D: complex = None,
    ) -> None:
        r"""Initialize the class."""
        if C is not None:
            assert (
                abs(A * D - B * C - 1) < 1e-6
            ), "ABCD Matrix is not a valid ABCD Matrix"

        if power_circle_type.lower() == "receiving":

            if A is not None and B is not None and Vr is not None:
                (
                    self.radius,
                    self.center,
                    self.operating_point,
                ) = PowerCircle._build_circle(
                    A, B, "receiving_end", Vr, Pr, Qr, Sr, power_factor, Vs
                )
            else:
                raise ValueError("Not enough attributes to build circle")

        elif power_circle_type.lower() == "sending":

            if B is not None and D is not None and Vs is not None:
                (
                    self.radius,
                    self.center,
                    self.operating_point,
                ) = PowerCircle._build_circle(
                    D, B, "sending_end", Vs, Ps, Qs, Ss, power_factor, Vr
                )
            else:
                raise ValueError("Not enough attributes to build power circle")

        else:
            raise ValueError("Invalid power circle type")

        self.circle = Circle(self.center, self.radius)
        self.parameters = locals()

    @staticmethod
    def _build_circle(
        a1, a2, circle_type, V, P=None, Q=None, S=None, power_factor=None, V_ref=None
    ):

        k = (abs(V) ** 2) * abs(a1) / abs(a2)
        alpha = _c.phase(a1)
        beta = _c.phase(a2)

        if circle_type == "receiving_end":
            center = Point(-k * _c.cos(alpha - beta), -k * _c.sin(alpha - beta))

        elif circle_type == "sending_end":
            center = Point(k * _c.cos(alpha - beta), -k * _c.sin(alpha - beta))

        if V_ref is not None and P is not None and Q is not None:
            radius = abs(V) * abs(V_ref) / (abs(a2))
            operation_point = Point(P, Q)

        elif V_ref is not None and S is not None:
            radius = abs(V) * abs(V_ref) / (abs(a2))
            operation_point = Point(S.real, S.imag)

        elif P is not None and Q is not None:
            radius = geometry.distance(center, Point(P, Q))
            operation_point = Point(P, Q)

        elif S is not None:
            radius = geometry.distance(center, Point(S.real, S.imag))
            operation_point = Point(S.real, S.imag)

        elif P is not None and power_factor is not None:

            Q = P * _c.sqrt(1 / power_factor**2 - 1).real

            if power_factor < 0:
                Q = -1 * Q

            radius = geometry.distance(center, Point(P, Q))
            operation_point = Point(P, Q)

        elif Q is not None and power_factor is not None:
            P = Q / _c.sqrt(1 / power_factor**2 - 1).real
            radius = geometry.distance(center, Point(P, Q))
            operation_point = Point(P, Q)

        else:
            raise AttributeError("Not enough attributes found to perform calculation")

        return radius, center, operation_point

    def _cal_parameters(self, type1, type2):

        if self.parameters["V" + type2] is None:
            self.parameters["V" + type2] = (
                abs(self.parameters["B"]) * self.radius / self.parameters["V" + type1]
            )

        if self.parameters["P" + type1] is None:
            self.parameters["P" + type1] = self.operating_point.x

        if self.parameters["Q" + type1] is None:
            self.parameters["Q" + type1] = self.operating_point.y

        if self.parameters["S" + type1] == None:
            self.parameters["S" + type1] = (
                self.operating_point.x + 1j * self.operating_point.y
            )

        if self.parameters["power_factor"] is None:
            self.parameters["power_factor"] = (
                self.operating_point.y / self.operating_point.x
            )

        if type1 == "r" and type2 == "s":
            self.parameters["Vs"] = (
                self.parameters["B"] * self.parameters["Sr"]
                + self.parameters["A"] * abs(self.parameters["Vr"]) ** 2
            )
            self.parameters["Vs"] = (
                self.parameters["Vs"] / self.parameters["Vr"].conjugate()
            )

        elif type1 == "s" and type2 == "r":
            self.parameters["Vr"] = (
                -self.parameters["B"] * self.parameters["Ss"]
                + self.parameters["D"] * abs(self.parameters["Vs"]) ** 2
            )
            self.parameters["Vr"] = (
                self.parameters["Vr"] / self.parameters["Vs"].conjugate()
            )

    def print_data(self):
        r"""Print the data of the circle."""
        if self.operating_point is None:
            return self.center, self.radius

        if self.parameters["power_circle_type"] == "receiving":
            self._cal_parameters("r", "s")

        if self.parameters["power_circle_type"] == "sending":
            self._cal_parameters("s", "r")

        for key, value in self.parameters.items():
            print(key, " => ", value)

    def __call__(self) -> dict:
        r"""Return the data of the circle."""
        if self.parameters["power_circle_type"] == "receiving":
            self._cal_parameters("r", "s")

        if self.parameters["power_circle_type"] == "sending":
            self._cal_parameters("s", "r")

        return self.parameters

    def plot(self):
        r"""Plot the circle."""
        circle_x = []
        circle_y = []

        for data in self.circle.parametric_equation(theta_resolution=1e-5):
            [x, y] = data
            circle_x.append(x)
            circle_y.append(y)

        c_x = self.center.x
        c_y = self.center.y

        op_x = self.operating_point.x
        op_y = self.operating_point.y

        # plot Circle and Diameter
        _plt.plot(circle_x, circle_y)
        _plt.plot([c_x - self.radius, c_x + self.radius], [c_y, c_y], "g--")
        _plt.plot([c_x, c_x], [c_y - self.radius, c_y + self.radius], "g--")

        _plt.plot([c_x, op_x], [c_y, op_y], "y*-.")
        _plt.plot([op_x, op_x], [op_y, c_y], "b*-.")
        _plt.scatter(op_x, op_y, marker="*", color="r")
        _plt.title(f"{self.parameters['power_circle_type'].capitalize()} Power Circle")
        _plt.xlabel("Active Power")
        _plt.ylabel("Reactive Power")
        _plt.grid()
        return _plt


def receiving_end_power_circle(
    Vr: complex = None,
    A: complex = None,
    B: complex = None,
    Pr: float = None,
    Qr: float = None,
    Sr: complex = None,
    power_factor: float = None,
    Vs: complex = None,
) -> PowerCircle:
    """
    Construct Receiving End Power Circle.

    Examples
    --------
    >>> import math, cmath
    >>> from electricpy import visu
    >>> visu.receiving_end_power_circle(
    ...     A=cmath.rect(0.895, math.radians(1.4)),
    ...     B=cmath.rect(182.5, math.radians(78.6)),
    ...     Vr=cmath.rect(215, 0),
    ...     Pr=50,
    ...     power_factor=-0.9
    ... )

    .. image:: /static/ReceivingEndPowerCircleExample.png

    Parameters
    ----------
    Vr:             complex
                    Receiving End Voltage, default = None.
    A:              complex
                    Transmission System ABCD Parameters, A, default = None.
    B:              complex,
                    Transmission System ABCD Parameters, B, default = None.
    Pr:             float, optional
                    Receiving End Real Power, default = None
    Qr:             float, optional
                    Receiving End Reactive Power, default = None
    Sr:             complex, optional
                    Receiving End Apparent Power, default = None
    power_factor:   float, optional
                    System End Power Factor, default = None
    Vs:             complex, optional
                    Sending End Voltage, default = None

    Returns
    -------
    Receiving End Power Circle: PowerCircle
    """
    try:
        assert Vr is not None and A is not None and B is not None
    except AssertionError:
        raise ValueError(
            "Not enough attributes to build Receiving end power circle at least"
            " provide `Vr`, `A`, `B`"
        )

    if not (
        (
            (Pr is not None and Qr is not None)
            or (Sr is not None and power_factor is not None)
        )
        or (
            (Pr is not None and power_factor is not None)
            or (Qr is not None and power_factor is not None)
        )
    ):
        raise ValueError(
            "Not enough attributes for marking an operating point on Receiving "
            "End Power Circle"
        )

    return PowerCircle(
        "receiving",
        **{
            "Vr": Vr,
            "A": A,
            "B": B,
            "Pr": Pr,
            "Qr": Qr,
            "Sr": Sr,
            "Vs": Vs,
            "power_factor": power_factor,
        },
    )


def sending_end_power_circle(
    Vs: complex = None,
    B: complex = None,
    D: complex = None,
    Ps: float = None,
    Qs: float = None,
    Ss: complex = None,
    power_factor: float = None,
    Vr: complex = None,
) -> PowerCircle:
    """
    Construct Receiving End Power Circle.

    Parameters
    ----------
    Vs:             complex
                    Sending End Voltage
    B:              complex
                    Transmission System ABCD Parameters, A
    D:              complex
                    Transmission System ABCD Parameters, B
    Ps:             float, optional
                    Sending End Real Power, default = None
    Qs:             float, optional
                    Sending End Reactive Power, default = None
    Ss:             complex, optional
                    Sending End Apparent Power, default = None
    power_factor:   float, optional
                    System Power Factor, default = None
    Vr:             complex, optional
                    Receiving End Voltage, default = None

    Returns
    -------
    Sending End Power Circle: PowerCircle
    """
    if not (Vs is not None and B is not None and D is not None):
        raise ValueError(
            "Not enough attributes to build Sending end power circle at least "
            "provide `Vs`, `B`, `D`"
        )

    if not (
        (
            (Ps is not None and Qs is not None)
            or (Ss is not None and power_factor is not None)
        )
        or (
            (Ps is not None and power_factor is not None)
            or (Qs is not None and power_factor is not None)
        )
    ):
        raise ValueError(
            "Not enough attributes for marking an operating point on Sending "
            "End Power Circle"
        )

    return PowerCircle(
        "sending",
        **{
            "Vr": Vr,
            "B": B,
            "D": D,
            "Ps": Ps,
            "Qs": Qs,
            "Ss": Ss,
            "Vs": Vs,
            "power_factor": power_factor,
        },
    )


class SeriesRLC():
    r"""
    Frequency Response for an RLC (Resistive, Inductive, Capacitive) Load.

    Generate unique information about an RLC circuit. Using this class, you may
    generate a variety of useful statistics including resonance frequency,
    bandwidth, lower and upper cuttoff frequencies, and more. Each of the
    specific parameters are evaluated as follows.

    **Resonance Frequency:**

    .. math:: \text{resonance_frequency} = \frac{1}{\sqrt{L * C} \cdot 2 \pi}

    
    **Bandwidth:**

    .. math:: \text{bandwidth} = \frac{R}{L \cdot 2 \pi}

    
    **Quality Factor:**

    .. math:: \text{quality_factor} = 2\pi \frac{\text{freq}}{R}


    Given the characteristics listed below, and the Python code described in the
    associated example, the following plot will be generated.
    
    * Resistance: 5 ohms
    * Inductance: 0.4 henreys
    * Capacitance: 25.3e-6 farads
    * Frequency: 50 Hz

    .. image:: /static/series-rlc-r5-l0.4.png


    * Resistance: 10 ohms
    * Inductance: 0.5 henreys
    * Capacitance: 25.3e-6 farads
    * Frequency: 50 Hz

    .. image:: /static/series-rlc-r10-l0.5.png

    
    Examples
    --------
    >>> from electricpy.visu import SeriesRLC
    >>> rlc_component = SeriesRLC(
    ...     resistance=5, inductance=0.4, capacitance=25.3e-6, frequency=50
    ... )
    >>> rlc_component.resonance_frequency
    50.029927713857425
    >>> rlc_component.bandwidth
    1.9894367886486917
    >>> plot_1 = rlc_component.graph(
    ...     lower_frequency_cut=0.1, upper_frequency_cut=100, samples=1000
    ... )
    >>> plot_1.show()
    >>> plot_2 = SeriesRLC(
    ...     resistance=10, inductance=0.5, capacitance=25.3e-6, frequency=50
    ... ).graph(
    ...     lower_frequency_cut=0.1, upper_frequency_cut=100, samples=1000,
    ...     show_legend=True,
    ... )
    >>> plot_2.show()


    Parameters
    ----------
    resistance:     float
                    Resistance (in Ohm) of the circuit.
    inductance:     float
                    Inductance (in Henry) of the circuit.
    capacitance:    float
                    Capacitance (in Hz) of the circuit.
    frequency:      float
                    Frequency (in Hz) at which the output gain should be
                    evaluated.
    """

    def __init__(
        self,
        resistance: float,
        inductance: float,
        capacitance: float,
        frequency: float
    ) -> None:
        """Form the Frequency Response Analysis System."""
        self.resistance = resistance
        self.inductance = inductance
        self.capacitance = capacitance
        self.frequency = frequency

    @property
    def resonance_frequency(self):
        """Resonance Frequency (in Hz) of the Described RLC Circuit."""
        return 1 / (_np.sqrt(self.inductance * self.capacitance) * 2 * _np.pi)

    @property
    def bandwidth(self):
        """Bandwidth of the Described RLC Circuit."""
        return self.resistance / (2 * _np.pi * self.inductance)

    @property
    def quality_factor(self):
        """Quality Factor of the Described RLC Circuit."""
        return 2 * _np.pi * self.frequency / self.resistance

    @property
    def lower_cutoff_frequency(self):
        """Lower Cutoff Frequency (in Hz) of the Described RLC Circuit."""
        x = (-self.resistance) / (2 * self.inductance)
        resonance_angular_frequency = 2 * _np.pi * self.resonance_frequency

        return (x + _np.sqrt(x**2 + resonance_angular_frequency**2)) / (2 * _np.pi)

    @property
    def upper_cutoff_frequency(self):
        """Upper Cutoff Frequency (in Hz) of the Described RLC Circuit."""
        x = (self.resistance) / (2 * self.inductance)
        resonance_angular_frequency = 2 * _np.pi * self.resonance_frequency

        return (x + _np.sqrt(x**2 + resonance_angular_frequency**2)) / (2 * _np.pi)

    def output_gain(self, frequency: float):
        """
        Evaluate Output Gain of Described RLC Circuit at a Particular Frequency.

        Parameters
        ----------
        frequency:  float
                    Frequency (in Hz) at which the output gain should be
                    evaluated.
        """
        ang_frq = 2 * _np.pi * frequency
        current_impedence = (
            self.resistance**2
            + (ang_frq * self.inductance - 1 / (ang_frq * self.capacitance)) ** 2
        )
        return (self.resistance) / (_np.sqrt(current_impedence))

    def legend(self):
        """Generate a Legend for the Graph."""
        f1, f2 = self.lower_cutoff_frequency, self.upper_cutoff_frequency
        f = self.resonance_frequency
        return [
            "Gain",
            f"Resonance frequency ({f}Hz)",
            f"Lower cutoff frequency ({f1}Hz)",
            f"Upper cutoff frequency ({f2}Hz)",
            f"Bandwidth ({f2 - f1}Hz)",
            f"Quality factor {self.quality_factor}",
        ]

    def graph(
        self,
        lower_frequency_cut: float,
        upper_frequency_cut: float,
        samples: int = 10000,
        show_legend: bool = False,
    ):
        """
        Generate a Plot to Represent all Data Respective of the RLC Circuit.

        Parameters
        ----------
        lower_frequency_cut:    float
                                Minimum frequency to demonstrate as a boundary
                                of the X-axis of the plot.
        upper_frequency_cut:    float
                                Maximum frequency to demonstrate as a boundary
                                of the X-axis of the plot.
        samples:                float, optional
                                Number of samples over which the plot should be
                                formed. Defaults to 1000.
        show_legend:            bool, optional
                                Control to enable or disable the display of the
                                legend. Defaults to False.
        """
        x = _np.linspace(lower_frequency_cut, upper_frequency_cut, samples)

        y = self.output_gain(x)

        _plt.title("Frequency response of series RLC circuit")
        _plt.grid(visible=True)

        _plt.plot(x, y, label="Gain")
        _plt.ylabel("Gain")
        _plt.xlabel("Frequency (Hz)")
        f1, f2 = self.lower_cutoff_frequency, self.upper_cutoff_frequency
        f = self.resonance_frequency
        _plt.scatter(
            [f1, f2],
            [_np.sqrt(0.5), _np.sqrt(0.5)],
            marker="*",
            c="black",
            label="_nolegend_",
        )

        half_power_gain = _np.sqrt(0.5)

        _plt.plot([f, f], [0, 1], ls="-.")
        _plt.plot([f1, f1], [half_power_gain, 0], ls="-.")
        _plt.plot([f2, f2], [half_power_gain, 0], ls="-.")
        _plt.plot([f1, f2], [half_power_gain, half_power_gain], ls="-.")

        _plt.plot(
            [0, f], [half_power_gain, half_power_gain], label="_nolegend_", ls="--"
        )
        _plt.plot([0, f], [1, 1], label="_nolegend_", ls="--")
        _plt.plot([0, 0], [half_power_gain, 1], label="Quality factor", c="black")
        _plt.scatter([0], [half_power_gain], label="_nolegend_", c="black", marker="v")
        _plt.scatter([0], [1], label="_nolegend_", c="black", marker="^")
        if show_legend:
            _plt.legend(self.legend(), loc='best')
        return _plt
# END
