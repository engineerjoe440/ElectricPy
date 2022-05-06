################################################################################
"""
`electricpy.visu` - Support for plotting and visualizations.

Filled with plotting functions and visualization tools for electrical engineers,
this module is designed to assist engineers visualize their designs.
"""
################################################################################

import numpy as _np
import matplotlib.pyplot as _plt
import cmath
from electricpy import geometry
from electricpy.geometry import Point
from electricpy.geometry.circle import Circle

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
                        Ration between rotor resitance to stator resistance
                        (i.e., R2/R1)
    frequency:          int
                        AC supply frequency
    poles:              int
                        Pole count of induction Motor
    """

    def __init__(self, no_load_data, blocked_rotor_data, output_power,
                torque_ration=1, frequency=50, poles=4):
        """Primary Entrypoint."""
        self.no_load_data = no_load_data
        self.blocked_rotor_data = blocked_rotor_data
        self.f = frequency
        self.operating_power = output_power
        self.torque_ratio = torque_ration
        self.poles = poles
        self.sync_speed = 120 * frequency / poles  # rpm

        v0 = no_load_data['V0']
        i0 = no_load_data['I0']
        w0 = no_load_data['W0']

        self.no_load_pf = w0 / (_np.sqrt(3) * v0 * i0)
        theta0 = _np.arccos(self.no_load_pf)

        # get short circuit power factor and Current at slip=1
        vsc = blocked_rotor_data['Vsc']
        isc = blocked_rotor_data['Isc']
        wsc = blocked_rotor_data['Wsc']

        self.blocked_rotor_pf = wsc / (_np.sqrt(3) * vsc * isc)
        theta_sc = _np.arccos(self.blocked_rotor_pf)

        # because V is on Y axis
        theta0 = _np.pi / 2 - theta0
        theta_sc = _np.pi / 2 - theta_sc

        # isc is the current at reduced voltage
        # calculate current at rated voltage
        isc = v0 * isc / vsc
        self.no_load_line = [
            [0, i0 * _np.cos(theta0)],
            [0, i0 * _np.sin(theta0)]
        ]
        self.full_load_line = [
            [0, isc * _np.cos(theta_sc)],
            [0, isc * _np.sin(theta_sc)]
        ]

        # secondary current line
        self.secondary_current_line = [
            [i0 * _np.cos(theta0), isc * _np.cos(theta_sc)],
            [i0 * _np.sin(theta0), isc * _np.sin(theta_sc)]
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
        self.torque_max, self.torque_max_x, self.torque_max_y = \
            self.get_torque_max()
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
            self.center,
            self.radius,
            semi=True
        )
        _plt.plot(circle_x, circle_y)

        InductionMotorCircle.__plot_line(self.no_load_line)
        InductionMotorCircle.__plot_line(self.secondary_current_line)
        InductionMotorCircle.__plot_line(self.full_load_line, ls='-.')
        InductionMotorCircle.__plot_line(self.torque_line, ls='-.')

        # Full load output
        _plt.plot(
            [self.secondary_current_line[0][1],
            self.secondary_current_line[0][1]],
            [self.secondary_current_line[1][1], self.center_y])
        # Diameter of the circle
        _plt.plot([self.center_x - self.radius, self.center_x + self.radius],
                [self.center_y, self.center_y], ls='-.')
        # Max torque line
        _plt.plot(
            [self.center_x, self.torque_max_x],
            [self.center_y, self.torque_max_y], ls='-.')
        # Max Output Power line
        _plt.plot(
            [self.center_x, self.center_x - self.radius * _np.sin(self.theta)],
            [self.center_y, self.center_y + self.radius * _np.cos(self.theta)],
            ls='-.'
        )
        # Operating Point
        _plt.plot([0, self.power_x], [0, self.power_y], c='black')

        _plt.scatter(self.power_x, self.power_y, marker='X', c='red')
        # mark the center of the circle
        _plt.scatter(self.center_x, self.center_y, marker='*', c='blue')
        _plt.scatter(
            self.center_x - self.radius * _np.sin(self.theta),
            self.center_y + self.radius * _np.cos(self.theta),
            linewidths=3, c='black', marker='*'
        )
        _plt.scatter(
            self.torque_max_x,
            self.torque_max_y,
            linewidths=3,
            c='black',
            marker='*'
        )

        _plt.title("Induction Motor Circle Diagram")
        _plt.grid()
        _plt.legend([
            'I2 locus',
            'No Load Current',
            'Output Line',
            'Blocked Rotor Current',
            'Torque line',
            'Full Load Losses',
            'Diameter',
            'Maximum Torque',
            'Maximum Output Power',
            f'Operating Power {self.operating_power}'
        ])
        return _plt

    def compute_efficiency(self):
        """Compute the output efficiency of induction motor."""
        [[_, no_load_x], [_, no_load_y]] = self.no_load_line
        no_load_losses = no_load_y * self.power_scale

        compute_slope = InductionMotorCircle.compute_slope

        torque_slope = compute_slope(self.torque_line)
        stator_cu_loss = (self.power_x - no_load_x) * \
                         torque_slope * self.power_scale

        rotor_current_slope = compute_slope(self.secondary_current_line)
        total_cu_loss = (self.power_x - no_load_x) * \
                        rotor_current_slope * self.power_scale

        rotor_cu_loss = total_cu_loss - stator_cu_loss

        rotor_output = self.power_y * self.power_scale - \
                    (rotor_cu_loss + stator_cu_loss + no_load_losses)

        slip = rotor_cu_loss / rotor_output

        self.rotor_speed = self.sync_speed * (1 - slip)

        data = {
            'no_load_loss': no_load_losses,
            'rotor_copper_loss': rotor_cu_loss,
            'stator_copper_loss': stator_cu_loss,
            'rotor_output': rotor_output,
            'slip': slip,
            'stator_rmf_speed (RPM)': self.sync_speed,
            'rotor_speed (RMP)': self.rotor_speed,
            'power_factor': (self.power_y / _np.sqrt(self.power_x ** 2 + self.power_y ** 2)),
            'efficiency': f"{rotor_output * 100 / (self.power_y * self.power_scale)} %"
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
    def __plot_line(line, mark_start=True, mark_end=True, ls='-', marker=None):
        """Supporting function to plot a line."""
        [x, y] = line
        [x1, x2] = x
        [y1, y2] = y
        _plt.plot(x, y, ls=ls)
        if mark_start:
            _plt.scatter(x1, y1, marker=marker)
        if mark_end:
            _plt.scatter(x2, y2, marker=marker)

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
            (self.operating_power / self.power_scale + (center_x - no_load_x) * _np.tan(alpha)) *
            _np.cos(alpha) / self.radius)
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
    system both sending and reciving ends.

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

    def __init__(self, power_circle_type: str, power_factor: float = None,
                 Vr: complex = None, Vs: complex = None,
                 Pr: float = None, Qr: float = None, Sr: complex = None,
                 Ps: float = None, Qs: float = None, Ss: complex = None,
                 A: complex = None, B: complex = None, C: complex = None,
                 D: complex = None) -> None:
        r"""Initialize the class."""
        if C is not None:
            assert abs(A*D - B*C - 1) < 1e-6, "ABCD Matrix is not a valid ABCD Matrix"

        if power_circle_type.lower() == "receiving":

            if A != None and B != None and Vr != None:
                self.radius, self.center, self.operating_point = PowerCircle._build_circle(A, B, "receiving_end", Vr, 
                Pr, Qr, Sr, power_factor, Vs)
            else:
                raise ValueError("Not enough attributes to build circle")

        elif power_circle_type.lower() == "sending":
            
            if B != None and D != None and Vs != None:
                self.radius, self.center, self.operating_point = PowerCircle._build_circle(D, B, "sending_end", Vs, 
                Ps, Qs, Ss, power_factor, Vr)
            else:
                raise ValueError("Not enough attributes to build power circle")

        else:
            raise ValueError("Invalid power circle type")

        self.circle = Circle(self.center, self.radius)
        self.parameters = locals()

    @staticmethod
    def _build_circle(a1, a2, circle_type, V, P = None, Q = None, S = None, power_factor = None, V_ref = None):

        k = (abs(V)**2)*abs(a1)/abs(a2)
        alpha = cmath.phase(a1)
        beta = cmath.phase(a2)
        
        if circle_type == "receiving_end":
            center = Point(-k*cmath.cos(alpha - beta), -k*cmath.sin(alpha - beta))

        elif circle_type == "sending_end":
            center = Point(k*cmath.cos(alpha -beta), -k*cmath.sin(alpha - beta))

        if V_ref != None and P != None and Q != None:
            radius = abs(V)*abs(V_ref)/(abs(a2))
            operation_point = Point(P, Q)

        elif V_ref != None and S != None:
            radius = abs(V)*abs(V_ref)/(abs(a2))
            operation_point = Point(S.real, S.imag)

        elif P != None and Q != None:
            radius = geometry.distance(center, Point(P, Q))
            operation_point = Point(P, Q)

        elif S != None:
            radius = geometry.distance(center, Point(S.real, S.imag))
            operation_point = Point(S.real, S.imag)

        elif P != None and power_factor != None:

            Q = P*cmath.sqrt(1/power_factor**2 - 1).real

            if power_factor < 0:
                Q = -Q

            radius = geometry.distance(center, Point(P, Q))
            operation_point = Point(P, Q)

        elif Q != None and power_factor != None:
            P = Q/cmath.sqrt(1/power_factor**2 - 1).real
            radius = geometry.distance(center, Point(P, Q))
            operation_point = Point(P, Q)

        else:
            raise AttributeError("Enought attributes to calculate not found")

        return radius, center, operation_point

    def _cal_parameters(self, type1, type2):
        
        if self.parameters['V'+type2] == None:
            self.parameters['V' + type2] = abs(self.parameters['B'])*self.radius/self.parameters['V' + type1]

        if self.parameters['P'+type1] == None:
            self.parameters['P' + type1] = self.operating_point.x

        if self.parameters['Q'+type1] == None:
            self.parameters['Q' + type1] = self.operating_point.y

        if self.parameters['S'+type1] == None:
            self.parameters['S' + type1] = self.operating_point.x + 1j*self.operating_point.y

        if self.parameters['power_factor'] == None:
            self.parameters['power_factor'] = self.operating_point.y/self.operating_point.x

        if type1 == 'r' and type2 == 's':
            self.parameters["Vs"] = self.parameters['B']*self.parameters["Sr"] + self.parameters["A"] * abs(self.parameters["Vr"])**2
            self.parameters["Vs"] = self.parameters["Vs"]/self.parameters["Vr"].conjugate()

        elif type1 == 's' and type2 == 'r':
            self.parameters["Vr"] = -self.parameters['B']*self.parameters["Ss"] + self.parameters["D"] * abs(self.parameters["Vs"])**2
            self.parameters["Vr"] = self.parameters["Vr"]/self.parameters["Vs"].conjugate()

    def print_data(self):
        r"""Print the data of the circle."""
        if self.operating_point == None:
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

        #plot Circle and Diameter
        _plt.plot(circle_x, circle_y)
        _plt.plot([c_x - self.radius, c_x + self.radius], [c_y, c_y], 'g--')
        _plt.plot([c_x, c_x], [c_y - self.radius, c_y + self.radius], 'g--')

        _plt.plot([c_x, op_x], [c_y, op_y], 'y*-.')
        _plt.plot([op_x, op_x], [op_y, c_y], 'b*-.') 
        _plt.scatter(op_x, op_y, marker='*', color='r')
        _plt.title(
            f"{self.parameters['power_circle_type'].capitalize()} Power Circle"
        )
        _plt.xlabel("Active Power")
        _plt.ylabel("Reactive Power")
        _plt.grid()
        return _plt

def receiving_end_power_circle(Vr: complex = None, A: complex = None,
                               B: complex = None, Pr:float = None,
                               Qr: float = None, Sr: complex = None,
                               power_factor: float = None, Vs: complex = None
                            ) -> PowerCircle :
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
        assert Vr != None and A != None and B != None
    except AssertionError:
        raise ValueError(
            "Not enough attributes to build Receiving end power circle at least"
            " provide `Vr`, `A`, `B`"
        )

    if not (
        ((Pr != None and Qr != None) or (Sr != None and power_factor != None))
        or
        (
            (Pr != None and power_factor != None) or
            (Qr != None and power_factor != None)
        )):
        raise ValueError(
            "Not enough attributes for marking an operating point on Receiving "
            "End Power Circle"
        )

    return PowerCircle(
        "receiving",
        **{
            'Vr': Vr,
            'A': A,
            'B': B,
            'Pr': Pr,
            'Qr': Qr,
            'Sr': Sr,
            'Vs': Vs,
            'power_factor': power_factor
        }
    )

def sending_end_power_circle(Vs: complex = None, B: complex = None,
                             D: complex = None, Ps:float = None,
                             Qs:float = None, Ss: complex = None,
                             power_factor: float = None, Vr: complex = None
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
    if not (Vs != None and B != None and D != None):
        raise ValueError(
            "Not enough attributes to build Sending end power circle at least "
            "provide `Vs`, `B`, `D`"
        )

    if not (
        ((Ps != None and Qs != None) or (Ss != None and power_factor != None))
        or
        (
            (Ps != None and power_factor != None) or
            (Qs != None and power_factor != None)
        )):
        raise ValueError(
            "Not enough attributes for marking an operating point on Sending "
            "End Power Circle"
        )

    return PowerCircle(
        "sending",
        **{
            'Vr': Vr,
            'B': B,
            'D': D,
            'Ps': Ps,
            'Qs': Qs,
            'Ss': Ss,
            'Vs': Vs,
            'power_factor': power_factor
        }
    )

# END
