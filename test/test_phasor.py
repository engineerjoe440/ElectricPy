import numpy as np
from electricpy.phasors import phs
from electricpy.phasors import phasor
from electricpy.phasors import vectarray
from numpy.testing import assert_almost_equal

def test_phasor():
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

class TestPhs():

    def test_0(self):
        inputs = [0, 90, 180, 270, 360]

        outputs = [phs(x) for x in inputs]
        actual_outputs = [1, 1j, -1, -1j, 1]

        for x,y in zip(outputs, actual_outputs):
            assert_almost_equal(x, y)

    def test_1(self):
        inputs = [30, 45, 60, 135]

        outputs = [phs(x) for x in inputs]
        actual_outputs = [0.866025+0.5j, 0.707106+0.707106j, 0.5+0.866025j, -0.707106+0.707106j]

        for x,y in zip(outputs, actual_outputs):
            assert_almost_equal(x, y, decimal = 3)

class TestVectarray():

    def test_0(self):

        A = [2+3j, 4+5j, 6+7j, 8+9j]
        B = vectarray(A)

        B_test = [[np.abs(x), np.degrees(np.angle(x))] for x in A]

        np.testing.assert_array_almost_equal(B, B_test)

    def test_1(self):

        A = np.random.random(size = 16)
        B = vectarray(A)

        B_test = [[np.abs(x), 0] for x in A]
        np.testing.assert_array_almost_equal(B, B_test)

        A = np.random.random(size = 16)*1j
        B = vectarray(A)

        B_test = [[np.abs(x), 90] for x in A]
        np.testing.assert_array_almost_equal(B, B_test)
