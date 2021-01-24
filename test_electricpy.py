import electricpy
import unittest
import numpy as np

class TestMain(unittest.TestCase):

    def test_bridge_impedance(self):
        
        # Perfectly Balanced Wheate Stone Bridge
        z1 = complex(1, 0)
        z2 = complex(1, 0)
        z3 = complex(1, 0)
        z4 = complex(1, 0)
        z5 = complex(np.random.random(), np.random.random())

        zeq = electricpy.bridge_impedance(z1, z2, z3, z4, z5)

        zactual = complex(1, 0)

        self.assertEqual(zeq, zactual)


        # Balanced Wheate Stone Bridge
        z1 = complex(2, 0)
        z2 = complex(4, 0)
        z3 = complex(8, 0)
        z4 = complex(4, 0)
        z5 = complex(np.random.random(), np.random.random())

        zeq = electricpy.bridge_impedance(z1, z2, z3, z4, z5)

        zactual = complex(4, 0)

        self.assertEqual(zeq, zactual)

        # Base Case
        z1 = complex(10, 0)
        z2 = complex(20, 0)
        z3 = complex(30, 0)
        z4 = complex(40, 0)
        z5 = complex(50, 0)

        zeq = electricpy.bridge_impedance(z1, z2, z3, z4, z5)

        zactual = complex(4+(50/3), 0)

        self.assertAlmostEqual(zeq, zactual, 6)
    
    def test_suspension_insulators(self):
        ''' 
            Electric Power Systems by C.L Wadhwa Overhead Line Insulator example
        '''

        number_capacitors = 5

        capacitance_ration = 5

        Voltage = 66

        capacitor_disk_voltages, string_efficiency = electricpy.suspension_insulators(number_capacitors,\
                                                        capacitance_ration, Voltage)

        string_efficiency_actual = 54.16
        
        self.assertAlmostEqual(string_efficiency, string_efficiency_actual, 2)
    
    def test_propagation_constants(self):

        z = complex(0.5, 0.9)

        y = complex(0, 6e-6)

        params_dict = electricpy.propagation_constants(z, y, 520)

        alpha_cal = 0.622*(10**-3)

        beta_cal = 2.4*(10**-3)

        self.assertAlmostEqual(params_dict['alpha'], alpha_cal, 4)
        self.assertAlmostEqual(params_dict['beta'], beta_cal, 4)

    def test_dynetz(self):

        z1 = complex(3, 3)
        z2 = complex(3, 3)
        z3 = complex(3, 3)

        za, zb, zc = electricpy.dynetz(delta = (z1, z2, z3))

        self.assertEqual((za, zb, zc), (z1/3, z2/3, z3/3))

        za, zb, zc = electricpy.dynetz(wye = (z1, z2, z3))

        self.assertEqual((za, zb, zc), (3*z1, 3*z2, 3*z3))




if __name__ == '__main__':
    unittest.main()
    '''
        ----------------------------------------------------------------------
        Ran 4 tests in 0.001s

        OK
    '''


