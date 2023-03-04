import cmath
import numpy as np

def test_abc_to_seq():

    from electricpy.conversions import abc_to_seq
    a = cmath.rect(1, np.radians(120))

    def test_0():
        np.testing.assert_array_almost_equal(abc_to_seq([1, 1, 1]), [1+0j, 0j, 0j])
        np.testing.assert_array_almost_equal(abc_to_seq([1, 0, 0]), [1/3+0j, 1/3+0j, 1/3+0j])
        np.testing.assert_array_almost_equal(abc_to_seq([0, 1, 0]), [1/3+0j, a/3+0j, a*a/3+0j])
        np.testing.assert_array_almost_equal(abc_to_seq([0, 0, 1]), [1/3+0j, a*a/3, a/3])

    test_0()

def test_seq_to_abc():
    from electricpy.conversions import seq_to_abc
    a = cmath.rect(1, np.radians(120))

    def test_0():
        np.testing.assert_array_almost_equal(seq_to_abc([1, 1, 1]), [3+0j, 0j, 0j])
        np.testing.assert_array_almost_equal(seq_to_abc([1, 0, 0]), [1+0j, 1+0j, 1+0j])
        np.testing.assert_array_almost_equal(seq_to_abc([0, 1, 0]), [1+0j, a*a+0j, a+0j])
        np.testing.assert_array_almost_equal(seq_to_abc([0, 0, 1]), [1+0j, a, a*a])

    test_0()