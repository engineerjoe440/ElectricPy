import numpy as np

from electricpy import constants


def test_constants_shapes_and_values():
    """Validate constants shapes and values behavior."""
    assert constants.pi == np.pi
    assert np.isclose(abs(constants.VLLcVLN), np.sqrt(3))
    assert constants.Aabc.shape == (3, 3)
