import numpy as np

from electricpy import machines


def test_phase_shift_transformer_respects_signed_shift():
    phase_pos = machines.phase_shift_transformer(style="DY", shift=30)
    phase_neg = machines.phase_shift_transformer(style="DY", shift=-30)
    assert np.isclose(phase_pos, np.conj(phase_neg))
