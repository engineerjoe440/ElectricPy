import electricpy.thermal as thermal


def test_rtdtemp_allows_no_rounding():
    temp = thermal.rtdtemp(100.0, rtdtype="PT100", round=None)
    assert isinstance(temp, float)
