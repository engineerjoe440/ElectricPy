import pytest
from electricpy.opamp import non_inverting


class TestNonInverting:

    def compare_results(self):
        for test_parameter, expected_output in zip(
            self.test_parameters,
            self.expected_outputs
        ):
            computed_output = non_inverting(**test_parameter)
            assert expected_output == computed_output

    def test_0(self):

        # perfectly valid inputs

        self.test_parameters = [
        {
            "Vin": 10, "Rg": 10, "Rf": 10
        },
        {
            "Vout": 10, "Rg": 10, "Rf": 10
        },
        {
            "Vin": 10, "Vout": 20, "Rg": 2,
        },
        {
            "Vin": 10, "Vout": 20, "Rf": 2,
        }]

        self.expected_outputs = [
        {
            "Vin": 10, "Rg": 10, "Rf": 10, "Vout": 20
        },
        {
            "Vin": 5, "Rg": 10, "Rf": 10, "Vout": 10
        },
        {
            "Vin": 10, "Rg": 2, "Rf": 2, "Vout": 20
        },
        {
            "Vin": 10, "Rg": 2, "Rf": 2, "Vout": 20
        }]

        self.compare_results()

    def test_1(self):

        test_parameter = {
            "Vin": None,
            "Vout": None,
            "Rg": 2,
            "Rf": 2,
        }

        with pytest.raises(ValueError):
            non_inverting(**test_parameter)


