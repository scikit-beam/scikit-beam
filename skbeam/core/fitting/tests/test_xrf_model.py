import pytest
import numpy.testing as npt

from skbeam.core.fitting.xrf_model import get_line_energy


@pytest.mark.parametrize("elemental_line, energy_expected", [
    ("Ca_K", 3.6917),
    ("Ca_Ka", 3.6917),
    ("Ca_Ka1", 3.6917),
    ("Ca_Kb", 4.0127),
    ("Ca_Kb1", 4.0127),
    ("Ca_Ka3", 3.6003),
    ("Ca_ka3", 3.6003),
    ("Ca_kA3", 3.6003),
    ("Eu_L", 5.8460),
    ("Eu_La", 5.8460),
    ("Eu_La1", 5.8460),
    ("Eu_Lb", 6.4565),
    ("Eu_Lb1", 6.4565),
    ("Eu_Lb3", 6.5714),
    ("U_M", 3.1710),
    ("U_Ma", 3.1710),
    ("U_Ma1", 3.1710),
    ("U_Ma2", 3.1610),
])
def test_get_line_energy(elemental_line, energy_expected):
    energy = get_line_energy(elemental_line)
    npt.assert_almost_equal(energy, energy_expected,
                            err_msg="Energy doesn't match expected")
