import numpy.testing as npt
import pytest
from xraylib import SymbolToAtomicNumber

from skbeam.core.fitting.xrf_model import K_LINE, L_LINE, M_LINE, get_line_energy


@pytest.mark.parametrize(
    "elemental_line, energy_expected",
    [
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
    ],
)
def test_get_line_energy(elemental_line, energy_expected):
    energy = get_line_energy(elemental_line)
    npt.assert_almost_equal(energy, energy_expected, err_msg="Energy doesn't match expected")


def test_K_L_M_lines():
    """
    Test the lists of supported emission lines.
    """
    lines = K_LINE + L_LINE + M_LINE

    # Check that all lines are unique
    assert len(lines) == len(set(lines))

    for line in lines:
        element, type = line.split("_")
        # Check emission line name
        assert type in ("K", "L", "M")
        # Check that the element is supported
        try:
            # Returns the number > 0 if successful. May return 0 (xraylib 3) or
            # raise an exception (xraylib 4) if element is not recognized.
            assert SymbolToAtomicNumber(element) > 0
        except ValueError as ex:
            assert False, f"{ex}: '{element}' (emission line: '{line}')"
