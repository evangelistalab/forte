import pytest

from forte import Results


def test_results():
    """Test the Results class."""

    res = Results()
    res.add('energy', -1.0, "The total energy", "Eh")
    res.add('dipole', [0.0, 0.0, 3.1], "The dipole moment vector", "D")

    assert res.value('energy') == -1.0
    assert res.description('energy') == "The total energy"
    assert res.units('energy') == "Eh"
    assert res.value('dipole') == pytest.approx([0.0, 0.0, 3.1], 1.0e-10)
    assert res.units('dipole') == "D"


if __name__ == "__main__":
    test_results()
