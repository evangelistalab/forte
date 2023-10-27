import pytest
from forte import Symmetry


def test_symmetry():
    """Test the Symmetry class"""

    sym = Symmetry('D2H')
    assert sym.point_group_label() == 'D2H'

    assert sym.irrep_labels() == ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']

    # test in and out of bounds
    assert sym.irrep_label(4) == 'Au'
    with pytest.raises(RuntimeError):
        sym.irrep_label(9)

    # test case insensitive access
    assert sym.irrep_label_to_index('au') == 4
    assert sym.irrep_label_to_index('Au') == 4
    assert sym.irrep_label_to_index('AU') == 4

    # test wrong irreps
    with pytest.raises(RuntimeError):
        sym.irrep_label_to_index('X')
    with pytest.raises(RuntimeError):
        sym.irrep_label_to_index('')

    assert sym.nirrep() == 8

    for h in range(8):
        for g in range(8):
            assert Symmetry.irrep_product(h, g) == h ^ g


def test_bad_symmetry():
    """Test initializing a Symmetry object with an invalid point group"""
    with pytest.raises(RuntimeError):
        Symmetry('D5H')


if __name__ == '__main__':
    test_symmetry()
