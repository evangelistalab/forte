import forte
import pytest
from forte import det


def test_determinant_hilber_space():
    dets = forte.hilbert_space(2, 1, 1)
    # compare with the expected result of the determinant
    expected = [det("20"), det("+-"), det("-+"), det("02")]
    assert len(dets) == 4
    assert sorted(dets) == sorted(expected)

    dets = forte.hilbert_space(2, 1, 1, nirrep=2, mo_symmetry=[0, 1], symmetry=0)
    # compare with the expected result of the determinant
    expected = [det("20"), det("02")]
    assert len(dets) == 2
    assert sorted(dets) == sorted(expected)

    dets = forte.hilbert_space(2, 1, 1, nirrep=2, mo_symmetry=[0, 1], symmetry=1)
    # compare with the expected result of the determinant
    expected = [det("+-"), det("-+")]
    assert len(dets) == 2
    assert sorted(dets) == sorted(expected)

    # test with the case of 6 electrons and 6 orbitals
    dets = forte.hilbert_space(6, 3, 3, 8, [0, 2, 3, 5, 6, 7], 0)
    assert len(dets) == 56


def test_determinant_hilber_space_edge_cases():
    dets = forte.hilbert_space(1, 1, 1)
    # compare with the expected result of the determinant
    expected = [det("2")]
    assert len(dets) == 1
    assert dets == expected

    dets = forte.hilbert_space(1, 1, 0)
    # compare with the expected result of the determinant
    expected = [det("+")]
    assert len(dets) == 1
    assert dets == expected

    dets = forte.hilbert_space(1, 0, 1)
    # compare with the expected result of the determinant
    expected = [det("-")]
    assert len(dets) == 1
    assert dets == expected

    dets = forte.hilbert_space(2, 0, 0)
    # compare with the expected result of the determinant
    assert len(dets) == 1
    assert dets == [det("")]

    dets = forte.hilbert_space(4, 2, 0)
    # compare with the expected result of the determinant
    expected = [det("++00"), det("+0+0"), det("+00+"), det("0++0"), det("0+0+"), det("00++")]
    assert len(dets) == 6
    assert sorted(dets) == sorted(expected)


if __name__ == "__main__":
    test_determinant_hilber_space()
    test_determinant_hilber_space_edge_cases()
