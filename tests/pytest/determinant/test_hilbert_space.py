import forte
import pytest
from forte import det


def test_determinant_hilbert_space():
    dets = forte.hilbert_space(2, 1, 1)
    # compare with the expected result of the determinant
    expected = [det("20"), det("+-"), det("-+"), det("02")]
    assert len(dets) == 4
    assert sorted(dets) == sorted(expected)

    # non-aufbau case
    ref = det("002")
    dets = forte.hilbert_space(3, 1, 1, ref, truncation=2)
    expected = [
        det("002"),
        det("0+-"),
        det("0-+"),
        det("+0-"),
        det("-0+"),
        det("020"),
        det("200"),
        det("+-0"),
        det("-+0"),
    ]
    assert len(dets) == 9
    assert sorted(dets) == sorted(expected)

    ref = det("0+-")
    dets = forte.hilbert_space(3, 1, 1, ref, truncation=1)
    expected = [det("0+-"), det("-+0"), det("+0-"), det("002"), det("020")]
    assert len(dets) == 5
    assert sorted(dets) == sorted(expected)

    ref = det("20")
    dets = forte.hilbert_space(2, 1, 1, ref, truncation=1)
    # compare with the expected result of the determinant
    expected = [det("20"), det("+-"), det("-+")]
    assert len(dets) == 3, dets
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

    ref = det("22000")
    dets = forte.hilbert_space(5, 2, 2, ref, truncation=1, nirrep=4, mo_symmetry=[0, 1, 2, 3, 0], symmetry=2)
    # compare with the expected result of the determinant
    expected = [det("2+0-0"), det("2-0+0"), det("+2-00"), det("-2+00")]
    assert len(dets) == 4
    assert sorted(dets) == sorted(expected)

    # test with the case of 6 electrons and 6 orbitals
    dets = forte.hilbert_space(6, 3, 3, 8, [0, 2, 3, 5, 6, 7], 0)
    assert len(dets) == 56

    # test with the case of 6 electrons and 7 orbitals, truncated to 2 excitations
    ref = det("2220000")
    dets = forte.hilbert_space(7, 3, 3, ref, truncation=2)
    assert len(dets) == 205

    # test with the case of 16 electrons and 18 orbitals, truncated to 3 excitations
    ref = det("2" * 8 + "0" * 10)
    dets = forte.hilbert_space(18, 8, 8, ref, truncation=3)
    assert len(dets) == 224121


def test_determinant_hilbert_space_edge_cases():
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

    ref = det("++00")
    dets = forte.hilbert_space(4, 2, 0, ref, truncation=1)
    # compare with the expected result of the determinant
    expected = [det("++00"), det("+0+0"), det("+00+"), det("0++0"), det("0+0+")]
    assert len(dets) == 5
    assert sorted(dets) == sorted(expected)


if __name__ == "__main__":
    test_determinant_hilbert_space()
    test_determinant_hilbert_space_edge_cases()
