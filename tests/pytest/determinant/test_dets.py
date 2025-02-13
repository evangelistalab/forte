#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte
import pytest


def det(s):
    return forte.det(s)


def test_det_constructors():
    """Test class constructors"""
    print("Testing determinant interface")
    d1 = det("22")
    d2 = forte.Determinant(d1)
    d3 = forte.Determinant([True, True], [True, True])
    assert d1 == d2
    assert d1 == d3


def test_det_fill():
    if forte.Determinant.norb() >= 64:
        n = 128
        print("Testing determinant fill function")
        for i in range(n + 1):
            d = forte.Determinant()
            d.fill_up_to(i)
            for j in range(i):
                assert d.get_bit(j) == 1
            for j in range(i, n):
                assert d.get_bit(j) == 0
    if forte.Determinant.norb() >= 128:
        n = 256
        print("Testing determinant fill function")
        for i in range(n + 1):
            d = forte.Determinant()
            d.fill_up_to(i)
            for j in range(i):
                assert d.get_bit(j) == 1
            for j in range(i, n):
                assert d.get_bit(j) == 0


def test_det_getset():
    """Test get/set operations"""
    print("Testing determinant interface")
    d = det("")
    assert d.get_alfa_bit(0) == 0
    assert d.get_alfa_bit(7) == 0
    assert d.get_beta_bit(0) == 0
    assert d.get_beta_bit(7) == 0
    d.set_alfa_bit(7, True)
    assert d.get_alfa_bit(7) == 1
    d.set_beta_bit(7, True)
    assert d.get_beta_bit(7) == 1
    d.set_alfa_bit(7, False)
    assert d.get_alfa_bit(7) == 0
    d.set_beta_bit(7, False)
    assert d.get_beta_bit(7) == 0


def test_det_get_occ_vir():
    """Test functions that return the index of occupied/virtual orbitals"""
    print("Testing functions that return the index of occupied/virtual orbitals")
    d1 = det("2a2b")
    # test get functions when used correctly
    assert d1.get_alfa_occ(4) == [0, 1, 2]
    assert d1.get_beta_occ(4) == [0, 2, 3]
    assert d1.get_alfa_vir(4) == [3]
    assert d1.get_beta_vir(4) == [1]
    # test some safety checks
    n = forte.Determinant.norb()
    with pytest.raises(ValueError):
        d1.get_alfa_occ(n + 1)
    with pytest.raises(ValueError):
        d1.get_beta_occ(n + 1)
    with pytest.raises(ValueError):
        d1.get_alfa_vir(n + 1)
    with pytest.raises(ValueError):
        d1.get_beta_vir(n + 1)


def test_det_creann():
    d = det("2200")
    sign = d.create_alfa_bit(2)
    assert d == det("22+0")
    assert sign == +1.0

    sign = d.create_alfa_bit(3)
    assert d == det("22++")
    assert sign == -1.0

    sign = d.destroy_alfa_bit(0)
    assert d == det("-2++")
    assert sign == 1.0

    sign = d.destroy_alfa_bit(2)
    assert d == det("-20+")
    assert sign == -1.0

    sign = d.create_beta_bit(2)
    assert d == det("-2-+")
    assert sign == 1.0

    sign = d.create_beta_bit(3)
    assert d == det("-2-2")
    assert sign == -1.0

    sign = d.destroy_beta_bit(0)
    assert d == det("02-2")
    assert sign == 1.0

    d = det("+")
    sign = d.create_beta_bit(0)
    assert d == det("2")
    assert sign == -1.0

    sign = d.create_beta_bit(0)
    assert sign == 0.0

    d = det("2")
    sign = d.create_alfa_bit(0)
    assert sign == 0.0

    d = det("2")
    sign = d.destroy_alfa_bit(1)
    assert sign == 0.0

    d = det("2")
    sign = d.destroy_beta_bit(1)
    assert sign == 0.0


def test_det_equality():
    """Test the __eq__ operator"""
    d1 = det("22")
    d2 = det("2+")
    d3 = det("22")
    d4 = det("0022")
    assert d1 == d1
    assert d1 != d2
    assert d1 == d3
    assert d2 != d4
    assert d1 != d4


def test_det_hash():
    """Test the __hash__ operator"""
    d1 = det("22")
    d2 = det("2+")
    d3 = det("22")
    d4 = det("0022")
    h = {}
    h[d1] = 1.0
    h[d2] = 2.0
    h[d3] += 0.25
    h[d4] = 3.0
    assert h[d1] == 1.25
    assert h[d3] == 1.25
    assert h[d2] == 2.00
    assert h[d4] == 3.00


def test_det_sorting():
    """Test the __lt__ operator"""
    d1 = det("22")
    d2 = det("2+")
    d3 = det("--")
    d4 = det("22")
    list = [d1, d2, d3, d4]
    sorted_list = sorted(list)
    assert sorted_list[0] == d2
    assert sorted_list[1] == d3
    assert sorted_list[2] == d1
    assert sorted_list[3] == d4
    assert sorted_list[2] == d4
    assert sorted_list[3] == d1


def test_det_exciting():
    """Test the gen_operator function"""
    # test a -> a excitation
    d1 = det("220")
    assert d1.gen_excitation([0], [3], [], []) == -1.0
    assert d1 == det("-20+")

    # test b -> b excitation
    d2 = det("2-+0")
    assert d2.gen_excitation([], [], [0, 1], [2, 3]) == -1.0
    assert d2 == det("+02-")

    # test b creation and counting number of a
    d3 = det("+000")
    assert d3.gen_excitation([], [], [], [0]) == -1.0
    assert d3 == det("2")
    d3 = det("0000")
    assert d3.gen_excitation([], [], [], [0]) == +1.0
    assert d3 == det("-")

    # test ab creation and sign
    d4 = det("000")
    assert d4.gen_excitation([], [2, 1], [], [0, 1]) == -1.0
    assert d4 == det("-2+")
    d5 = det("000")
    assert d5.gen_excitation([], [2, 1], [], [1, 0]) == +1.0
    assert d5 == det("-2+")
    d6 = det("000")
    assert d6.gen_excitation([], [1, 2], [], [0, 1]) == +1.0
    assert d6 == det("-2+")
    d7 = det("000")
    assert d7.gen_excitation([], [1, 2], [], [1, 0]) == -1.0
    assert d7 == det("-2+")


def test_excitation_connection():
    """Test the excitation_connection function"""
    d1 = det("220")
    d2 = det("022")
    conn = d1.excitation_connection(d2, 3)
    assert conn[0] == [0]  # alfa hole
    assert conn[1] == [2]  # alfa particle
    assert conn[2] == [0]  # beta hole
    assert conn[3] == [2]  # beta particle
    conn = d2.excitation_connection(d1, 3)
    assert conn[0] == [2]  # alfa hole
    assert conn[1] == [0]  # alfa particle
    assert conn[2] == [2]  # beta hole
    assert conn[3] == [0]  # beta particle

    # test different number of electrons
    d1 = det("2")
    d2 = det("0")
    conn = d1.excitation_connection(d2, 1)
    assert conn[0] == [0]  # alfa hole
    assert conn[1] == []  # alfa particle
    assert conn[2] == [0]  # beta hole
    assert conn[3] == []  # beta particle
    conn = d2.excitation_connection(d1, 1)
    assert conn[0] == []  # alfa hole
    assert conn[1] == [0]  # alfa particle
    assert conn[2] == []  # beta hole
    assert conn[3] == [0]  # beta particle

    d1 = det("222+-00000")
    d2 = det("-++0200-02")
    conn = d1.excitation_connection(d2, 10)
    assert conn[0] == [0, 3]  # alfa hole
    assert conn[1] == [4, 9]  # alfa particle
    assert conn[2] == [1, 2]  # beta hole
    assert conn[3] == [7, 9]  # beta particle

    # test for when some electrons get masked by the norb argument
    d1 = det("222+-00000")
    d2 = det("-++0200-02")
    conn = d1.excitation_connection(d2, 9)
    assert conn[0] == [0, 3]  # alfa hole
    assert conn[1] == [4]  # alfa particle
    assert conn[2] == [1, 2]  # beta hole
    assert conn[3] == [7]  # beta particle


def test_det_symmetry():
    """Test class constructors"""
    if forte.Determinant.norb() >= 128:
        print("Testing determinant symmetry function")
        symm = [0, 1, 2, 3, 4, 5, 6, 7] * 32
        d = det(
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        assert d.symmetry(symm) == 0
        d = det(
            "000000000000000000000000000000000000000000000000000000000000000+"
            "-000000000000000000000000000000000000000000000000000000000000000"
        )
        assert d.symmetry(symm) == 7
        d = det(
            "000000000000000000000000000000000000000000000000000000000000000-"
            "-000000000000000000000000000000000000000000000000000000000000000"
        )
        assert d.symmetry(symm) == 7
        d = det(
            "000000000000000000000000000000000000000000000000000000000000000-"
            "-00000000000000000000000000000000000000000000000000000000000000+"
        )
        assert d.symmetry(symm) == 0
        d = det(
            "000-00000000000000000000000000000000000000000000000000000000000-"
            "-00000000000000000000000000000000000000000000000000000000000000+"
        )
        assert d.symmetry(symm) == 3


def test_det_slater_sign():
    """Test Slater sign functions"""

    #        012345
    # parity 011001
    d = det("+0+0++")
    assert d.slater_sign(0) == 1
    assert d.slater_sign(1) == -1
    assert d.slater_sign(2) == -1
    assert d.slater_sign(3) == 1
    assert d.slater_sign(4) == 1
    assert d.slater_sign(5) == -1
    assert d.slater_sign(6) == 1
    assert d.slater_sign(7) == 1
    assert d.slater_sign_reverse(0) == -1
    assert d.slater_sign_reverse(1) == -1
    assert d.slater_sign_reverse(2) == 1
    assert d.slater_sign_reverse(3) == 1
    assert d.slater_sign_reverse(4) == -1
    assert d.slater_sign_reverse(5) == 1
    assert d.slater_sign_reverse(6) == 1

    if forte.Determinant.norb() >= 128:
        print("Testing determinant Slater sign functions")
        d = det("2" * 128)
        for i in range(128):
            assert d.slater_sign(i) == (-1) ** (i)
        for i in range(128):
            assert d.slater_sign_reverse(i) == (-1) ** (i + 1)


def test_spin_flip():
    """Test spin flip functions"""
    d = det("2-+0+-0++--")
    assert d.spin_flip() == det("2+-0-+0--++")


if __name__ == "__main__":
    test_det_constructors()
    test_det_fill()
    test_det_getset()
    test_det_exciting()
    test_excitation_connection()
    test_det_get_occ_vir()
    test_det_creann()
    test_det_equality()
    test_det_hash()
    test_det_sorting()
    test_det_symmetry()
    test_det_slater_sign()
