#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte
import pytest

def make_str(s):
    return forte.str(s)

def test_str_constructors():
    """Test class constructors"""
    print("Testing string interface")
    s1 = make_str("1100")
    assert str(s1) == f"|1100{'0' * (forte.Determinant.norb() - 4)}>"
    assert repr(s1) == f"|1100{'0' * (forte.Determinant.norb() - 4)}>"    

def test_str_address():
    """Test string address"""
    print("Testing string address")
    nmo = 4
    ne = 2
    strings = [["1100","0011","1111"],[],[],["1010","0101","0001","0010"]]
    strings = [[make_str(s) for s in strings_irrep] for strings_irrep in strings]
    str_add = forte.StringAddress(nmo,ne,strings)
    test_strings = [make_str(s) for s in ["1100","0011","1111","1010","0101","0001","0010"]]
    address = [0, 1, 2, 0, 1, 2, 3]
    symmetry = [0, 0, 0, 3, 3, 3, 3]
    for s,a in zip(test_strings,address):
        assert str_add.add(s) == a
    for s,a in zip(test_strings,symmetry):
        assert str_add.sym(s) == a        
    for h, nh in zip([0,3],[3,4]):
        assert str_add.strpi(h) == nh

if __name__ == "__main__":
    test_str_constructors()
    test_str_address()
