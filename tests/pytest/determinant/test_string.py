#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte
import pytest

def str(s):
    return forte.str(s)

def test_str_constructors():
    """Test class constructors"""
    print("Testing string interface")
    s1 = str("1100")

def test_str_address():
    """Test string address"""
    print("Testing string address")
    strings = [[str("1100"),str("0011"),str("1111")],[],[],[str("1010"),str("0101"),str("0001"),str("0010")]]
    str_add = forte.StringAddress(strings)
    test_strings = [str("1100"),str("0011"),str("1111"),str("1010"),str("0101"),str("0001"),str("0010")]
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