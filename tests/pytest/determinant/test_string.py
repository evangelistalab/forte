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
    str_add = forte.StringAddress([2,2])
    strings = [str("1100"),str("1010"),str("0011"),str("1111"),str("0101"),str("0001"),str("0010")]
    addresses = [0, 0, 1, 2, 1, 2, 3]
    for s,a in zip(strings,addresses):
        assert str_add.rel_add(s) == a
    for s,a in zip(strings,addresses):
        assert str_add.rel_add(s) == a

    for h, nh in zip([0,1],[3,4]):
        assert str_add.strpi(h) == nh

if __name__ == "__main__":
    test_str_constructors()
    test_str_address()