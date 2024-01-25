#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte
import pytest


def det(s):
    return forte.det(s)

def test_configuration_constructors():
    """Test class constructors"""
    print("Testing determinant interface")
    d1 = det("22+-0-+")
    c1 = forte.Configuration(d1)
    assert c1.str(7) == '|2211011>'
    # test the counting functions
    assert c1.count_docc() == 2
    assert c1.count_socc() == 4
    # test functions that extract lists
    assert c1.get_docc_vec() == [0,1]
    assert c1.get_socc_vec() == [2,3,5,6]

def test_configurationt_getset():
    """Test get/set operations"""
    print("Testing configuration interface")
    c = forte.Configuration()
    assert c.is_empt(0) == True
    assert c.is_docc(0) == False
    assert c.is_docc(7) == False
    assert c.is_socc(0) == False
    assert c.is_socc(7) == False
    c.set_occ(7, 2)
    assert c.is_docc(7) == True
    assert c.is_empt(7) == False
    assert c.is_socc(7) == False
    c.set_occ(7, 1)
    assert c.is_socc(7) == True
    assert c.is_docc(7) == False
    c.set_occ(7, 0)
    assert c.is_docc(7) == False
    assert c.is_socc(7) == False
    assert c.is_empt(7) == True

def test_configurationt_ops():
    """Test class operators"""
    print("Testing configuration operators")
    d1 = det("22+-0-+")
    d2 = det("22+--+0")
    d3 = det("22+-0+-")
    d4 = det("2+-0+-2")
    c1 = forte.Configuration(d1)
    c2 = forte.Configuration(d2)
    c3 = forte.Configuration(d3)
    c4 = forte.Configuration(d4)
    assert c2 != c1
    assert c3 == c1
    assert c1.__hash__() == c3.__hash__()
    assert c1 > c2
    assert c2 < c1
    assert c3 > c2
    assert c2 < c3
    assert c4 < c1
    assert c4 < c2
    assert c4 < c3

if __name__ == "__main__":
    test_configuration_constructors()
    test_configurationt_getset()
    test_configurationt_ops()