#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte

def test_ambit():
    """Test the conversion of ambit::Tensor to numpy"""
    forte.startup()
    t = forte.test_ambit_3d()
    n = 0
    shape = t.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                assert t[i][j][k] == n
                n += 1
    forte.cleanup()
