#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte

def test_ambit():
    """Test the conversion of ambit::Tensor to numpy"""
    forte.startup()
    t = forte.test_ambit()
    n = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert t[i][j][k] == n
                n += 1
    forte.cleanup()
