#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte
import pytest


def test_gas_occupation1():
    """Test class constructors"""
    print("Testing determinant interface")
    na = 2
    nb = 2
    gas_size = [2, 2]
    gas_min = [1]
    gas_max = [1]
    n, occsa, occsb, pair_occs = forte.get_gas_occupation(na, nb, gas_min, gas_max, gas_size)
    assert n == 2
    assert occsa == [[1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]
    assert occsb == [[0, 2, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
    assert pair_occs == [(0, 0), (1, 1)]
    ref_occs = []
    for i, j in pair_occs:
        ref_occs.append(occsa[i] + occsb[j])
    ref_occs.sort()

    n, occsa, occsb, pair_occs = forte.get_ci_occupation_patterns(na, nb, gas_min, gas_max, gas_size)
    occs = []
    for i, j in pair_occs:
        occs.append(occsa[i] + occsb[j])
    occs.sort()
    assert occs == ref_occs


def test_gas_occupation2():
    """Test class constructors"""
    print("Testing determinant interface")
    na = 2
    nb = 2
    gas_size = [2, 2]
    gas_min = [0]
    gas_max = [0]
    n, occsa, occsb, pair_occs = forte.get_gas_occupation(na, nb, gas_min, gas_max, gas_size)
    assert n == 2
    assert occsa == [[0, 2, 0, 0, 0, 0]]
    assert occsb == [[0, 2, 0, 0, 0, 0]]
    assert pair_occs == [(0, 0)]
    ref_occs = []
    for i, j in pair_occs:
        ref_occs.append(occsa[i] + occsb[j])
    ref_occs.sort()

    n, occsa, occsb, pair_occs = forte.get_ci_occupation_patterns(na, nb, gas_min, gas_max, gas_size)
    occs = []
    for i, j in pair_occs:
        occs.append(occsa[i] + occsb[j])
    occs.sort()
    assert occs == ref_occs


def test_gas_occupation3():
    """Test class constructors"""
    print("Testing determinant interface")
    na = 2
    nb = 2
    gas_size = [2, 2]
    gas_min = [0]
    gas_max = [1]
    n, occsa, occsb, pair_occs = forte.get_gas_occupation(na, nb, gas_min, gas_max, gas_size)
    assert n == 2
    assert occsa == [[1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]
    assert occsb == [[0, 2, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
    assert pair_occs == [(0, 0), (1, 1), (1, 0)]
    ref_occs = []
    for i, j in pair_occs:
        ref_occs.append(occsa[i] + occsb[j])
    ref_occs.sort()

    n, occsa, occsb, pair_occs = forte.get_ci_occupation_patterns(na, nb, gas_min, gas_max, gas_size)
    occs = []
    for i, j in pair_occs:
        occs.append(occsa[i] + occsb[j])
    occs.sort()
    assert occs == ref_occs


if __name__ == "__main__":
    test_gas_occupation1()
    test_gas_occupation2()
    test_gas_occupation3()
