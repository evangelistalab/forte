#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte

def parse_sign(s):
    if s == '' or s == '+':
        return 1.0
    if s == '-':
        return -1.0
    print(f'There was an error parsing the sign {s}')

def parse_factor(s):
    if s == '':
        return 1.0
    return(float(s))

def parse_ops(s):
    ops = []
    # we reverse the operator order
    for op in s[1:-1].split(' ')[::-1]:
        creation = True if op[-1] == '+' else False
        alpha = True if op[-2] == 'a' else False
        orb = int(op[0:-2])
        ops.append((creation,alpha,orb))
    return ops

def to_ops(str):
    terms = []
    # '<something>[1b+ 0b+] +-<something>[1b+ 0b+]'
    import re
    match_op = r'\s?([\+\-])?\s*(\d*\.?\d*)?\s*\*?\s*(\[[0-9ab\+\-\s]*\])'
    m = re.findall(match_op,str)
    if m:
        for group in m:
            sign = parse_sign(group[0])
            factor = parse_factor(group[1])
            ops = parse_ops(group[2])
            terms.append((sign * factor,ops))
    return terms

def print_wfn(wfn, n):
    for d, c in wfn.items():
        print(f'{c:+20.12f} {d.str(n)}')

def test_sparse_ci4():
    import math
    import psi4
    import forte
    import itertools
    import numpy as np
    import pytest
    from forte import forte_options
    from forte import det

    ### Operator ordering tests ###
    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[1a+ 0a-] - [0a+ 1a-]'),0.1)
    gop.add_operator(to_ops('[1a+ 1b+ 0b- 0a-] - [0a+ 0b+ 1b- 1a-]'),-0.3)
    gop.add_operator(to_ops('[1b+ 0b-] - [0b+ 1b-]'),0.05)
    gop.add_operator(to_ops('[2a+ 2b+ 1b- 1a-] - [1a+ 1b+ 2b- 2a-]'),-0.07)

    gop_fast = forte.GeneralOperator()
    gop_fast.add_operator(to_ops('[1a+ 0a-]'),0.1)
    gop_fast.add_operator(to_ops('[1a+ 1b+ 0b- 0a-]'),-0.3)
    gop_fast.add_operator(to_ops('[1b+ 0b-]'),0.05)
    gop_fast.add_operator(to_ops('[2a+ 2b+ 1b- 1a-]'),-0.07)

    dtest = det("20")
    ref = { det("20"): 0.5, det("02"): 0.8660254038}
    wfn = forte.apply_exp_ah_factorized(gop,ref)
    wfn_fast = forte.apply_exp_ah_factorized_fast(gop_fast,ref)

    for d, c in wfn.items():
        assert c == pytest.approx(wfn_fast[d], abs=1e-9)

    assert wfn_fast[det("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn_fast[det("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn_fast[det("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn_fast[det("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn_fast[det("-+0")] == pytest.approx(0.016058887563, abs=1e-9)

    ### Test the linear operator ###
    print('Test the linear operator')
    gop = forte.GeneralOperator()
    ref = { det("22"): 1.0 } #0.5, det("02"): 0.8660254038}
    gop.add_operator(to_ops('[2a+ 0a-] + 2.0 [2b+ 0b-]'),0.1)
    gop.add_operator(to_ops('2.0 [2a+ 0a-] + [2b+ 0b-]'),0.1)
    gop.add_operator(to_ops('0.5 [2a+ 2b+ 0b- 0a-] - 0.7 [3a+ 3b+ 1b- 1a-]'),0.3)
    gop.add_operator(to_ops('+0.17 [1a+ 1b+ 3b- 3a-]'),0.13)
    wfn = forte.apply_operator_fast(gop,ref)
    print_wfn(wfn,4)
    assert det("2200") not in wfn
    assert wfn[det("+2-0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.15, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.21, abs=1e-9)

    ### Test the exponential operator ###
    print('Test the exponential operator')
    gop = forte.GeneralOperator()
    ref = { det("22"): 1.0 } #0.5, det("02"): 0.8660254038}
    gop.add_operator(to_ops('[2a+ 0a-] + [2b+ 0b-]'),0.1)
    gop.add_operator(to_ops('0.5 [2a+ 2b+ 0b- 0a-]'),0.3)
    gop.add_operator(to_ops('-0.7 [3a+ 3b+ 1b- 1a-]'),0.11)
    wfn = forte.apply_exp_operator_fast(gop,ref)
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    print_wfn(wfn,4)

    ### Test the exponential operator 2 ###
    print('Test the exponential operator 2')
    gop = forte.GeneralOperator()
    ref = { det("22"): 1.0 } #0.5, det("02"): 0.8660254038}
    gop.add_operator(to_ops('[2a+ 0a-] + [2b+ 0b-] - [0a+ 2a-] - [0b+ 2b-]'),0.1)
    gop.add_operator(to_ops('0.5 [2a+ 2b+ 0b- 0a-] - 0.5 [0a+ 0b+ 2b- 2a-]'),0.3)
    wfn = forte.apply_exp_operator_fast(gop,ref)
    print_wfn(wfn,4)

    ### Test the exponential operator 3 ###
    print('Test the exponential operator 3')
    gop = forte.GeneralOperator()
    ref = { det("22"): 1.0 } #0.5, det("02"): 0.8660254038}
    gop.add_operator(to_ops('[2a+ 0a-] + [2b+ 0b-] - [0a+ 2a-] - [0b+ 2b-]'),0.1)
    gop.add_operator(to_ops('0.5 [2a+ 2b+ 0b- 0a-] - 0.5 [0a+ 0b+ 2b- 2a-]'),0.3)
    wfn = forte.apply_exp_operator_fast(gop,ref)
    wfn2 = forte.apply_exp_operator_fast(gop,wfn,-1.0)
    print_wfn(wfn2,4)
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[det("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("-2+0")] == pytest.approx(0.0, abs=1e-9)

    ### Test the factorized exponential operator ###
    print('Test the factorized exponential operator (slow)')
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 0a-] - [0a+ 2a-]'),0.1)
    gop.add_operator(to_ops('[2b+ 0b-] - [0b+ 2b-]'),0.2)
    gop.add_operator(to_ops('0.5 [2a+ 2b+ 0b- 0a-] - 0.5 [0a+ 0b+ 2b- 2a-]'),0.3)
    ref = { det("22"): 1.0 }
    wfn = forte.apply_exp_ah_factorized(gop,ref)
    print_wfn(wfn,4)

    print('Test the factorized exponential operator (fast)')
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 0a-] - [0a+ 2a-]'),0.1)
    gop.add_operator(to_ops('[2b+ 0b-] - [0b+ 2b-]'),0.2)
    gop.add_operator(to_ops('0.5 [2a+ 2b+ 0b- 0a-] - 0.5 [0a+ 0b+ 2b- 2a-]'),0.3)
    ref = { det("22"): 1.0 }
    wfn = forte.apply_exp_ah_factorized_fast(gop,ref)
    print_wfn(wfn,4)

    print('Test the factorized exponential operator (fast) inverse')
    wfn2 = forte.apply_exp_ah_factorized_fast(gop,wfn,True)
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    print_wfn(wfn2,4)

    ops, l = gop.get_operator(0)
    gop2 = forte.GeneralOperator()
    gop2.add_operator2(ops,l)
    print(gop2.str())

test_sparse_ci4()





