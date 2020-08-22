#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte

def det(s):
    d = forte.Determinant();
    for k, c in enumerate(s):
        if c == '+':
            d.create_alfa_bit(k)
        elif c == '-':
            d.create_beta_bit(k)
        elif c == '2':
            d.create_alfa_bit(k)
            d.create_beta_bit(k)
    return d

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
#    print(f'Converting the string {str} to an operator')
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
#            print(ops)
            terms.append((sign * factor,ops))
    return terms

def print_wfn(wfn, n):
    for d, c in wfn.items():
        print(f'{c} {d.str(n)}')

def test_sparse_ci4():
    import math
    import psi4
    import forte
    import itertools
    import numpy as np
    import pytest
    from forte import forte_options

    ### Operator ordering tests ###
    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    gop = forte.GeneralOperator()
#    gop.add_operator(to_ops('[1a+ 0a-] - [0a+ 1a-]'),0.1)
#    gop.add_operator(to_ops('[1a+ 1b+ 0b- 0a-] - [0a+ 0b+ 1b- 1a-]'),0.1)


    gop.add_operator(to_ops('[0a+ 0a-]'),0.1)
    gop.add_operator(to_ops('[0a+ 1a-]'),0.1)
    gop.add_operator(to_ops('[1a+ 0a-]'),0.1)
    gop.add_operator(to_ops('[1a+ 1a-]'),0.1)
    gop.add_operator(to_ops('[2a+ 2a-]'),0.1)
    gop.add_operator(to_ops('[2a+ 1a-]'),0.1)
    gop.add_operator(to_ops('[1a+ 2a-]'),0.1)

    dtest = det("+0")
    ref = { dtest: 1.0}
    wfn = forte.apply_exp_ah_factorized_fast(gop,ref)
    norm = 0.0
    for d, c in wfn.items():
        norm += c**2
    print(norm)
    print_wfn(wfn,2)
#    assert norm == pytest.approx(1.0, abs=1e-9)

test_sparse_ci4()





