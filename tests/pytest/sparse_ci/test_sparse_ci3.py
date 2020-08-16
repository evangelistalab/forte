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

def test_sparse_ci3():
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
    gop.add_operator(to_ops('[0a+ 0b+ 0b- 0a-]'),1.0)
    dtest = det("20")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 0a+ 0b+ 0a- 0b- |2> = -|2>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0b+ 0a- 0b-]'),1.0)
    dtest = det("20")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0b- 0a- |2> = -|2>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0b+ 0a+ 0b- 0a-]'),1.0)
    dtest = det("20")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0a- 0b- |2> = +|2>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0b+ 0a+ 0a- 0b-]'),1.0)
    dtest = det("20")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    # 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    #                             = + 3a+ 4a+ 2b+ |-+000>
    #                             = + 3a+ 4a+ |-+-00>
    #                             = - 3a+ |-+-0+>
    #                             = + 3a+ |-+-++>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[3a+ 4a+ 2b+ 1b- 0a-]'),1.0)
    ref = { det("22"): 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("-+-++")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+]'),1.0)
    wfn = forte.apply_operator(gop,{ det(""): 1.0})
    assert wfn[det("+")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 1a+]'),1.0)
    wfn = forte.apply_operator(gop,{ det(""): 1.0})
    assert wfn[det("++")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 1a+ 2a+]'),1.0)
    wfn = forte.apply_operator(gop,{ det(""): 1.0})
    assert wfn[det("+++")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 1a+ 2a+ 3a+]'),1.0)
    wfn = forte.apply_operator(gop,{ det(""): 1.0})
    assert wfn[det("++++")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 1a+ 2a+ 3a+ 4a+]'),1.0)
    wfn = forte.apply_operator(gop,{ det(""): 1.0})
    assert wfn[det("+++++")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 0a+ 3a+ 1a+ 4a+]'),1.0)
    wfn = forte.apply_operator(gop,{ det(""): 1.0})
    assert wfn[det("+++++")] == pytest.approx(-1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a-]'),1.0)
    wfn = forte.apply_operator(gop,{ det("22222"): 1.0})
    assert wfn[det("-2222")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[1a- 0a-]'),1.0)
    wfn = forte.apply_operator(gop,{ det("22222"): 1.0})
    assert wfn[det("--222")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a- 1a- 0a-]'),1.0)
    wfn = forte.apply_operator(gop,{ det("22222"): 1.0})
    assert wfn[det("---22")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[3a- 2a- 1a- 0a-]'),1.0)
    wfn = forte.apply_operator(gop,{ det("22222"): 1.0})
    assert wfn[det("----2")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[4a- 3a- 2a- 1a- 0a-]'),1.0)
    wfn = forte.apply_operator(gop,{ det("22222"): 1.0})
    assert wfn[det("-----")] == pytest.approx(1.0, abs=1e-9)

    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[1a- 3a- 2a- 4a- 0a-]'),1.0)
    wfn = forte.apply_operator(gop,{ det("22222"): 1.0})
    assert wfn[det("-----")] == pytest.approx(-1.0, abs=1e-9)


    ### Test for cases that are supposed to return zero ###
    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0b+ 0b- 0a-]'),1.0)
    dtest = det("00")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0b+ 0b-]'),1.0)
    dtest = det("00")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[1a+ 0a+ 0a-]'),1.0)
    dtest = det("+")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[1b+ 0a+ 0a-]'),1.0)
    dtest = det("+")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn


    ### Number operator tests ###
    # test number operator: (0a+ 0a-) |0> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0a-]'),1.0)
    dtest = det("0")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |+> = |+>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0a-]'),1.0)
    dtest = det("+")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0a+ 0a-) |-> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0a-]'),1.0)
    dtest = det("-")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |2> = |2>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a+ 0a-]'),1.0)
    dtest = det("2")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |0> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0b+ 0b-]'),1.0)
    dtest = det("0")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |+> = 0
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0b+ 0b-]'),1.0)
    dtest = det("+")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |-> = |->
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0b+ 0b-]'),1.0)
    dtest = det("-")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |2> = |2>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0b+ 0b-]'),1.0)
    ref = { det("2"): 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("2")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |222> = |222>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 2a-] + [2b+ 2b-]'),1.0)
    ref = { det("222"): 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("222")] == pytest.approx(2.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22+> = |22+>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 2a-] + [2b+ 2b-]'),1.0)
    ref = { det("22+"): 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("22+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22-> = |22->
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 2a-] + [2b+ 2b-]'),1.0)
    ref = { det("22-"): 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("22-")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |220> = |220>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[2a+ 2a-] + [2b+ 2b-]'),1.0)
    ref = { det("220"): 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert dtest not in wfn


    ### Excitation operator tests ###
    # test excitation operator: (3a+ 0a-) |2200> = 3a+ |-200> = -|-20+>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[3a+ 0a-]'),1.0)
    dtest = det("22")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("-20+")] == pytest.approx(-1.0, abs=1e-9)

    # test excitation operator: (0a- 3a+) |22> = 0a- |220+> = |-20+>
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[0a- 3a+]'),1.0)
    dtest = det("22")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("-20+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (3b+ 0b-) |2200> = 3b+ |+200> = |+20->
    gop = forte.GeneralOperator()
    gop.add_operator(to_ops('[3b+ 0b-]'),1.0)
    dtest = det("22")
    ref = { dtest: 1.0}
    wfn = forte.apply_operator(gop,ref)
    assert wfn[det("+20-")] == pytest.approx(-1.0, abs=1e-9)

test_sparse_ci3()





