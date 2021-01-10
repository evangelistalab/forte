#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte

def parse_sign(s):
    if s == '' or s == '+':
        return 1.0
    if s == '-':
        return -1.0
    print(f'There was an error parsing the sign {s}')

def print_wfn(wfn, n):
    for d, c in wfn.map().items():
        print(f'{c:+20.12f} {d.str(n)}')

def test_sparse_operator2():
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
    sop = forte.SparseOperator()
    sop.add_term_from_str('[1a+ 0a-] - [0a+ 1a-]',0.1)
    sop.add_term_from_str('[1a+ 1b+ 0b- 0a-] - [0a+ 0b+ 1b- 1a-]',-0.3)
    sop.add_term_from_str('[1b+ 0b-] - [0b+ 1b-]',0.05)
    sop.add_term_from_str('[2a+ 2b+ 1b- 1a-] - [1a+ 1b+ 2b- 2a-]',-0.07)

    sop_fast = forte.SparseOperator()
    sop_fast.add_term_from_str('[1a+ 0a-]',0.1)
    sop_fast.add_term_from_str('[1a+ 1b+ 0b- 0a-]',-0.3)
    sop_fast.add_term_from_str('[1b+ 0b-]',0.05)
    sop_fast.add_term_from_str('[2a+ 2b+ 1b- 1a-]',-0.07)

    dtest = det("20")
    ref = forte.StateVector({ det("20"): 0.5, det("02"): 0.8660254038})
    wfn = forte.apply_exp_ah_factorized_safe(sop,ref)
    wfn_fast = forte.apply_exp_ah_factorized(sop_fast,ref)

    for d, c in wfn.map().items():
        assert c == pytest.approx(wfn_fast[d], abs=1e-9)

    assert wfn_fast[det("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn_fast[det("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn_fast[det("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn_fast[det("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn_fast[det("-+0")] == pytest.approx(0.016058887563, abs=1e-9)

    ### Test the linear operator ###
    print('Test the linear operator')
    sop = forte.SparseOperator()
    ref = forte.StateVector({ det("22"): 1.0 })
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.2)
    sop.add_term_from_str('[2a+ 0a-]',0.2)
    sop.add_term_from_str('[2b+ 0b-]',0.1)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',+0.15)
    sop.add_term_from_str('[3a+ 3b+ 1b- 1a-]',-0.21)
    sop.add_term_from_str('[1a+ 1b+ 3b- 3a-]',0.0221)
    wfn = forte.apply_operator(sop,ref)
#    print_wfn(wfn,4)
    assert det("2200") not in wfn
    assert wfn[det("+2-0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.15, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.21, abs=1e-9)

    ### Test the exponential operator ###
    print('Test the exponential operator')
    sop = forte.SparseOperator()
    ref = forte.StateVector({ det("22"): 1.0 })
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.1)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',0.15)
    sop.add_term_from_str('[3a+ 3b+ 1b- 1a-]',-0.077)
    wfn = forte.apply_exp_operator(sop,ref)
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    ### Test the exponential operator 2 ###
    print('Test the exponential operator 2')
    sop = forte.SparseOperator(antihermitian=True)
    ref = forte.StateVector({ det("22"): 1.0 })
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.1)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',0.15)
    wfn = forte.apply_exp_operator(sop,ref)
    print_wfn(wfn,4)

    assert wfn[det("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.978860446763, abs=1e-9)

    print('Test the exponential operator 2')
    sop = forte.SparseOperator(antihermitian=True)
    ref = forte.StateVector({ det("22"): 1.0 })
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.1)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',0.15)
    wfn = forte.apply_exp_operator_2(sop,ref)
    print_wfn(wfn,4)

    assert wfn[det("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.978860446763, abs=1e-9)

    ### Test the exponential operator 3 ###
    print('Test the exponential operator 3 (inverse)')
    sop = forte.SparseOperator()
    ref = forte.StateVector({ det("22"): 1.0 })
    sop.add_term_from_str('[2a+ 0a-] + [2b+ 0b-] - [0a+ 2a-] - [0b+ 2b-]',0.1)
    sop.add_term_from_str('0.5 [2a+ 2b+ 0b- 0a-] - 0.5 [0a+ 0b+ 2b- 2a-]',0.3)
    wfn = forte.apply_exp_operator(sop,ref)
    wfn2 = forte.apply_exp_operator(sop,wfn,-1.0)
    print_wfn(wfn2,4)
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[det("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("-2+0")] == pytest.approx(0.0, abs=1e-9)

    ### Test the factorized exponential operator ###
    print('Test the factorized exponential operator (safe)')
    sop = forte.SparseOperator(antihermitian=True) # we want the antihermitian version so set to True
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.2)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',0.15)
    ref = forte.StateVector({ det("22"): 1.0 })
    wfn = forte.apply_exp_ah_factorized_safe(sop,ref)
    assert wfn[det("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    print('Test the factorized exponential operator (fast)')
    sop = forte.SparseOperator(antihermitian=True)
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.2)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',0.15)
    ref = forte.StateVector({ det("22"): 1.0 })
    wfn = forte.apply_exp_ah_factorized(sop,ref)
    print_wfn(wfn,4)
    assert wfn[det("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    print('Test the factorized exponential operator (fast)')
    sop = forte.GeneralOperator()
    sop.add_term_from_str('[2a+ 0a-] - [0a+ 2a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-] - [0b+ 2b-]',0.2)
    sop.add_term_from_str('0.5 [2a+ 2b+ 0b- 0a-] - 0.5 [0a+ 0b+ 2b- 2a-]',0.3)
    ref = forte.StateVector({ det("22"): 1.0 })
    wfn = forte.apply_exp_ah_factorized(sop,ref)
    print_wfn(wfn,4)

    print('Test the factorized exponential operator (fast) inverse')
    sop = forte.SparseOperator(antihermitian=True)
    sop.add_term_from_str('[2a+ 0a-]',0.1)
    sop.add_term_from_str('[2b+ 0b-]',0.2)
    sop.add_term_from_str('[2a+ 2b+ 0b- 0a-]',0.15)
    ref = forte.StateVector({ det("22"): 1.0 })
    wfn = forte.apply_exp_ah_factorized(sop,ref)
    wfn2 = forte.apply_exp_ah_factorized(sop,wfn,inverse=True)
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    print_wfn(wfn2,4)


test_sparse_operator2()





