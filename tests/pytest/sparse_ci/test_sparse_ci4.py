#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def test_sparse_ci4():
    import pytest
    from forte import det

    ### Test the linear operator ###
    op = forte.SparseOperator()
    ref = forte.StateVector({det("22"): 1.0})
    op.add_term_from_str('[2a+ 0a-]', 0.1)
    op.add_term_from_str('[2b+ 0b-]', 0.2)
    op.add_term_from_str('[2a+ 0a-]', 0.2)
    op.add_term_from_str('[2b+ 0b-]', 0.1)
    op.add_term_from_str('[2a+ 2b+ 0b- 0a-]', 0.15)
    op.add_term_from_str('[3a+ 3b+ 1b- 1a-]', -0.21)
    op.add_term_from_str('[1a+ 1b+ 3b- 3a-]', 0.13 * 0.17)

    wfn = forte.apply_operator(op, ref)
    assert det("2200") not in wfn
    assert wfn[det("+2-0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.15, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.21, abs=1e-9)

    ### Test the exponential operator with excitation operator ###
    op = forte.SparseOperator()
    ref = forte.StateVector({det("22"): 1.0})
    op.add_term_from_str('[2a+ 0a-]', 0.1)
    op.add_term_from_str('[2b+ 0b-]', 0.1)
    op.add_term_from_str('[2a+ 2b+ 0b- 0a-]', 0.15)
    op.add_term_from_str('[3a+ 3b+ 1b- 1a-]', -0.077)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref)
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    wfn = exp.compute(op, ref, algorithm='onthefly')
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    wfn = exp.compute(op, ref, algorithm='ontheflystd')
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    ### Test the exponential operator with antihermitian operator ###
    op = forte.SparseOperator(antihermitian=True)
    ref = forte.StateVector({det("22"): 1.0})
    op.add_term_from_str('[2a+ 0a-]', 0.1)
    op.add_term_from_str('[2b+ 0b-]', 0.1)
    op.add_term_from_str('[2a+ 2b+ 0b- 0a-]', 0.15)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref)
    assert wfn[det("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.978860446763, abs=1e-9)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref, algorithm='onthefly')
    assert wfn[det("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.978860446763, abs=1e-9)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref, algorithm='ontheflystd')
    assert wfn[det("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.978860446763, abs=1e-9)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref)
    wfn2 = exp.compute(op, wfn, scaling_factor=-1.0)
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[det("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("-2+0")] == pytest.approx(0.0, abs=1e-9)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref, algorithm='onthefly')
    wfn2 = exp.compute(op, wfn, scaling_factor=-1.0, algorithm='onthefly')
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[det("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("-2+0")] == pytest.approx(0.0, abs=1e-9)

    exp = forte.SparseExp()
    wfn = exp.compute(op, ref, algorithm='ontheflystd')
    wfn2 = exp.compute(op, wfn, scaling_factor=-1.0, algorithm='ontheflystd')
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[det("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("-2+0")] == pytest.approx(0.0, abs=1e-9)

    ### Test the factorized exponential operator ###
    op = forte.SparseOperator(antihermitian=True)
    op.add_term_from_str('[2a+ 0a-]', 0.1)
    op.add_term_from_str('[2b+ 0b-]', 0.2)
    op.add_term_from_str('[2a+ 2b+ 0b- 0a-]', 0.15)
    ref = forte.StateVector({det("22"): 1.0})

    factexp = forte.SparseFactExp()
    wfn = factexp.compute(op, ref)

    assert wfn[det("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    wfn2 = factexp.compute(op, wfn, inverse=True)

    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)

    factexp = forte.SparseFactExp()
    wfn = factexp.compute(op, ref, algorithm='onthefly')

    assert wfn[det("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    wfn2 = factexp.compute(op, wfn, inverse=True, algorithm='onthefly')

    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator(antihermitian=True)
    op.add_term_from_str('[1a+ 0a-]', 0.1)
    op.add_term_from_str('[1a+ 1b+ 0b- 0a-]', -0.3)
    op.add_term_from_str('[1b+ 0b-]', 0.05)
    op.add_term_from_str('[2a+ 2b+ 1b- 1a-]', -0.07)

    dtest = det("20")
    ref = forte.StateVector({det("20"): 0.5, det("02"): 0.8660254038})
    factexp = forte.SparseFactExp()
    wfn = factexp.compute(op, ref)

    assert wfn[det("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn[det("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn[det("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn[det("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn[det("-+0")] == pytest.approx(0.016058887563, abs=1e-9)

    wfn = factexp.compute(op, ref, algorithm='onthefly')

    assert wfn[det("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn[det("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn[det("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn[det("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn[det("-+0")] == pytest.approx(0.016058887563, abs=1e-9)


test_sparse_ci4()
