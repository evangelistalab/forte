#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def test_sparse_operator():
    import forte
    import pytest
    from forte import det

    # test get/set
    sop = forte.SparseOperator()
    sop.add_term_from_str("[]")
    sop.add_term_from_str("[0a+ 0b+ 0b- 0a-]")
    to_str = sop.str()
    assert to_str == []
    to_latex = sop.latex()
    assert (
        to_latex
        == r"0.000000\; 0.000000\;\hat{a}_{0 \alpha}^\dagger\hat{a}_{0 \beta}^\dagger\hat{a}_{0 \beta}\hat{a}_{0 \alpha}"
    )

    sop = forte.SparseOperator(antihermitian=True)
    sop.add_term_from_str("[1a+ 1b+ 0b- 0a-]", 1.0)
    to_str = sop.str()
    assert to_str == ["+ [ 1a+ 1b+ 0b- 0a- ]", "- [ 0a+ 0b+ 1b- 1a- ]"]
    to_latex = sop.latex()
    assert (
        to_latex
        == r"+\;\hat{a}_{1 \alpha}^\dagger\hat{a}_{1 \beta}^\dagger\hat{a}_{0 \beta}\hat{a}_{0 \alpha} -\;\hat{a}_{0 \alpha}^\dagger\hat{a}_{0 \beta}^\dagger\hat{a}_{1 \beta}\hat{a}_{1 \alpha}"
    )

    sop = forte.SparseOperator()
    sop.add_term_from_str("[]")
    sop.add_term_from_str("[0a+ 0b+ 0b- 0a-]")

    coeff = sop.coefficients()
    assert coeff == pytest.approx([0.0, 0.0], abs=1e-9)
    sop.set_coefficient(0, 0.5)
    sop.set_coefficient(1, 0.3)
    coeff = sop.coefficients()
    assert coeff == pytest.approx([0.5, 0.3], abs=1e-9)
    # remove one term from sop
    sop.pop_term()
    # check the size
    assert sop.size() == 1
    # copy a term into a new operator
    sop2 = forte.SparseOperator()
    term = sop.term(0)
    sop2.add_term(*term)

    # test apply operator against safe implementation
    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a+ 0a-]", 0.0)
    sop.add_term_from_str("[1a+ 0a-]", 0.1)
    sop.add_term_from_str("[1b+ 0b-]", 0.3)
    sop.add_term_from_str("[1a+ 1b+ 0b- 0a-]", 0.5)
    dtest = det("20")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    wfn_safe = forte.apply_operator_safe(sop, ref)
    assert wfn == wfn_safe

    sop = forte.SparseOperator(antihermitian=True)
    sop.add_term_from_str("[2a+ 0a-]", 0.0)
    sop.add_term_from_str("[1a+ 0a-]", 0.1)
    sop.add_term_from_str("[1b+ 0b-]", 0.3)
    sop.add_term_from_str("[1a+ 1b+ 0b- 0a-]", 0.7)
    ref = forte.StateVector({det("20"): 1.0, det("02"): 0.5})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("20")] == pytest.approx(-0.35, abs=1e-9)
    assert wfn[det("02")] == pytest.approx(0.7, abs=1e-9)
    assert wfn[det("+-")] == pytest.approx(0.25, abs=1e-9)
    assert wfn[det("-+")] == pytest.approx(-0.05, abs=1e-9)

    # wfn_safe = forte.apply_operator_safe(sop,ref)
    # assert wfn == wfn_safe

    ### Operator ordering tests ###

    # test repeated operator exception
    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add_term_from_str("[0a+ 0a+]", 1.0)

    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add_term_from_str("[1b+ 1b+]", 1.0)

    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add_term_from_str("[5a- 5a-]", 1.0)

    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add_term_from_str("[3b- 3b-]", 1.0)

    # test ordering exception
    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add_term_from_str("[0a+ 0b+ 0a- 0b-]", 1.0)

    sop = forte.SparseOperator(antihermitian=True)
    sop.add_term_from_str("[]", 1.0)
    assert sop.size() == 0

    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("20")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 0a+ 0b+ 0a- 0b- |2> = -|2>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0b+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0b- 0a- |2> = -|2>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0b+ 0a+ 0b- 0a-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0a- 0b- |2> = +|2>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0b+ 0a+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    # 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    #                             = + 3a+ 4a+ 2b+ |-+000>
    #                             = + 3a+ 4a+ |-+-00>
    #                             = - 3a+ |-+-0+>
    #                             = + 3a+ |-+-++>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[3a+ 4a+ 2b+ 1b- 0a-]", 1.0)
    ref = forte.StateVector({det("22"): 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("-+-++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det(""): 1.0}))
    assert wfn[det("+")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 1a+]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det(""): 1.0}))
    assert wfn[det("++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 1a+ 2a+]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det(""): 1.0}))
    assert wfn[det("+++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 1a+ 2a+ 3a+]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det(""): 1.0}))
    assert wfn[det("++++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 1a+ 2a+ 3a+ 4a+]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a+ 0a+ 3a+ 1a+ 4a+]", 1.0, allow_reordering=True)
    wfn = forte.apply_operator(sop, forte.StateVector({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(-1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a-]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det("22222"): 1.0}))
    assert wfn[det("-2222")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[1a- 0a-]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det("22222"): 1.0}))
    assert wfn[det("--222")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det("22222"): 1.0}))
    assert wfn[det("---22")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det("22222"): 1.0}))
    assert wfn[det("----2")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[4a- 3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_operator(sop, forte.StateVector({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add_term_from_str("[1a- 3a- 2a- 4a- 0a-]", 1.0, allow_reordering=True)
    wfn = forte.apply_operator(sop, forte.StateVector({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(-1.0, abs=1e-9)

    ### Test for cases that are supposed to return zero ###
    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("00")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0b+ 0b-]", 1.0)
    dtest = det("00")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[1a+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[1b+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    ### Number operator tests ###
    # test number operator: (0a+ 0a-) |0> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("0")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |+> = |+>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("+")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0a+ 0a-) |-> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("-")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |2> = |2>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("2")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |0> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0b+ 0b-]", 1.0)
    dtest = det("0")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |+> = 0
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0b+ 0b-]", 1.0)
    dtest = det("+")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |-> = |->
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0b+ 0b-]", 1.0)
    dtest = det("-")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |2> = |2>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0b+ 0b-]", 1.0)
    ref = forte.StateVector({det("2"): 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("2")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |222> = |222>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a+ 2a-]", 1.0)
    sop.add_term_from_str("[2b+ 2b-]", 1.0)
    ref = forte.StateVector({det("222"): 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("222")] == pytest.approx(2.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22+> = |22+>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a+ 2a-]", 1.0)
    sop.add_term_from_str("[2b+ 2b-]", 1.0)
    ref = forte.StateVector({det("22+"): 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("22+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22-> = |22->
    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a+ 2a-]", 1.0)
    sop.add_term_from_str("[2b+ 2b-]", 1.0)
    ref = forte.StateVector({det("22-"): 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("22-")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |220> = |220>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[2a+ 2a-]", 1.0)
    sop.add_term_from_str("[2b+ 2b-]", 1.0)
    ref = forte.StateVector({det("220"): 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert dtest not in wfn

    ### Excitation operator tests ###
    # test excitation operator: (3a+ 0a-) |2200> = 3a+ |-200> = -|-20+>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[3a+ 0a-]", 1.0)
    dtest = det("22")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("-20+")] == pytest.approx(-1.0, abs=1e-9)

    # test excitation operator: (0a- 3a+) |22> = 0a- |220+> = |-20+>
    sop = forte.SparseOperator()
    sop.add_term_from_str("[0a- 3a+]", 1.0, allow_reordering=True)
    dtest = det("22")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("-20+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (3b+ 0b-) |2200> = 3b+ |+200> = |+20->
    sop = forte.SparseOperator()
    sop.add_term_from_str("[3b+ 0b-]", 1.0)
    dtest = det("22")
    ref = forte.StateVector({dtest: 1.0})
    wfn = forte.apply_operator(sop, ref)
    assert wfn[det("+20-")] == pytest.approx(-1.0, abs=1e-9)

    ### Adjoint tests
    # test adjoint of SqOperator is idempotent
    sqop = forte.make_sparse_operator("[0a+ 1a+ 0b+ 2b- 2a-]", 1.0)
    sqopd = sqop.adjoint()
    sqopdd = sqopd.adjoint()
    assert sqop.str() == sqopdd.str()
    # test adjoint of SqOperator
    ref = forte.make_sparse_operator("[2a+ 2b+ 0b- 1a- 0a-]", 1.0)
    assert sqopd.str() == ref.str()
    # test adjoint of SparseOperator is idempotent
    sop = forte.SparseOperator()
    sop.add_term_from_str("[3a+ 0a-]", 1.0)
    sop.add_term_from_str("[2b+ 1b-]", 1.0)
    sopd = sop.adjoint()
    sopdd = sopd.adjoint()
    assert sop.str() == sopdd.str()
    # test adjoint of SparseOperator
    ref = forte.SparseOperator()
    ref.add_term_from_str("[0a+ 3a-]", 1.0)
    ref.add_term_from_str("[1b+ 2b-]", 1.0)
    assert sopd.str() == ref.str()


def test_sparse_operator_product():
    sop1 = forte.SparseOperator()
    sop1.add_term_from_str("[1a+ 1a-]", 1.0)
    sop1.add_term_from_str("[0a+ 0a-]", 1.0)
    sop2 = forte.SparseOperator()
    sop2.add_term_from_str("[3a+ 3a-]", 1.0)
    sop2.add_term_from_str("[2a+ 2a-]", 1.0)
    sop3 = sop1 @ sop2
    sop3_test = forte.SparseOperator()
    sop3_test.add_term_from_str("[0a+ 2a+ 2a- 0a-]", 1.0)
    sop3_test.add_term_from_str("[0a+ 3a+ 3a- 0a-]", 1.0)
    sop3_test.add_term_from_str("[1a+ 2a+ 2a- 1a-]", 1.0)
    sop3_test.add_term_from_str("[1a+ 3a+ 3a- 1a-]", 1.0)
    assert sop3.str() == sop3_test.str()


if __name__ == "__main__":
    test_sparse_operator()
    test_sparse_operator_product()
