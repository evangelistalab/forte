#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def print_wfn(wfn, n):
    for d, c in wfn.items():
        print(f"{c} {d.str(n)}")


def test_sparse_ci3():
    import forte
    import pytest
    from forte import det

    ### Operator ordering tests ###
    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 0a+ 0b+ 0a- 0b- |2> = -|2>
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0b+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0b- 0a- |2> = -|2>
    op = forte.SparseOperator()
    op.add_term_from_str("[0b+ 0a+ 0b- 0a-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0a- 0b- |2> = +|2>
    op = forte.SparseOperator()
    op.add_term_from_str("[0b+ 0a+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    # 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    #                             = + 3a+ 4a+ 2b+ |-+000>
    #                             = + 3a+ 4a+ |-+-00>
    #                             = - 3a+ |-+-0+>
    #                             = + 3a+ |-+-++>
    op = forte.SparseOperator()
    op.add_term_from_str("[3a+ 4a+ 2b+ 1b- 0a-]", 1.0)
    ref = forte.SparseState({det("22"): 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[det("-+-++")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[0a+]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 1a+]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det(""): 1.0}))
    assert wfn[det("++")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 1a+ 2a+]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+++")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 1a+ 2a+ 3a+]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det(""): 1.0}))
    assert wfn[det("++++")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 1a+ 2a+ 3a+ 4a+]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[2a+ 0a+ 3a+ 1a+ 4a+]", 1.0, allow_reordering=True)
    wfn = forte.apply_op(op, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(-1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[0a-]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("-2222")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[1a- 0a-]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("--222")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("---22")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("----2")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[4a- 3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_op(op, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperator()
    op.add_term_from_str("[1a- 3a- 2a- 4a- 0a-]", 1.0, allow_reordering=True)
    wfn = forte.apply_op(op, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(-1.0, abs=1e-9)

    ### Test for cases that are supposed to return zero ###
    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("00")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0b+ 0b-]", 1.0)
    dtest = det("00")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[1a+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[1b+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    ### Number operator tests ###
    # test number operator: (0a+ 0a-) |0> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("0")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |+> = |+>
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0a+ 0a-) |-> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("-")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |2> = |2>
    op = forte.SparseOperator()
    op.add_term_from_str("[0a+ 0a-]", 1.0)
    dtest = det("2")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |0> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[0b+ 0b-]", 1.0)
    dtest = det("0")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |+> = 0
    op = forte.SparseOperator()
    op.add_term_from_str("[0b+ 0b-]", 1.0)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |-> = |->
    op = forte.SparseOperator()
    op.add_term_from_str("[0b+ 0b-]", 1.0)
    dtest = det("-")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |2> = |2>
    op = forte.SparseOperator()
    op.add_term_from_str("[0b+ 0b-]", 1.0)
    ref = forte.SparseState({det("2"): 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[det("2")] == pytest.approx(1.0, abs=1e-9)

    # make sure that the  number operator throws an exception: (2a+ 2a- + 2b+ 2b-) |222> = |222>
    op = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        op.add_term_from_str("[2a+ 2a-] + [2b+ 2b-]", 1.0)

    ### Excitation operator tests ###
    # test excitation operator: (3a+ 0a-) |2200> = 3a+ |-200> = -|-20+>
    op = forte.SparseOperator()
    op.add_term_from_str("[3a+ 0a-]", 1.0)
    dtest = det("22")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[det("-20+")] == pytest.approx(-1.0, abs=1e-9)

    # test excitation operator: (0a- 3a+) |22> = 0a- |220+> = |-20+>
    op = forte.SparseOperator()
    op.add_term_from_str("[0a- 3a+]", 1.0, allow_reordering=True)
    dtest = det("22")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[det("-20+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (3b+ 0b-) |2200> = 3b+ |+200> = |+20->
    op = forte.SparseOperator()
    op.add_term_from_str("[3b+ 0b-]", 1.0)
    dtest = det("22")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(op, ref)
    assert wfn[det("+20-")] == pytest.approx(-1.0, abs=1e-9)


if __name__ == "__main__":
    test_sparse_ci3()
