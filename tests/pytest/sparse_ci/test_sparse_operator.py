import forte
import pytest
from forte import det


def test_sparse_operator_creation():
    sop = forte.SparseOperator()
    sop.add([1], [], [1], [])
    assert sop["[1a+ 1a-]"] == 1.0

    # verify that order is ignored when adding terms (1,3 and 3,1 are the same)
    sop = forte.SparseOperator()
    sop.add([3, 1], [], [], [])
    assert sop["[1a+ 3a+]"] == 1.0

    sop = forte.SparseOperator()
    sop.add([1, 3], [], [], [])
    assert sop["[1a+ 3a+]"] == 1.0

    sop = forte.SparseOperatorList()
    sop.add([1], [], [1], [])
    assert sop["[1a+ 1a-]"] == 1.0

    # verify that order is ignored when adding terms (1,3 and 3,1 are the same)
    sop = forte.SparseOperatorList()
    sop.add([3, 1], [], [], [])
    assert sop["[1a+ 3a+]"] == 1.0

    sop = forte.SparseOperatorList()
    sop.add([1, 3], [], [], [])
    assert sop["[1a+ 3a+]"] == 1.0


def test_sparse_operator():
    # test get/set
    sop = forte.SparseOperator()
    sop.add("[]", 2.3)
    sop.add("[0a+ 0b+ 0b- 0a-]")
    to_str = sorted(sop.str())
    assert to_str == sorted(["(2.3 + 0i) * []", "(1 + 0i) * [0a+ 0b+ 0b- 0a-]"])
    to_latex = sop.latex()
    assert (
        to_latex
        == r"(2.300000 + 0.000000 i)\; + \;\hat{a}_{0 \alpha}^\dagger\hat{a}_{0 \beta}^\dagger\hat{a}_{0 \beta}\hat{a}_{0 \alpha}"
    )

    sop = forte.SparseOperator()
    sop.add("[1a+ 1b+ 0b- 0a-]", 1.0)
    to_str = sop.str()
    assert to_str == ["(1 + 0i) * [1a+ 1b+ 0b- 0a-]"]
    to_latex = sop.latex()
    assert to_latex == r"\;\hat{a}_{1 \alpha}^\dagger\hat{a}_{1 \beta}^\dagger\hat{a}_{0 \beta}\hat{a}_{0 \alpha}"

    sop = forte.SparseOperator()
    sop.add("[]", 1.0)
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)

    assert sop.coefficient("[]") == 1.0
    assert sop.coefficient("[0a+ 0b+ 0b- 0a-]") == 1.0
    sop.set_coefficient("[]", 0.5)
    sop.set_coefficient("[0a+ 0b+ 0b- 0a-]", 0.3)
    assert sop.coefficient("[]") == 0.5
    assert sop.coefficient("[0a+ 0b+ 0b- 0a-]") == 0.3
    # remove one term from sop
    sop.remove("[0a+ 0b+ 0b- 0a-]")
    # check the size
    assert len(sop) == 1
    # check the element
    assert sop.coefficient("[]") == 0.5
    assert sop.coefficient("[0a+ 0b+ 0b- 0a-]") == 0.0
    assert len(sop) == 1
    sop.remove("[]")
    assert len(sop) == 0

    # copy a term into a new operator
    sop = forte.SparseOperator()
    sop.add("[]", 1.0)
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)

    # test apply operator against safe implementation
    sop = forte.SparseOperator()
    sop.add("[2a+ 0a-]", 0.0)
    sop.add("[1a+ 0a-]", 0.1)
    sop.add("[1b+ 0b-]", 0.3)
    sop.add("[1a+ 1b+ 0b- 0a-]", 0.5)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = sop @ ref
    assert wfn[det("20")] == pytest.approx(0.0, abs=1e-9)
    assert wfn[det("-+")] == pytest.approx(0.1, abs=1e-9)
    assert wfn[det("+-")] == pytest.approx(0.3, abs=1e-9)
    assert wfn[det("02")] == pytest.approx(0.5, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[2a+ 0a-]", 0.0)
    sop.add("[1a+ 0a-]", 0.1)
    sop.add("[1b+ 0b-]", 0.3)
    sop.add("[1a+ 1b+ 0b- 0a-]", 0.7)
    ref = forte.SparseState({det("20"): 1.0, det("02"): 0.5})
    wfn = forte.apply_antiherm(sop, ref)
    assert wfn[det("20")] == pytest.approx(-0.35, abs=1e-9)
    assert wfn[det("02")] == pytest.approx(0.7, abs=1e-9)
    assert wfn[det("+-")] == pytest.approx(0.25, abs=1e-9)
    assert wfn[det("-+")] == pytest.approx(-0.05, abs=1e-9)

    ### Operator ordering tests ###

    # test repeated operator exception
    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[0a+ 0a+]", 1.0)

    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[1b+ 1b+]", 1.0)

    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[5a- 5a-]", 1.0)

    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[3b- 3b-]", 1.0)

    # test ordering exception
    sop = forte.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[0a+ 0b+ 0a- 0b-]", 1.0)

    sop = forte.SparseOperator()
    sop.add("[]", 1.0)
    assert len(sop) == 1

    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    sop = forte.SparseOperator()
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 0a+ 0b+ 0a- 0b- |2> = -|2>
    sop = forte.SparseOperator()
    sop.add("[0a+ 0b+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0b- 0a- |2> = -|2>
    sop = forte.SparseOperator()
    sop.add("[0b+ 0a+ 0b- 0a-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0a- 0b- |2> = +|2>
    sop = forte.SparseOperator()
    sop.add("[0b+ 0a+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    # 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    #                             = + 3a+ 4a+ 2b+ |-+000>
    #                             = + 3a+ 4a+ |-+-00>
    #                             = - 3a+ |-+-0+>
    #                             = + 3a+ |-+-++>
    sop = forte.SparseOperator()
    sop.add("[3a+ 4a+ 2b+ 1b- 0a-]", 1.0)
    ref = forte.SparseState({det("22"): 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("-+-++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[0a+]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[0a+ 1a+]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det(""): 1.0}))
    assert wfn[det("++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[0a+ 1a+ 2a+]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[0a+ 1a+ 2a+ 3a+]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det(""): 1.0}))
    assert wfn[det("++++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[0a+ 1a+ 2a+ 3a+ 4a+]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[2a+ 0a+ 3a+ 1a+ 4a+]", 1.0, allow_reordering=True)
    wfn = forte.apply_op(sop, forte.SparseState({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(-1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[0a-]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("-2222")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[1a- 0a-]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("--222")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("---22")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("----2")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[4a- 3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte.apply_op(sop, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(1.0, abs=1e-9)

    sop = forte.SparseOperator()
    sop.add("[1a- 3a- 2a- 4a- 0a-]", 1.0, allow_reordering=True)
    wfn = forte.apply_op(sop, forte.SparseState({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(-1.0, abs=1e-9)

    ### Test for cases that are supposed to return zero ###
    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    sop = forte.SparseOperator()
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("00")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    sop = forte.SparseOperator()
    sop.add("[0a+ 0b+ 0b-]", 1.0)
    dtest = det("00")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    sop = forte.SparseOperator()
    sop.add("[1a+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    sop = forte.SparseOperator()
    sop.add("[1b+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    ### Number operator tests ###
    # test number operator: (0a+ 0a-) |0> = 0
    sop = forte.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("0")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |+> = |+>
    sop = forte.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0a+ 0a-) |-> = 0
    sop = forte.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("-")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |2> = |2>
    sop = forte.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("2")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |0> = 0
    sop = forte.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    dtest = det("0")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |+> = 0
    sop = forte.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    dtest = det("+")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |-> = |->
    sop = forte.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    dtest = det("-")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |2> = |2>
    sop = forte.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    ref = forte.SparseState({det("2"): 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("2")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |222> = |222>
    sop = forte.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte.SparseState({det("222"): 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("222")] == pytest.approx(2.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22+> = |22+>
    sop = forte.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte.SparseState({det("22+"): 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("22+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22-> = |22->
    sop = forte.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte.SparseState({det("22-"): 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("22-")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |220> = |220>
    sop = forte.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte.SparseState({det("220"): 1.0})
    wfn = forte.apply_op(sop, ref)
    assert dtest not in wfn

    ### Excitation operator tests ###
    # test excitation operator: (3a+ 0a-) |2200> = 3a+ |-200> = -|-20+>
    sop = forte.SparseOperator()
    sop.add("[3a+ 0a-]", 1.0)
    dtest = det("22")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("-20+")] == pytest.approx(-1.0, abs=1e-9)

    # test excitation operator: (0a- 3a+) |22> = 0a- |220+> = |-20+>
    sop = forte.SparseOperator()
    sop.add("[0a- 3a+]", 1.0, allow_reordering=True)
    dtest = det("22")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("-20+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (3b+ 0b-) |2200> = 3b+ |+200> = |+20->
    sop = forte.SparseOperator()
    sop.add("[3b+ 0b-]", 1.0)
    dtest = det("22")
    ref = forte.SparseState({dtest: 1.0})
    wfn = forte.apply_op(sop, ref)
    assert wfn[det("+20-")] == pytest.approx(-1.0, abs=1e-9)

    ### Adjoint tests
    # test adjoint of SqOperator is idempotent
    sqop = forte.sparse_operator("[0a+ 1a+ 0b+ 2b- 2a-]", 1.0)
    sqopd = sqop.adjoint()
    sqopdd = sqopd.adjoint()
    assert sqop.str() == sqopdd.str()
    # test adjoint of SqOperator
    ref = forte.sparse_operator("[2a+ 2b+ 0b- 1a- 0a-]", 1.0)
    assert sqopd.str() == ref.str()
    # test adjoint of SparseOperator is idempotent
    sop = forte.SparseOperator()
    sop.add("[3a+ 0a-]", 1.0)
    sop.add("[2b+ 1b-]", 1.0)
    sopd = sop.adjoint()
    sopdd = sopd.adjoint()
    assert sop.str() == sopdd.str()
    # test adjoint of SparseOperator
    ref = forte.SparseOperator()
    ref.add("[0a+ 3a-]", 1.0)
    ref.add("[1b+ 2b-]", 1.0)
    assert sopd.str() == ref.str()


def test_sparse_operator_api():
    from forte import SparseState, SparseOperator, SparseOperatorList, exp, exp_antiherm, fact_exp, fact_exp_antiherm

    # Test apply
    sop = SparseOperator()
    sop.add("[1a+ 0a-]", 0.5)
    sop.add("[1b+ 0b-]", 0.7)
    state = SparseState({det("2"): 1.0})
    new_state = sop @ state
    test_state = SparseState({det("-+"): 0.5, det("+-"): 0.7})
    assert new_state == test_state

    # Test exp
    new_state = exp(sop) @ state
    test_state = forte.SparseState(
        {
            det("2"): 1.0,
            det("-+"): 0.5,
            det("+-"): 0.7,
            det("02"): 0.35,
        }
    )
    assert new_state == test_state

    # Test apply_antiherm
    new_state = exp_antiherm(sop) @ state
    test_state = forte.SparseState(
        {
            det("2"): 0.67121217,
            det("-+"): 0.36668488,
            det("+-"): 0.56535421,
            det("02"): 0.30885441,
        }
    )
    diff = new_state - test_state
    assert diff.norm() < 1e-7

    # Test fact_exp
    sop = SparseOperatorList()
    sop.add("[1a+ 0a-]", 0.5)
    sop.add("[1b+ 0b-]", 0.7)
    new_state = fact_exp(sop) @ state
    test_state = forte.SparseState(
        {
            det("2"): 1.0,
            det("-+"): 0.5,
            det("+-"): 0.7,
            det("02"): 0.35,
        }
    )
    assert new_state == test_state

    # Test fact_exp_antiherm
    sop = SparseOperatorList()
    sop.add("[1a+ 0a-]", 0.5)
    sop.add("[1b+ 0b-]", 0.7)
    new_state = fact_exp_antiherm(sop) @ state
    test_state = forte.SparseState(
        {
            det("2"): 0.67121217,
            det("-+"): 0.36668488,
            det("+-"): 0.56535421,
            det("02"): 0.30885441,
        }
    )
    diff = new_state - test_state
    assert diff.norm() < 1e-7


def test_sparse_operator_product():
    sop1 = forte.SparseOperator()
    sop1.add("[1a+ 1a-]", 1.0)
    sop1.add("[0a+ 0a-]", 1.0)
    sop2 = forte.SparseOperator()
    sop2.add("[3a+ 3a-]", 1.0)
    sop2.add("[2a+ 2a-]", 1.0)
    sop3 = sop1 @ sop2
    sop3_test = forte.SparseOperator()
    sop3_test.add("[0a+ 2a+ 2a- 0a-]", 1.0)
    sop3_test.add("[0a+ 3a+ 3a- 0a-]", 1.0)
    sop3_test.add("[1a+ 2a+ 2a- 1a-]", 1.0)
    sop3_test.add("[1a+ 3a+ 3a- 1a-]", 1.0)
    assert sop3.str() == sop3_test.str()


def test_sparse_operator_list_reverse():
    sopl = forte.SparseOperatorList()
    sopl.add("[1a+ 1a-]", 1.0)
    sopl.add("[0a+ 0a-]", 2.0)
    reversed_sopl = sopl.reverse()
    assert len(sopl) == 2
    assert reversed_sopl[0] == 2.0
    assert reversed_sopl[1] == 1.0
    assert reversed_sopl(0)[0].str() == "[0a+ 0a-]"
    assert reversed_sopl(1)[0].str() == "[1a+ 1a-]"


def test_sparse_operator_list_remove():
    sopl = forte.SparseOperatorList()
    sopl.add("[1a+ 1a-]", 1.0)
    sopl.add("[0a+ 0a-]", 1.0)
    sopl.add("[1a+ 1a-]", 1.0)
    sopl.add("[1a+ 1a-]", 1.0)
    sopl.remove("[1a+ 1a-]")
    assert len(sopl) == 1


def test_sparse_operator_list_add():
    sop1 = forte.SparseOperatorList()
    sop1.add("[1a+ 1a-]", 1.0)
    sop2 = forte.SparseOperatorList()
    sop2.add("[0a+ 0a-]", 1.0)
    sop3 = sop1 + sop2
    assert len(sop3) == 2
    assert sop3(0)[0].str() == "[1a+ 1a-]"
    assert sop3(1)[0].str() == "[0a+ 0a-]"


def test_sparse_operator_list_pop():
    sop = forte.SparseOperatorList()
    sop.add("[1a+ 1a-]", 1.0)
    sop.add("[0a+ 0a-]", 1.0)
    sop.add("[1a+ 1a-]", 1.0)
    assert len(sop) == 3
    sop = sop.pop_left()
    assert len(sop) == 2
    sop = sop.pop_right()
    assert len(sop) == 1
    assert sop(0)[0].str() == "[0a+ 0a-]"


def test_sparse_operator_list_slice():
    sop = forte.SparseOperatorList()
    sop.add("[1a+ 1a-]", 1.0)
    sop.add("[0a+ 0a-]", 1.0)
    sop.add("[1a+ 1a-]", 1.0)
    sop_sl = sop.slice(1, 2)
    assert len(sop_sl) == 1
    assert sop_sl(0)[0].str() == "[0a+ 0a-]"
    sop_sl = sop.slice(1,1)
    assert len(sop_sl) == 0


if __name__ == "__main__":
    test_sparse_operator_creation()
    test_sparse_operator()
    test_sparse_operator_api()
    test_sparse_operator_product()
    test_sparse_operator_list_reverse()
    test_sparse_operator_list_remove()
    test_sparse_operator_list_add()
    test_sparse_operator_list_pop()
    test_sparse_operator_list_slice()
