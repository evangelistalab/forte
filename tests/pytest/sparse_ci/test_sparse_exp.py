import math
import forte
import numpy as np
import time
from forte import det


def test_sparse_exp_1():
    import pytest

    ### Test the linear operator ###
    op = forte.SparseOperator()
    ref = forte.SparseState({det("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.2)
    op.add("[2a+ 0a-]", 0.2)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)
    op.add("[3a+ 3b+ 1b- 1a-]", -0.21)
    op.add("[1a+ 1b+ 3b- 3a-]", 0.13 * 0.17)

    wfn = forte.apply_op(op, ref)
    assert det("2200") not in wfn
    assert wfn[det("+2-0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.15, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.21, abs=1e-9)

    ### Test the exponential operator with excitation operator ###
    op = forte.SparseOperator()
    ref = forte.SparseState({det("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)
    op.add("[3a+ 3b+ 1b- 1a-]", -0.077)

    exp = forte.SparseExp()
    wfn = exp.apply_op(op, ref)
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    wfn = exp.apply_op(op, ref)
    assert wfn[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[det("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[det("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[det("-0+2")] == pytest.approx(-0.0077, abs=1e-9)

    ### Test the exponential operator with antihermitian operator ###
    op = forte.SparseOperator()
    ref = forte.SparseState({det("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)

    exp = forte.SparseExp()
    wfn = exp.apply_antiherm(op, ref)
    assert wfn[det("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.978860446763, abs=1e-9)

    exp = forte.SparseExp()
    wfn = exp.apply_antiherm(op, ref)
    wfn2 = exp.apply_antiherm(op, wfn, scaling_factor=-1.0)
    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[det("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[det("-2+0")] == pytest.approx(0.0, abs=1e-9)

    ### Test the factorized exponential operator with an antihermitian operator ###
    op = forte.SparseOperatorList()
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.2)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)
    ref = forte.SparseState({det("22"): 1.0})

    factexp = forte.SparseFactExp()
    wfn = factexp.apply_antiherm(op, ref)

    assert wfn[det("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    wfn2 = factexp.apply_antiherm(op, wfn, inverse=True)

    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)

    factexp = forte.SparseFactExp()
    wfn = factexp.apply_antiherm(op, ref)

    assert wfn[det("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[det("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[det("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[det("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    wfn2 = factexp.apply_antiherm(op, wfn, inverse=True)

    assert wfn2[det("2200")] == pytest.approx(1.0, abs=1e-9)

    op = forte.SparseOperatorList()
    op.add("[1a+ 0a-]", 0.1)
    op.add("[1a+ 1b+ 0b- 0a-]", -0.3)
    op.add("[1b+ 0b-]", 0.05)
    op.add("[2a+ 2b+ 1b- 1a-]", -0.07)

    dtest = det("20")
    ref = forte.SparseState({det("20"): 0.5, det("02"): 0.8660254038})
    factexp = forte.SparseFactExp()
    wfn = factexp.apply_antiherm(op, ref)

    assert wfn[det("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn[det("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn[det("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn[det("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn[det("-+0")] == pytest.approx(0.016058887563, abs=1e-9)


def test_sparse_exp_2():
    # Compare the performance of the two methods to apply an operator to a state
    # when the operator all commute with each other
    norb = 10
    nocc = 5
    nvir = norb - nocc
    amp = 0.1

    # Create a random operator
    oplist = forte.SparseOperatorList()

    for i in range(nocc):
        for a in range(nocc, norb):
            oplist.add(f"[{a}a+ {i}a-]", amp / (1 + (a - i) ** 2))
            oplist.add(f"[{a}b+ {i}b-]", amp / (1 + (a - i) ** 2))

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, norb):
                for b in range(nocc, norb):
                    if i < j and a < b:
                        oplist.add(f"[{a}a+ {b}a+ {j}a- {i}a-]", amp / (1 + (a + b - i - j) ** 2))
                        oplist.add(f"[{a}b+ {b}b+ {j}b- {i}b-]", amp / (1 + (a + b - i - j) ** 2))
                    oplist.add(f"[{a}a+ {b}b+ {j}b- {i}a-]", amp / (1 + (a + b - i - j) ** 2))

    op = oplist.to_operator()

    # Apply the operator to the reference state timing it
    ref = forte.SparseState({det("2" * nocc): 1.0})
    start = time.time()
    exp = forte.SparseExp()
    A = exp.apply_op(op, ref)
    end = time.time()
    print(f"Time to apply operator: {end - start:.8f} (SparseExp)")

    # Apply the operator to the reference state timing it
    ref = forte.SparseState({det("2" * nocc): 1.0})
    start = time.time()
    exp = forte.SparseFactExp()
    B = exp.apply_op(oplist, ref)
    end = time.time()
    print(f"Time to apply operator: {end - start:.8f} (SparsFactExp)")

    # Check that the two methods give the same result
    AmB = forte.SparseState(A)
    AmB -= B
    print(f"|A| = {A.norm()}")
    print(f"|B| = {B.norm()}")
    print(f"size(A) = {len(A)}")
    print(f"size(B) = {len(B)}")
    print(f"|A - B| = {AmB.norm()}")
    assert abs(AmB.norm()) < 1e-9

    # Apply the operator to the reference state timing it
    ref = forte.SparseState({det("2" * nocc): 1.0})
    start = time.time()
    exp = forte.SparseFactExp(screen_thresh=1.0e-14)
    C = exp.apply_antiherm(oplist, ref)
    end = time.time()
    print(f"Time to apply operator: {end - start:.8f} (SparsFactExp::antiherm)")
    print(f"|C| = {C.norm()}")
    assert abs(C.norm() - 1) < 1.0e-10


if __name__ == "__main__":
    test_sparse_exp_1()
    test_sparse_exp_2()
