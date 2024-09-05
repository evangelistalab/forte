#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import forte


def compute_st_taylor(O, S):
    """Compute stO = exp(-S) O exp(S) numerically"""
    stO = forte.SparseOperator()
    stO += O
    C = forte.SparseOperator()
    C += O
    for i in range(1, 25):
        C = (1 / i) * C.commutator(S)
        stO += C
        if abs(C.norm()) < 1e-16:
            break
    return stO


def compute_st_antihermitian(O, unormA, theta):
    """Compute stO = exp(-A) O exp(A) from exact equations for antihermitian A"""
    A = unormA * (1 / theta)
    cOA = O.commutator(A)
    cOAA = cOA.commutator(A)
    N = -1.0 * A @ A
    stO = forte.SparseOperator(O)
    stO += cOA * np.sin(theta) * (2 - np.cos(theta))
    stO += cOAA * 0.5 * np.sin(theta) ** 2
    stO += (N @ O + O @ N) * (-2.0 * np.sin(theta / 2) ** 4)
    stO += (N @ cOA + cOA @ N) * np.sin(theta) * (np.cos(theta) - 1)
    stO += (N @ O @ N) * 4.0 * np.sin(theta / 2) ** 4
    return stO


def compute_st_nilpotent(O, unormA, theta):
    import numpy as np

    sqop, a = unormA(0)
    A = forte.SparseOperator()
    A.add(sqop, 1.0)
    stO = forte.SparseOperator()
    cOA = O.commutator(A)

    # working version
    stO += O
    stO += theta * cOA
    stO += theta**2 * A @ cOA

    return stO


def run_test_sparse_operator_transform(type, O, A, theta):
    print(f"\n{O = }")
    print(f"{A = }")

    sqop, a = A(0)
    A2 = forte.SparseOperator()
    A2.add(sqop, a)
    if type == "antiherm":
        A2 = A2 - A2.adjoint()

    C_taylor = compute_st_taylor(O, A2)
    if type == "antiherm":
        C_python = compute_st_antihermitian(O, A2, theta)
        forte.sim_trans_fact_antiherm(O, A)
    else:
        C_python = compute_st_nilpotent(O, A, theta)
        forte.sim_trans_fact_op(O, A)

    python_error = (C_python - C_taylor).norm()
    forte_error = (O - C_taylor).norm()
    if forte_error > 1e-10:
        print("O: ", O)
        print("A: ", A)
        print("C_taylor: ", C_taylor)
        print("C_python: ", C_python)
    # print("Error for python: ", python_error)
    # print("Error for c++:    ", forte_error)
    assert abs(python_error) < 1e-10
    assert abs(forte_error) < 1e-10


def test_sparse_operator_transform_1():
    O = forte.sparse_operator("[2a+ 0a-]", 1.0)
    theta = 0.3  # math.pi / 2
    A = forte.operator_list("[2a+ 0a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_2():
    O = forte.sparse_operator("[0a+ 2a+ 1a- 0a-]", 1.0)
    theta = 0.23  # math.pi / 2
    A = forte.operator_list("[2a+ 0a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_3():
    O = forte.sparse_operator("[0a+ 3a+ 2a- 0a-]", 1.0)
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[2a+ 0a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_4():
    O = forte.sparse_operator("[0a+ 3b+ 2b- 0a-]", 1.0)
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[2a+ 0a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_5():
    O = forte.sparse_operator("[0b+ 0b-]", 1.0)
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[2b+ 0b-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_6():
    O = forte.sparse_operator("[0a+ 1a+ 7a- 3a-]", 1.0)
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[1a+ 7a+ 3a- 2a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_7():
    O = forte.sparse_operator("[1a+ 4a+ 7a- 3a-]", 1.0)
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[0a+ 7a+ 2a- 1a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_8():
    O = forte.sparse_operator("[0a+ 0a-]", 1.0)
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[2a- 0a-]", theta)
    run_test_sparse_operator_transform("exc", O, A, theta)


def test_sparse_operator_transform_9():
    O = forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[1a+ 1a-]", 1.0), ("[1a+ 4a+ 7a- 3a-]", 1.0)])
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[0a+ 7a+ 2a- 1a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


def test_sparse_operator_transform_10():
    O = forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[1a+ 1a-]", 1.0), ("[1a+ 4a+ 7a- 3a-]", 1.0)])
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[0a+ 1a+ 2a- 0a-]", theta)
    run_test_sparse_operator_transform("exc", O, A, theta)

    O = forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[1a+ 1a-]", 1.0), ("[1a+ 4a+ 7a- 3a-]", 1.0)])
    theta = 0.37  # math.pi / 2
    A = forte.operator_list("[0a+ 1a+ 2a- 0a-]", theta)
    run_test_sparse_operator_transform("antiherm", O, A, theta)


if __name__ == "__main__":
    test_sparse_operator_transform_1()
    test_sparse_operator_transform_2()
    test_sparse_operator_transform_3()
    test_sparse_operator_transform_4()
    test_sparse_operator_transform_5()
    test_sparse_operator_transform_6()
    test_sparse_operator_transform_7()
    test_sparse_operator_transform_8()
    test_sparse_operator_transform_9()
    test_sparse_operator_transform_10()
