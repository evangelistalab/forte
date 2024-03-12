#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def compute_st_taylor(O, A, theta):
    stO = forte.SparseOperator()
    stO += O
    C = forte.SparseOperator()
    C += O
    for i in range(1, 30):
        C = (1 / i) * C.commutator(A)
        stO += C
        if C.norm() < 1e-16:
            print(f"  A norm: {C.norm()} after {i} iterations")
            break
    return stO


def compute_st(O, unormA, theta):
    import numpy as np

    A = unormA * (1 / theta)
    stO = forte.SparseOperator()
    cOA = O.commutator(A)
    cOAA = cOA.commutator(A)
    N = -1.0 * A @ A
    # NO = N @ O
    # AO = A @ O

    # NOA = NO @ A
    # AON = AO @ N
    # print(f"{NOA = }")
    # print(f"{AON = }")

    # aNcOA = N @ cOA + cOA @ N
    # print(f"{aNcOA = }")
    # print(f"{cOA = }")

    # stO += O
    # stO += np.sin(theta) * cOA
    # stO += 0.5 * np.sin(theta) ** 2 * cOAA
    # stO += -2.0 * np.sin(theta / 2) ** 4 * (N @ O + O @ N)
    # stO += np.sin(theta) * (np.cos(theta) - 1) * (N @ O @ A)
    # stO += -np.sin(theta) * (np.cos(theta) - 1) * (A @ O @ N)
    # stO += 4.0 * np.sin(theta / 2) ** 4 * (N @ O @ N)

    # working version
    stO += O
    stO += np.sin(theta) * (2 - np.cos(theta)) * cOA
    stO += 0.5 * np.sin(theta) ** 2 * cOAA
    stO += -2.0 * np.sin(theta / 2) ** 4 * (N @ O + O @ N)
    stO += np.sin(theta) * (np.cos(theta) - 1) * (N @ cOA + cOA @ N)
    stO += 4.0 * np.sin(theta / 2) ** 4 * (N @ O @ N)

    return stO


def test_sparse_operator_transform():
    import math

    O = forte.make_sparse_operator("[2a+ 0a-]", 1.0)
    print(O)

    theta = 0.7  # math.pi / 2
    A = forte.make_sparse_operator([("[2a+ 0a-]", theta), ("[0a+ 2a-]", -theta)])
    print(A)

    C1 = compute_st_taylor(O, A, theta)
    print("C1 = ", C1)

    # N = B @ B
    # N *= -1.0
    # print(N.str())

    C2 = compute_st(O, A, theta)
    print("C2 = ", C2)

    print((C1 - C2).norm())

    forte.similarity_transform(O, A)
    print("O = ", O)


if __name__ == "__main__":
    test_sparse_operator_transform()
