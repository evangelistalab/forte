import math
import forte
import numpy as np

# Description:
# This file contains tests for the SparseOperator class
# and its algebraic operations.
# The tests are divided into three main categories:
# 1. Tests for basic algebraic operations of operators
# 2. Tests for operator products and commutators
# 3. Tests for the fast product of operators


def test_sparse_operator_ops_1():
    nullop = forte.SparseOperator()
    # identity operator

    # test identity operator is not null
    A = forte.sparse_operator("[]", 1.0)
    assert A != nullop

    # test that a regular operator is not the null operator
    A = forte.sparse_operator("[0a+ 0a-]", 1.0)
    assert A != nullop

    # test basic algebraic operations of operators
    A = forte.sparse_operator("[0a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0b+ 0b-]", 1.0)
    C = A + B
    assert C == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[0b+ 0b-]", 1.0)])
    C = A - B
    assert C == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[0b+ 0b-]", -1.0)])
    C = 2.0 * A
    assert C == forte.sparse_operator("[0a+ 0a-]", 2.0)
    C = A * 2.0
    assert C == forte.sparse_operator("[0a+ 0a-]", 2.0)
    C.copy(A)
    C += B
    assert C == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[0b+ 0b-]", 1.0)])
    C.copy(A)
    C -= B
    assert C == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[0b+ 0b-]", -1.0)])
    C.copy(A)
    C *= 2.0
    assert C == forte.sparse_operator("[0a+ 0a-]", 2.0)
    C.copy(A)
    C *= 2.0j
    assert C == forte.sparse_operator("[0a+ 0a-]", 2.0j)

    A = forte.sparse_operator("[0a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0b+ 0b-]", 1.0)
    A += 2.0 * B
    assert A == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[0b+ 0b-]", 2.0)])
    A += B * np.cos(2.0)
    assert A == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[0b+ 0b-]", 2.0 + np.cos(2.0))])

    # test == operator
    assert A == forte.sparse_operator("[0a+ 0a-]", 1.0)

    A = forte.sparse_operator([("[13a+ 2a-]", 1.0), ("[2a+ 13a-]", -1.0)])
    B = forte.sparse_operator([("[13a+ 2a-]", 1.0), ("[2a+ 13a-]", -1.0)])
    C = forte.sparse_operator([("[13a+ 2a-]", 1.0), ("[2a+ 14a-]", -1.0)])
    D = forte.sparse_operator([("[13a+ 2a-]", 1.0), ("[1a+ 13a-]", -1.0)])
    E = forte.sparse_operator([("[14a+ 2a-]", 1.0), ("[2a+ 13a-]", -1.0)])
    F = forte.sparse_operator([("[14a+ 2a-]", 1.0), ("[2a+ 13a-]", -1.1)])
    # test equivalent operators
    assert A == B
    # test non-equivalent operators
    assert A != C
    assert A != D
    assert A != E
    assert A != F
    assert A != nullop

    A = forte.sparse_operator([("[2a+ 1a-]", +0.6), ("[1a+ 2a-]", -0.6)])
    B = forte.sparse_operator([("[1a+ 2a-]", -0.6), ("[2a+ 1a-]", +0.6)])
    C = forte.sparse_operator([("[1a+ 2a-]", +0.6), ("[2a+ 1a-]", -0.6)])
    assert A == B
    assert A != C
    assert A == -C
    assert C == -B

    C.copy(A)
    C /= 2.0
    assert C == forte.sparse_operator([("[2a+ 1a-]", +0.3), ("[1a+ 2a-]", -0.3)])
    C = A / 2.0j
    print(f"{A = }")
    print(f"{C = }")
    assert C == forte.sparse_operator([("[2a+ 1a-]", -0.3j), ("[1a+ 2a-]", 0.3j)])

    A_str = forte.sqop("[2a+ 1a-]")[0]
    Ad_str = A_str.adjoint()
    A = forte.SparseOperator()
    A += A_str * 0.6j
    A += -0.6j * Ad_str
    assert A == forte.sparse_operator([("[2a+ 1a-]", +0.6j), ("[1a+ 2a-]", -0.6j)])


def test_sparse_operator_ops_2():
    nullop = forte.SparseOperator()

    # test commuting operators
    A = forte.sparse_operator("[0a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0b+ 0b-]", 1.0)
    C = A @ B
    assert C == forte.sparse_operator("[0a+ 0b+ 0b- 0a-]", 1.0)
    D = B @ A
    assert D == forte.sparse_operator("[0a+ 0b+ 0b- 0a-]", 1.0)
    E = A.commutator(B)
    assert E == nullop

    # test operator repetition
    A = forte.sparse_operator("[1a+ 0a-]", 1.0)
    B = forte.sparse_operator("[1a+ 0b-]", 1.0)
    C = A @ B
    assert C == nullop
    D = B @ A
    assert D == nullop
    E = A.commutator(B)
    assert E == nullop

    # test product with a contraction and repeated indices
    A = forte.sparse_operator("[0a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0a+ 0b-]", -1.0)
    C = A @ B
    assert C == forte.sparse_operator("[0a+ 0b-]", -1.0)
    D = B @ A
    assert D == nullop
    E = A.commutator(B)
    assert E == forte.sparse_operator("[0a+ 0b-]", -1.0)

    # test product with a contraction and repeated indices
    A = forte.sparse_operator("[0a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0b+ 0a-]", -1.0)
    C = A @ B
    assert C == nullop
    D = B @ A
    assert D == forte.sparse_operator("[0b+ 0a-]", -1.0)
    E = A.commutator(B)
    assert E == forte.sparse_operator("[0b+ 0a-]", 1.0)

    # test commutator with one anti-hermitian operator
    A = forte.sparse_operator([("[1a+ 0a-]", 1.0), ("[0a+ 1a-]", -1.0)])
    B = forte.sparse_operator("[0a+ 2a-]", 1.0)
    C = A @ B
    assert C == forte.sparse_operator([("[1a+ 2a-]", 1.0), ("[0a+ 1a+ 2a- 0a-]", -1.0)])
    C = B @ A
    assert C == forte.sparse_operator("[0a+ 1a+ 2a- 0a-]", -1.0)

    A = forte.sparse_operator([("[1a+ 0a-]", 2.0), ("[0a+ 1a-]", -2.0)])
    B = forte.sparse_operator([("[0a+ 2a-]", -0.3), ("[2a+ 0a-]", 0.3)])
    C_test = forte.sparse_operator([("[1a+ 2a-]", -0.6), ("[0a+ 1a+ 2a- 0a-]", 0.6), ("[0a+ 2a+ 1a- 0a-]", 0.6)])
    C = A @ B
    assert C == C_test
    C = A.commutator(B)
    C_test = forte.sparse_operator([("[2a+ 1a-]", 0.6), ("[1a+ 2a-]", -0.6)])
    assert C == C_test


def test_sparse_operator_commutator():
    A = forte.sparse_operator("[1a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0a+ 1a-]", 1.0)
    C = A.commutator(B)
    assert C == forte.sparse_operator([("[0a+ 0a-]", -1.0), ("[1a+ 1a-]", 1.0)])

    A = forte.sparse_operator("[1a+ 0a-]", 1.0)
    B = forte.sparse_operator("[0a+ 2a-]", 1.0)
    C = A.commutator(B)
    assert C == forte.sparse_operator("[1a+ 2a-]", 1.0)

    # test commutator with one anti-hermitian operator
    A = forte.sparse_operator([("[1a+ 0a-]", 1.0), ("[0a+ 1a-]", -1.0)])
    B = forte.sparse_operator("[0a+ 2a-]", 1.0)
    C = A.commutator(B)
    assert C == forte.sparse_operator("[1a+ 2a-]", 1.0)

    # test commutator with two anti-hermitian operators
    A = forte.sparse_operator([("[1a+ 0a-]", 1.0), ("[0a+ 1a-]", -1.0)])
    B = forte.sparse_operator([("[0a+ 2a-]", 1.0), ("[2a+ 0a-]", -1.0)])
    C = A.commutator(B)
    assert C == forte.sparse_operator([("[1a+ 2a-]", 1.0), ("[2a+ 1a-]", -1.0)])


def test_sparse_operator_fast_product():
    A = forte.sparse_operator("[1a+ 2a-]", 1.0)
    B = forte.sparse_operator("[0a+ 3a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[1a+ 2b+ 3b- 4a-]", 1.0)
    B = forte.sparse_operator("[5a+ 6b+ 7b- 8a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[1a+ 2b+ 4a-]", 1.0)
    B = forte.sparse_operator("[5a+ 7b- 8a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[1a+ 4a-]", 1.0)
    B = forte.sparse_operator("[5a+ 2b+ 7b- 8a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[]", -1.0)
    B = forte.sparse_operator("[5a+ 2b+ 7b- 8a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[0a+ 2a-]", 1.0)
    B = forte.sparse_operator("[2a+ 2a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[0a+ 1a+ 4a- 2a-]", 1.0)
    B = forte.sparse_operator("[2a+ 7a+ 3a- 2a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[0a+ 1a+ 3a- 2a-]", 1.0)
    B = forte.sparse_operator("[2a+ 7a+ 3a- 2a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[0a+ 2a-]", 1.0)
    B = forte.sparse_operator("[2a+ 1a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator("[0a+ 2a-]", 1.0)
    B = forte.sparse_operator("[2a+ 0a-]", 1.0)
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator([("[1a+ 0a-]", 1.0)])
    B = forte.sparse_operator([("[0a+ 2a-]", 1.0)])
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator([("[1a+ 0a-]", 1.0)])
    B = forte.sparse_operator([("[0a+ 2a+ 2a- 1a-]", 1.0)])
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator([("[1a+ 2a+ 0a-]", 1.0)])
    B = forte.sparse_operator([("[0a+ 2a+ 2a- 1a-]", 1.0)])
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator([("[1a+ 2a+ 2a-]", 1.0)])
    B = forte.sparse_operator([("[0a+ 2a+ 2a- 1a-]", 1.0)])
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    A = forte.sparse_operator([("[1a+ 0a-]", 2.0), ("[0a+ 1a-]", -2.0)])
    B = forte.sparse_operator([("[0a+ 2a-]", -0.3), ("[2a+ 0a-]", 0.3)])
    C = forte.new_product(A, B)
    D = A @ B
    assert C == D

    # generate all possible operators with 2 indices in the range [0,1,2,3]
    operators = []
    A = forte.SparseOperator()

    max_single_index = 6
    for i in range(max_single_index):
        A += forte.sparse_operator(f"[{i}a+]", 1.0)
        A += forte.sparse_operator(f"[{i}b+]", 1.0)
        A += forte.sparse_operator(f"[{i}a-]", 1.0)
        A += forte.sparse_operator(f"[{i}b-]", 1.0)
        for j in range(max_single_index):
            A += forte.sparse_operator(f"[{i}a+ {j}a-]", 1.0)
            A += forte.sparse_operator(f"[{i}a+ {j}b-]", 1.0)
            A += forte.sparse_operator(f"[{i}b+ {j}a-]", 1.0)
            A += forte.sparse_operator(f"[{i}b+ {j}b-]", 1.0)

    # add all operators with two indices in the range [0,1,2,3]
    # of the form [ia+ ja+ la+ ka+] and i < j and l > k
    max_double_index = 6
    for i in range(max_double_index):
        for j in range(max_double_index):
            for l in range(max_double_index):
                if i < j:
                    A += forte.sparse_operator(f"[{i}a+ {j}a+ {l}a-]", 1.0)
                    A += forte.sparse_operator(f"[{i}b+ {j}b+ {l}b-]", 1.0)
                    A += forte.sparse_operator(f"[{i}b+ {j}b+ {l}a-]", 1.0)
                    A += forte.sparse_operator(f"[{i}a+ {j}a+ {l}b-]", 1.0)
                for k in range(max_double_index):
                    if i < j and l > k:
                        A += forte.sparse_operator(f"[{i}a+ {j}a+ {l}a- {k}a-]", 1.0)
                        A += forte.sparse_operator(f"[{i}b+ {j}b+ {l}a- {k}a-]", 1.0)
                        A += forte.sparse_operator(f"[{i}b+ {j}b+ {l}b- {k}b-]", 1.0)
                        A += forte.sparse_operator(f"[{i}a+ {j}b+ {l}b- {k}a-]", 1.0)

    # add all operators with three indices in the range [0,1,2,3,4,5]
    # of the form [ia+ ja+ ka+ na- ma- la-] and i < j < k and n > m > l
    max_triple_index = 3
    for i in range(max_triple_index):
        for j in range(max_triple_index):
            for k in range(max_triple_index):
                for n in range(max_triple_index):
                    for m in range(max_triple_index):
                        for l in range(max_triple_index):
                            if i < j < k and n > m > l:
                                A += forte.sparse_operator(f"[{i}a+ {j}a+ {k}a+ {n}a- {m}a- {l}a-]", 1.0)
                                A += forte.sparse_operator(f"[{i}b+ {j}b+ {k}b+ {n}b- {m}b- {l}b-]", 1.0)
                            if i < j and m > l:
                                A += forte.sparse_operator(f"[{i}a+ {j}a+ {k}b+ {n}b- {m}a- {l}a-]", 1.0)
                            if j < k and n > m:
                                A += forte.sparse_operator(f"[{i}a+ {j}b+ {k}b+ {n}b- {m}b- {l}a-]", 1.0)

    print(f"Computing {A.size()} operators")

    # add timing
    import time

    B = forte.SparseOperator()
    B += A

    start = time.time()
    C = forte.new_product(A, B)
    end = time.time()
    print(f"Time elapsed for new_product : {end - start}")

    start = time.time()
    D = A @ B
    end = time.time()
    print(f"Time elapsed for @           : {end - start}")

    start = time.time()
    E = forte.new_product2(A, B)
    end = time.time()
    print(f"Time elapsed for new_product2: {end - start}")

    print(f"Number of items in C: {C.size()}")
    print(f"Number of items in D: {D.size()}")
    print(f"Number of items in E: {E.size()}")

    start = time.time()
    assert C == D
    assert C == E
    assert D == E
    end = time.time()
    print(f"Time elapsed: {end - start}")


if __name__ == "__main__":
    test_sparse_operator_ops_1()
    test_sparse_operator_ops_2()
    test_sparse_operator_commutator()
    test_sparse_operator_fast_product()
