#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte
import pytest


def test_sq_operator():
    op1 = forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[1a+ 0a-]", -1.0)])
    op1 *= -2.0
    assert op1 == forte.sparse_operator([("[0a+ 0a-]", -2.0), ("[1a+ 0a-]", 2.0)])
    with pytest.raises(Exception):
        op1 = forte.sparse_operator("[0a+ 0a]")
    with pytest.raises(Exception):
        op1 = forte.sparse_operator("[0a+ 0]")
    op1 = forte.sparse_operator("[0a+")
    assert op1 == forte.sparse_operator("[0a+]")

    op1 = forte.sparse_operator([("0a+ 0a-", 1.0), ("1a+ 0a-", -1.0)])
    assert op1 == forte.sparse_operator([("[0a+ 0a-]", 1.0), ("[1a+ 0a-]", -1.0)])


def test_sq_operator_product():
    #
    # test the alpha creation part of the algorithm
    #
    op1 = forte.sparse_operator("[0a+ 0a-]")
    op2 = forte.sparse_operator("[1a+]")
    op3_test = forte.sparse_operator("[0a+ 1a+ 0a-]", -1.0)
    op3 = op1 @ op2
    assert op3 == op3_test

    # test product of commuting operators
    op1 = forte.sparse_operator("[0a+ 2a+]")
    op2 = forte.sparse_operator("[1a+ 3a+]")
    op3_test = forte.sparse_operator("[0a+ 1a+ 2a+ 3a+]", -1.0)
    op3 = op1 @ op2
    assert op3 == op3_test

    op1 = forte.sparse_operator("[0a+ 1a+ 0a-]")
    op2 = forte.sparse_operator("[1a+]")
    op3 = op1 @ op2
    assert len(op3) == 0

    op1 = forte.sparse_operator("[0a+ 2a+ 4b+ 0a-]")
    op2 = forte.sparse_operator("[1a+]")
    op3_test = forte.sparse_operator("[0a+ 1a+ 2a+ 4b+ 0a-]", -1.0)
    op3 = op1 @ op2
    assert op3 == op3_test

    op1 = forte.sparse_operator("[0a+ 1a-]")
    op2 = forte.sparse_operator("[1a+]")
    op3_test = forte.sparse_operator([("[0a+]", 1.0), ("[0a+ 1a+ 1a-]", -1.0)])

    op3 = op1 @ op2
    assert op3 == op3_test

    op1 = forte.sparse_operator("[0b+ 0b-]")
    op2 = forte.sparse_operator("[1b+]")
    op3_test = forte.sparse_operator("[0b+ 1b+ 0b-]", -1.0)
    op3 = op1 @ op2
    assert op3 == op3_test

    print("[2a+ 1b+ 2b- 2a-] [3a+ 3b+]")
    op1 = forte.sparse_operator("[2a+ 1b+ 2b- 2a-]")
    op2 = forte.sparse_operator("[3a+ 3b+]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator("[2a+ 3a+ 1b+ 3b+ 2b- 2a-]", -1.0)
    assert op3 == op3_test

    print("[2a+ 1b+ 3b- 3a-] [3a+]")
    op1 = forte.sparse_operator("[2a+ 1b+ 3b- 3a-]")
    op2 = forte.sparse_operator("[3a+]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator([("[2a+ 1b+ 3b-]", 1.0), ("[2a+ 3a+ 1b+ 3b- 3a-]", -1.0)])
    assert op3 == op3_test

    print("[3b- 3a-] [3b+]")
    op1 = forte.sparse_operator("[3b- 3a-]")
    op2 = forte.sparse_operator("[3b+]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator([("[3a-]", -1.0), ("[3b+ 3b- 3a-]", 1.0)])
    assert op3 == op3_test

    print("[2a+ 1b+ 3b- 3a-] [3b+]")
    op1 = forte.sparse_operator("[2a+ 1b+ 3b- 3a-]")
    op2 = forte.sparse_operator("[3b+]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator([("[2a+ 1b+ 3a-]", -1.0), ("[2a+ 1b+ 3b+ 3b- 3a-]", 1.0)])
    assert op3 == op3_test

    print("[2a+ 1b+ 3b- 3a-] [3a+ 3b+]")
    op1 = forte.sparse_operator("[2a+ 1b+ 3b- 3a-]")
    op2 = forte.sparse_operator("[3a+ 3b+]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator(
        [
            ("[2a+ 1b+]", 1.0),
            ("[2a+ 1b+ 3b+ 3b-]", -1.0),
            ("[2a+ 3a+ 1b+ 3a-]", +1.0),
            ("[2a+ 3a+ 1b+ 3b+ 3b- 3a-]", -1.0),
        ]
    )
    assert op3 == op3_test

    print("[2a+ 3a-] [3a+ 2a-]")
    op1 = forte.sparse_operator("[2a+ 3a-]")
    op2 = forte.sparse_operator("[3a+ 2a-]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator([("[2a+ 2a-]", 1.0), ("[2a+ 3a+ 3a- 2a-]", -1.0)])
    assert op3 == op3_test

    print("[2b+ 3b-] [3b+ 2b-]")
    op1 = forte.sparse_operator("[2b+ 3b-]")
    op2 = forte.sparse_operator("[3b+ 2b-]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator([("[2b+ 2b-]", 1.0), ("[2b+ 3b+ 3b- 2b-]", -1.0)])
    assert op3 == op3_test

    print("[5a+ 2b+ 3b- 7a-] [2a+ 1b+ 1b- 1a-]")
    op1 = forte.sparse_operator("[5a+ 2b+ 3b- 7a-]")
    op2 = forte.sparse_operator("[2a+ 1b+ 1b- 1a-]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator("[2a+ 5a+ 1b+ 2b+ 3b- 1b- 7a- 1a-]", 1.0)
    assert op3 == op3_test

    print("[5a+ 2b+ 3b- 7a-] [6a+ 1b+ 1b- 1a-]")
    op1 = forte.sparse_operator("[5a+ 2b+ 3b- 7a-]")
    op2 = forte.sparse_operator("[6a+ 1b+ 1b- 1a-]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator("[5a+ 6a+ 1b+ 2b+ 3b- 1b- 7a- 1a-]", -1.0)
    assert op3 == op3_test

    print("[5a+ 2b+ 3b- 7a-] [6a+ 4b+ 1b- 1a-]")
    op1 = forte.sparse_operator("[5a+ 2b+ 3b- 7a-]")
    op2 = forte.sparse_operator("[6a+ 4b+ 1b- 1a-]")
    op3 = op1 @ op2
    op3_test = forte.sparse_operator("[5a+ 6a+ 2b+ 4b+ 3b- 1b- 7a- 1a-]", 1.0)
    assert op3 == op3_test


def test_sq_operator_commutator():
    print("[[1a+ 0a-],[2a+ 3a-]]")
    op1 = forte.sparse_operator("[1a+ 0a-]")
    op2 = forte.sparse_operator("[2a+ 3a-]")
    op3 = op1.commutator(op2)
    nullop = forte.SparseOperator()
    assert op3 == nullop

    print("[[1a+ 0b-],[0b+ 3a-]]")
    op1 = forte.sparse_operator("[1a+ 0b-]")
    op2 = forte.sparse_operator("[0b+ 3a-]")
    op3 = op1.commutator(op2)
    op3_test = forte.sparse_operator("[1a+ 3a-]", 1.0)
    assert op3 == op3_test

    print("[[1a+ 0b-],[0b+ 1a-]]")
    op1 = forte.sparse_operator("[1a+ 0b-]")
    op2 = forte.sparse_operator("[0b+ 1a-]")
    op3 = op1.commutator(op2)
    op3_test = forte.sparse_operator([("[0b+ 0b-]", -1.0), ("[1a+ 1a-]", 1.0)])
    assert op3 == op3_test


if __name__ == "__main__":
    test_sq_operator()
    test_sq_operator_product()
    test_sq_operator_commutator()
