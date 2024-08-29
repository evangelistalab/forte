#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def test_sparse_operator_string_count():
    sop, _ = forte.sqop("[1a+ 3a+ 3a- 2a-]")
    assert sop.count() == 4


def test_sparse_operator_string_is_number():
    sop, _ = forte.sqop("[1a+ 3a+ 3a- 2a-]")
    assert sop.is_identity() is False

    sop, _ = forte.sqop("[1a+]")
    assert sop.is_identity() is False

    sop, _ = forte.sqop("[1a+ 1a-]")
    assert sop.is_identity() is False

    sop, _ = forte.sqop("[]")
    assert sop.is_identity() is True


def test_sparse_operator_string_is_nilpotent():
    sop, _ = forte.sqop("[1a+ 3a+ 3a- 2a-]")
    assert sop.is_nilpotent() is True

    sop, _ = forte.sqop("[1a+]")
    assert sop.is_nilpotent() is True

    # number operators and the identity operator are not nilpotent
    sop, _ = forte.sqop("[1a+ 1a-]")
    assert sop.is_nilpotent() is False

    sop, _ = forte.sqop("[]")
    assert sop.is_nilpotent() is False


def test_sparse_operator_string_commutator_type():
    # test commuting terms
    sop1, _ = forte.sqop("[1a+ 0a-]")
    sop2, _ = forte.sqop("[3a+ 2a-]")
    assert forte.commutator_type(sop1, sop2) == forte.CommutatorType.commute

    sop1, _ = forte.sqop("[1a+]")
    sop2, _ = forte.sqop("[3a+]")
    assert forte.commutator_type(sop1, sop2) == forte.CommutatorType.anticommute

    sop1, _ = forte.sqop("[1a+]")
    sop2, _ = forte.sqop("[3a-]")
    assert forte.commutator_type(sop1, sop2) == forte.CommutatorType.anticommute

    sop1, _ = forte.sqop("[1a+ 3a-]")
    sop2, _ = forte.sqop("[3a-]")
    assert forte.commutator_type(sop1, sop2) == forte.CommutatorType.may_not_commute


def test_sparse_operator_string_components():
    # test the number and non-number components functions
    sop, _ = forte.sqop("[1a+ 3a+ 3a- 2a-]")
    sop_n = sop.number_component()
    assert sop_n == forte.sqop("[3a+ 3a-]")[0]
    sop_nn = sop.non_number_component()
    assert sop_nn == forte.sqop("[1a+ 2a-]")[0]


if __name__ == "__main__":
    test_sparse_operator_string_count()
    test_sparse_operator_string_is_number()
    test_sparse_operator_string_is_nilpotent()
    test_sparse_operator_string_commutator_type()
    test_sparse_operator_string_components()
