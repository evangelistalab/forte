#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def test_sparse_operator_product():
    sop1 = forte.SparseOperator()
    sop1.add_term_from_str("[0a+ 0a-]", 1.0)

    sop2 = forte.SparseOperator()
    sop2.add_term_from_str("[1a+ 0a-]", 1.0)
    sop2.add_term_from_str("[0a+ 1a-]", -1.0)

    sqop = sop1
    sqop += sop2

    print(sqop.str())


if __name__ == "__main__":
    test_sparse_operator_product()
