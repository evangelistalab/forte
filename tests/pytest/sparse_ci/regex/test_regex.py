#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def test_regex():
    import forte

    ### Operator parsing tests ###
    gop = forte.SparseOperator()
    gop.add_term_from_str('[1a+ 0a-]', 1.0)
    test_str = gop.str()
    print(test_str)
    ref_str = '1.000000000000 * [ 1a+ 0a- ]'
    assert test_str[0] == ref_str


if __name__ == "__main__":
    test_regex()
