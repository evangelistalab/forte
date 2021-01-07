#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte

def test_regex():
    import forte

    ### Operator ordering tests ###
    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    gop = forte.GeneralOperator()
    gop.add_term_from_str('[1a+ 0a-] - [0a+ 1a-]',1.0)
    test_str = gop.str()
    ref_str = '1.000000 * ( 1.000000 * [ 1a+ 0a- ] -1.000000 * [ 0a+ 1a- ] )'
    assert test_str[0] == ref_str

test_regex()
