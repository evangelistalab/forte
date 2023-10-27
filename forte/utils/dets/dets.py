#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

def parse_sign(s):
    if s == '' or s == '+':
        return 1.0
    if s == '-':
        return -1.0
    print(f'There was an error parsing the sign {s}')

def parse_factor(s):
    if s == '':
        return 1.0
    return(float(s))

def parse_ops(s):
    ops = []
    # we reverse the operator order
    for op in s[1:-1].split(' ')[::-1]:
        creation = True if op[-1] == '+' else False
        alpha = True if op[-2] == 'a' else False
        orb = int(op[0:-2])
        ops.append((creation,alpha,orb))
    return ops

def to_ops(str):
    terms = []
    # '<something>[1b+ 0b+] +-<something>[1b+ 0b+]'
    match_op = r'\s?([\+\-])?\s*(\d*\.?\d*)?\s*\*?\s*(\[[0-9ab\+\-\s]*\])'
    m = re.findall(match_op,str)
    if m:
        for group in m:
            sign = parse_sign(group[0])
            factor = parse_factor(group[1])
            ops = parse_ops(group[2])
            terms.append((sign * factor,ops))
    return terms

def print_wfn(wfn, n):
    for d, c in wfn.items():
        print(f'{c:+20.12f} {d.str(n)}')
