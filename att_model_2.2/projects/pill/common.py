#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from projects.hw import common as hw

MAX_SEQ_LEN = 20


parent_dir = os.path.dirname(__file__)

DIGITS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
DIGITS.extend(['-', '＜', '.', ' ', '﹀', '®', '|', '△', '&', '/', '＞', '+'])
DIGITS = DIGITS[:2500]


NUM_CLASSES = len(DIGITS) + 1
_code_map = {}
for i in range(len(DIGITS)):
    _code_map[DIGITS[i]] = i


def get_code(s):
    return _code_map[s]


def code2vec(s):
    vec = []
    for c in s:
        if c in _code_map:
            n = _code_map[c]
            vec.append(n)
    if len(vec) >= MAX_SEQ_LEN:
        return [-1] * MAX_SEQ_LEN
    vec.extend([-1] * (MAX_SEQ_LEN - len(vec)))
    return vec


def vec2code(vec):
    code = ''
    hw = False
    for i in vec:
        if i == -1:
            break
        if i >= len(DIGITS):
            i -= len(DIGITS)
            if not hw:
                hw = True
                code += "$"
        elif hw:
            hw = False
            code += "$"
        code += DIGITS[i]
    if hw:
        code += "$"
    return code


if __name__ == '__main__':
    #aa = code2vec("`abc`")
    #from utils import uniform_utils
    #for d in DIGITS:
    #    r = uniform_utils.str_uniform(d)
    #    if r != d:
    #        print d, r
    #print "done"
    x = code2vec('D.\sqrt(x^3y)')
    print(vec2code(x))
    

