#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from projects.hw import common as hw

MAX_SEQ_LEN = 60


PLACE_HOLDER = u"\u02fd"
parent_dir = os.path.dirname(__file__)

DIGITS = []
for w in codecs.open(os.path.join(parent_dir, "data/synset.txt"), encoding="utf-8"):
    DIGITS.append(w.rstrip())

DIGITS = DIGITS[:2500]


hw_digits, _ = hw.create_code_dic(ignore_repeat=True)
for hw in hw_digits:
    if hw not in DIGITS:
        DIGITS.append(hw)

if 'O' not in DIGITS:
    DIGITS.append('O')


# Why Add a space HERE??, FOR ENGISH
if ' ' not in DIGITS:
    DIGITS.append(' ')

cs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
for c in cs:
    if c not in DIGITS:
        DIGITS.append(c)


NUM_CLASSES = len(DIGITS) + 1
_code_map = {}
for i in range(len(DIGITS)):
    _code_map[DIGITS[i]] = i


def get_code(s):
    return _code_map[s]


def ignore_hw(s):
    ret = ""
    ignore = False
    for c in s:
        if c == '$':
            ignore = not ignore
            continue
        else:
            if not ignore:
                ret += c
    return ret


def frac_count(s):
    frac_lens =[0]
    current_frac = ""
    is_frac = False
    for c in s:
        if c == '`':
            is_frac = not is_frac
            if not is_frac:
                frac_lens.append(len(current_frac))
                current_frac = ""
            continue
        else:
            if is_frac:
                current_frac += c
    return max(frac_lens)


def code2vec(s):
    s = s.replace(u'＋', u'+')
    s = s.replace(u'－', u'-')
    s = s.replace(u'＝', u'=')
    s = s.replace(u'：', u':')
    s = s.replace(u'？', u'?')
    s = s.replace(u'；', u':')
    s = s.replace(u'：', u':')
    s = s.replace(u'＞', u'>')
    s = s.replace(u'＜', u'<')
    s = s.replace(u'（', u'(')
    s = s.replace(u'）', u')')
    s = s.replace(u'】', u']')
    s = s.replace(u'【', u'[')
    s = s.replace(u'”', u'"')
    s = s.replace(u'“', u'"')

    s = ignore_hw(s)

    s = s.replace(u'……', u'…')
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
    

