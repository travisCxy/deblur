#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from projects.hw import common as hw

UNIFORM = True
# UNIFORM = False

GO_SYMBOL = 1
END_SYMBOL = 2
PAD_SYMBOL = 0
VALID_TOKEN_OFFSET = 3

MAX_SEQ_LEN = 96  # 92  longer to handle long frac.


PLACE_HOLDER = u"\u02fd"
parent_dir = os.path.dirname(__file__)

DIGITS = []
for w in codecs.open(os.path.join(parent_dir, "data/synset.txt"), encoding="utf-8"):
    DIGITS.append(w.rstrip())

DIGITS = DIGITS[:6000]

for w in codecs.open(os.path.join(parent_dir, "data/synset2.txt"), encoding="utf-8"):
    DIGITS.append(w.rstrip())

DIGITS = DIGITS[:7000]

for w in codecs.open(os.path.join(parent_dir, "data/synset3.txt"), encoding="utf-8"):
    DIGITS.append(w.rstrip())

DIGITS = DIGITS[:8000]

hw_digits, _ = hw.create_code_dic(ignore_repeat=True)
for hw in hw_digits:
    if hw not in DIGITS:
        DIGITS.append(hw)

if 'O' not in DIGITS:
    DIGITS.append('O')


if PLACE_HOLDER in DIGITS:
    raise ValueError("Place holder char in digits")
DIGITS.append(PLACE_HOLDER)
    
if '$' not in DIGITS:
    DIGITS.append('$')

# Why Add a space HERE??, FOR ENGISH
if ' ' not in DIGITS:
    DIGITS.append(' ')

cs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
for c in cs:
    if c not in DIGITS:
        DIGITS.append(c)

len1 = len(DIGITS)
cnt = 0
for w in codecs.open(os.path.join(parent_dir, "data/synset_cht_main.txt"), encoding="utf-8"):
    w = w.rstrip()
    if w not in DIGITS:
        DIGITS.append(w)
    cnt += 1
    if cnt >1500:
        break
print('total chars:%d' % (len(DIGITS) ))
print('-'*40)

NEW_DIGITS = []
with open(os.path.join(parent_dir, 'data/new_digits.txt'), 'r') as f:
    NEW_DIGITS = [line.strip() for line in f]
DIGITS = NEW_DIGITS
assert '' in DIGITS
print('new chars:%d' % (len(NEW_DIGITS) ))
print('-'*40)
# with open(os.path.join(parent_dir, 'data/synset_ch_cht.txt'), 'w') as f:
#     for w in DIGITS:
#         f.write(w + '\n')

NUM_CLASSES = len(DIGITS) + 3
_code_map = {}
for i in range(len(DIGITS)):
    _code_map[DIGITS[i]] = i


def get_code(s):
    return _code_map[s]


def ignore_hw(s):
    ret = ""
    ignore = False
    for ind, c in enumerate(s):
        if c == '$':
            if ind - 1 < 0 or s[ind - 1] != '\\':
                ignore = not ignore
                continue
            else:
                if not ignore:
                    ret += c
        else:
            if not ignore:
                ret += c
    return ret


def frac_count(s):
    frac_lens = [0]
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
    #s = s.replace(u' ', u'')
    s = s.replace(u'＋', u'+')
    s = s.replace(u'－', u'-')
    s = s.replace(u'—', u'-')
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

    # replace '_____' with '_', '     ' with ' '
    rs = ""
    repeat_chars = {
        '_': True,
        ' ': True,
    }
    for c in s:
        if c in repeat_chars:
            if repeat_chars[c]:
                rs += c
            for k in repeat_chars.keys():
                repeat_chars[k] = k != c
        else:
            rs += c
            for k in repeat_chars.keys():
                repeat_chars[k] = True
    s = rs

    if not UNIFORM:
        s = ignore_hw(s)
    #print(s)
    s = s.replace(u'……', u'…')
    vec = []
    for c in s:
        # if UNIFORM:
        #     if c not in _code_map:
        #         c = PLACE_HOLDER
        #     n = _code_map[c]
        #     vec.append(n)
        # else:
        if c in _code_map:
            n = _code_map[c]
            vec.append(n)
    #print(vec)
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


def code2attvec(s):
    vec = code2vec(s)
    ret = [GO_SYMBOL]
    for v in vec:
        if v != -1:
            ret.append(v + VALID_TOKEN_OFFSET)
        else:
            break
    ret.append(END_SYMBOL)
    if len(ret) >= MAX_SEQ_LEN:
        raise Exception("exceed MAX_SEQ_LEN")
    ret.extend([PAD_SYMBOL] * (MAX_SEQ_LEN - len(ret)))
    return ret


def attvec2code(vec):
    ret = []
    for v in vec:
        if v == GO_SYMBOL:
            continue
        if v == END_SYMBOL:
            break
        if v == PAD_SYMBOL:
            break
        else:
            ret.append(v - VALID_TOKEN_OFFSET)
    return vec2code(ret)


if __name__ == '__main__':
    #aa = code2vec("`abc`")
    #from utils import uniform_utils
    #for d in DIGITS:
    #    r = uniform_utils.str_uniform(d)
    #    if r != d:
    #        print d, r
    #print "done"
    print(code2vec('A____A   $1_23$'))
    print(code2vec('A _ A$1\\$23$'))
    print(code2vec('A\\$A$123$EE\\$'))

