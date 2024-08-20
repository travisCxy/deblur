#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import json

MAX_SEQ_LEN = 50
FRAC_SYMBOL = '`'

GO_SYMBOL = 1
END_SYMBOL = 2
PAD_SYMBOL = 0

NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '%', '@']
NUMBERS.append(FRAC_SYMBOL)
_number_code_map = {}
for num in NUMBERS:
    _number_code_map[num] = NUMBERS.index(num)

META_DIGITS = [u'姓', u'名', u'年', u'班', u'级', u'学', u'号',u'校', u':']
parent_dir = os.path.dirname(__file__)

DIGITS = []
_code_map = {}

def create_code_dic(ignore_repeat=False):
    codes = []
    code_map = {}
    
    code_lines = []
    repeat_dic = {}
    read_code = True
    for line in codecs.open(os.path.join(parent_dir, "data/synset_hw2.txt"), encoding="utf-8"):
        line = line.rstrip()
        if len(line) == 0:
            continue
        if "##" == line[:2]:
            read_code = False
            continue
        if read_code:
            code_lines.append(line)
        else:
            pairs = line.split(" ")
            for pair in pairs:
                if len(pair) > 0:
                    for c in pair:
                        repeat_dic[c] = pair[0]

    if ignore_repeat:
        repeat_dic = {}

    for line in code_lines:
        line = line.replace(" ", "")
        for c in line:
            code = repeat_dic.get(c, c)
            if code not in codes:
                codes.append(code)
            code_map[c] = codes.index(code)

    with codecs.open(os.path.join(parent_dir, "data/hw_summary.txt"), encoding="utf-8") as fp:
        high_freqs = json.load(fp, encoding="utf-8")
        high_freqs = high_freqs[:500]
        high_freqs = [x[0] for x in high_freqs]
        for m in META_DIGITS:
            if m not in high_freqs:
                high_freqs.append(m)

    with codecs.open(os.path.join(parent_dir, "data/hw_summary2.txt"), encoding="utf-8-sig") as fp:
        for ind, line in enumerate(fp):
            if ind >= 500:
                break
            new_char = line.rstrip().split(",")[0]
            if new_char not in high_freqs:
                high_freqs.append(new_char)

    for c in high_freqs:
        if c == FRAC_SYMBOL:
            continue
        code = repeat_dic.get(c, c)
        if code not in codes:
            codes.append(code)
        code_map[c] = codes.index(code)

    if FRAC_SYMBOL in codes:
        raise Exception("frac symbol should not in codes")
    
    codes.append(FRAC_SYMBOL)
    code_map[FRAC_SYMBOL] = codes.index(FRAC_SYMBOL)
    return codes, code_map

DIGITS, _code_map = create_code_dic()

NUM_CLASSES = len(DIGITS) + 3

def code2vec(s, code_map=_code_map):
    vec = []
    s = s.replace('$', '')
    s = s.replace('`dot', '`')
    s = s.replace(u'……', u'…')
    s = s.replace(u'＋', u'+')
    s = s.replace(u'－', u'-')
    s = s.replace(u'＝', u'=')
    s = s.replace(u'：', u':')
    for c in s:
        if c in code_map:
            n = code_map[c]
            vec.append(n)
    if len(vec) >= MAX_SEQ_LEN:
        raise Exception("exceed MAX_SEQ_LEN")
    vec.extend([-1] * (MAX_SEQ_LEN - len(vec)))
    return vec

def code2attvec(s, code_map=_code_map):
    vec = code2vec(s, code_map)
    ret = [GO_SYMBOL]
    for v in vec:
        if v != -1:
            ret.append(v + 3)
        else:
            break
    ret.append(END_SYMBOL)
    if len(ret) >= MAX_SEQ_LEN:
        raise Exception("exceed MAX_SEQ_LEN")
    ret.extend([PAD_SYMBOL] * (MAX_SEQ_LEN - len(ret)))
    return ret

def vec2code(vec):
    code = ''
    for i in vec:
        if i == -1:
            break
        code += DIGITS[i]
    return code

def attvec2code(vec):
    ret = []
    for v in vec:
        if v == GO_SYMBOL:
            continue
        if v == END_SYMBOL:
            break
        if v == PAD_SYMBOL:
            break
            #raise Exception("meet pad symbol in attvec!")
        else:
            ret.append(v-3)
    return vec2code(ret)

def contains_frac(codes):
    start = -1
    for ind, code in enumerate(codes):
        if code == FRAC_SYMBOL:
            if start == -1:
                start =  ind
            else:
                part = codes[start: ind]
                if '/' in part:
                    return True
                else:
                    start = -1
    return False

if __name__ == '__main__':
#    code2vec(u'0.`dot18`')
#    print reencoding(u"70~89")
#    l1, l2, l3, l4 = frac_code2vec("`222/22`")
#    print l1
#    print l2
#    print l3
#    print l4
    #from utils import uniform_utils
    #print len(DIGITS), len(_code_map)
    #for d in DIGITS:
    #    r = uniform_utils.str_uniform(d)
    #    if r != d:
    #        print d, r
    #print "done"
    
    print(contains_frac('`111/2`'))

