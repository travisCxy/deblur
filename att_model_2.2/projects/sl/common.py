#!/usr/bin/env python
# -*- coding: utf-8 -*-

MAX_SEQ_LEN = 90
FRAC_SYMBOL = '~'

GO_SYMBOL = 1
END_SYMBOL = 2
PAD_SYMBOL = 0

DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
          "+", "-", "*", "/", "(", ")", "@", "!", "=", ".", ",", "、", "|",
          "~", ":",
          "时", "分", "秒", "钟", "小"]

# trick for easy handle different expression of handwrite in attention.
DIGITS_COPY = list(DIGITS)
for d in DIGITS_COPY:
    DIGITS.append("$"+d+"$")

_code_map = {}
for ind, num in enumerate(DIGITS):
    _code_map[num] = ind

NUM_CLASSES = len(DIGITS) + 3
print(NUM_CLASSES)

def parse_frac(s):
    l1 = ""
    p = 0
    while True:
        i = s.find("\\frac", p)
        if i == -1:
            break
        l1 += s[p:i] + FRAC_SYMBOL
        assert s[i + 5] == '{'
        j = s.find("}", i + 6)
        assert s[j + 1] == '{'
        k = s.find("}", j + 2)
        if s.count('$', 0, i) % 2 == 1:
            l1 += "$" + s[i + 6:j] + "$" + FRAC_SYMBOL
            l1 += "$" + s[j + 2:k] + "$" + FRAC_SYMBOL
        else:
            l1 += s[i + 6:j] + FRAC_SYMBOL
            l1 += s[j + 2:k] + FRAC_SYMBOL
        p = k + 1
    l1 += s[p:]

    return l1


def parse_tokens(s):
    tokens = []
    unit = None

    for c in s:
        if c == '[':
            if unit is not None:
                raise ValueError(s)
            else:
                unit = c
        elif c == ']':
            if unit is None:
                raise ValueError(s)
            else:
                unit = unit + c
                tokens.append(unit)
                unit = None
        else:
            if unit is None:
                tokens.append(c)
            else:
                unit = unit + c
    return tokens


def code2vec(s):
    vec = []
    hw = False
    s = parse_frac(s)
    tokens = parse_tokens(s)
    for c in tokens:
        if c == '$':
            hw = not hw
            continue
        n = _code_map[c]
        if hw and c != FRAC_SYMBOL:
            n = _code_map["$" + c + "$"]

        vec.append(n)
    vec.extend([-1] * (MAX_SEQ_LEN - len(vec)))
    return vec


def code2attvec(s):
    vec = code2vec(s)
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
    print(parse_frac("\\frac{2}{5}"))


