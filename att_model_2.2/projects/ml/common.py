#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SYMBOLS = []
for s in open(os.path.join(CURRENT_DIR, "data/synset.txt")):
    SYMBOLS.append(s.rstrip())

_code_map = {}
for i in range(len(SYMBOLS)):
    _code_map[SYMBOLS[i]] = i


MAX_SEQ_LEN = 80
PAD_SYMBOL = 0
GO_SYMBOL = 1
END_SYMBOL = 2

NUM_CLASSES = len(SYMBOLS) + 3

INPUT_IMG_HEIGHT = 120
INPUT_IMG_WIDTH = 384

# INPUT_IMG_HEIGHT_2 = 160
# INPUT_IMG_WIDTH_2 = 192

RATIO_THRESHOLD = 0.5

STRUCT_SYMBOLS = [67, 68, 79, 80]


def get_code(s):
    return _code_map[s]

'''
no_sp_list = ['[yuan]', '[jiao]', '[fen]', '[shi1]', '[miao]', '[xiao]', '[zhong]',
              '[yi]', '[wan]', '[qian]', '[bai]', '[shi2]', '[ge]',
              '[mi]', '[hao]', '[li2]', '[gong]', '[li3]', '[mi^2]', '[mi^3]',
              '[ke]', '[dun]', '[ping]', '[fang]', '[qing]', '[li4]', '[sheng]',
              '[mm]', '[cm]', '[dm]', '[m]', '[km]',
              '[mm^2]', '[cm^2]', '[dm^2]', '[m^2]', '[km^2]',
              '[mm^3]', '[cm^3]', '[dm^3]', '[m^3]', '[ml]', '[l]',
              '[mg]', '[g]', '[kg]', '[t]',

              ]
'''

def code2vec_lagacy(s):
    # extract frac
    s = s.replace("$", "")
    # for item in no_sp_list:
    #     if s.find(item) != -1:
    #         raise ValueError("invalid string")
    s = " ".join(s)
    s = s.replace("*", "\\times").replace("/", "\\div")
    s = s.replace("\\ f r a c", "\\frac")
    return code2vec(s), s


def code2vec(s):
    vec = []
    tokens = s.split(" ")
    # print(s)
    # print(tokens)
    for c in tokens:
        # print(c)
        vec.append(_code_map[c])
    if len(vec) >= MAX_SEQ_LEN:
        raise ValueError("exceed max limit " + s)
    vec.extend([-1] * (MAX_SEQ_LEN - len(vec)))
    return vec


def vec2code(vec):
    code = []
    for i in vec:
        code.append(SYMBOLS[i])
    return " ".join(code)


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


def seq_wrap(vec):
    ret = [GO_SYMBOL]
    for c in vec:
        if c == -1:
            break
        ret.append(c + 3)
    if len(ret) >= MAX_SEQ_LEN:
        raise ValueError(vec)
    ret.extend([PAD_SYMBOL] * (MAX_SEQ_LEN - len(ret)))
    return ret


def seq_vec2code(vec):
    new_vec = []
    for c in vec:
        if c == END_SYMBOL:
            break
        elif c == GO_SYMBOL or c == PAD_SYMBOL:
            continue
        new_vec.append(c - 3)
    return vec2code(new_vec)


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
            ret.append(v - 3)
    return vec2code(ret)


def seq_vec2code_with_score(vec, score):
    new_vec = []
    p = 0
    for c, s in zip(vec, score):
        if c == END_SYMBOL:
            break
        if c == GO_SYMBOL:
            continue
        p += s
        new_vec.append(c - 2)
    prob = math.exp(p / len(new_vec))
    return vec2code(new_vec), prob


if __name__ == '__main__':
    # s = "(\\frac{4}{5}-\\frac{4}{9}*\\frac{3}{10})/\\frac{2}{3}$=1$ [ping]"
    s = "(\\frac{4}{5}-\\frac{4}{9}*\\frac{3}{10})/\\frac{2}{3}=1s[pt]"
    s = "{}"
    vec, s = code2vec_lagacy(s)
    print(s)
    # print(SYMBOLS)
    print(vec)

    vec = [item + 3 for item in vec]
    seq_s = seq_vec2code(vec)
    print(seq_s)
    # fp = open("/mnt/data2/hwhelper_train/data/mathlens/raw/ml.lst")
    # for line in fp:
    #     arr = line.rstrip().split(",")
    #     lbl = seq_wrap(code2vec(arr[1]))
