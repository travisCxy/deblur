#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

STD_LABEL_LEN = 30
VTL_LABEL_LEN = 30
VTL_LABEL_LEN_2 = 15

STD_OFFSET = 0
VTL_OFFSET = 2
VTL_OFFSET_2 = 0

VTL_DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.',
           '+', '-', '*', '/', '(', ')', '=',  '@', '!', '|'
           ]
VTL_DIGITS_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',',
            '+', '-', '*', '/', '(', ')', '@', '!', '_',
           ]

VTL_LINE_MARK = '_'
VTL_LINE_SPLITTER = '|'

STD_DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',',
          '+', '-', '*', '/', '(', ')', '=', '@', '!', '#', '>', '<', '%', ':', '~', 
          '[yuan]', '[jiao]', '[fen]', '[shi1]', '[miao]', '[xiao]', '[zhong]',
          '[yi]', '[wan]', '[qian]','[bai]', '[shi2]','[ge]',
          '[mi]', '[hao]', '[li2]', '[gong]', '[li3]', '[mi^2]', '[mi^3]',
          '[ke]', '[dun]', '[ping]','[fang]', '[qing]', '[li4]', '[sheng]',
          '[mm]', '[cm]', '[dm]', '[m]','[km]',
          '[mm^2]', '[cm^2]', '[dm^2]', '[m^2]', '[km^2]',
          '[mm^3]', '[cm^3]', '[dm^3]', '[m^3]', '[ml]', '[l]',
          '[mg]', '[g]', '[kg]', '[t]',
          'x','y'
          ]

FRAC_SYMBOL = '~'

def parse_tokens(s):
    tokens = []
    unit = None
    for c in s:
        if c == '[':
            if unit != None:
                raise ValueError(s)
            else:
                unit = c;
        elif c==']':
            if unit == None:
                raise ValueError(s)
            else:
                unit = unit + c
                tokens.append(unit)
                unit = None
        else:
            if unit == None:
                tokens.append(c)
            else:
                unit = unit + c
    return tokens

def is_frac(v1):
    i = v1.find(FRAC_SYMBOL)
    return i != -1

def frac_expr(v1, v2, v3):
    arr1 = v2.split(FRAC_SYMBOL)
    arr2 = v3.split(FRAC_SYMBOL)
    ret = ""
    p = 0
    cnt = 1
    while True:
        i = v1.find(FRAC_SYMBOL,p)
        if i == -1:
            break
        ret += v1[p:i]
        if cnt >= len(arr1) or cnt >= len(arr2):
            break
        f1 = arr1[cnt]
        f2 = arr2[cnt]
        if ret.count('$') % 2 == 1:
            f1 = f1.replace("$", "")
            f2 = f2.replace("$", "")
        ret += "\\frac{" + f1 + "}" + "{" + f2 + "}"
        cnt += 1
        p = i + 1
    ret += v1[p:]
    return ret

class LabelCoder:
    def __init__(self, digit_set, code_len, offset):
        self.digit_set = digit_set
        self.code_len = code_len
        self.offset = offset
        self.code_map = {}
        for i in xrange(len(digit_set)):
            self.code_map[digit_set[i]] = i

    def encode(self, label):
        vec = []
        hw = False
        tokens = parse_tokens(label)
        for c in tokens:
            if c == '$':
                hw = not hw
                continue
            n = self.code_map[c]
            if hw:
                n += len(self.digit_set)
            vec.append(n)
        vec.extend([-1] * (self.code_len - len(vec)))
        return vec

    def decode(self, code_vec):
        code_vec = [v - self.offset for v in code_vec]
        code = ''
        hw = False
        for i in code_vec:
            if i == -1:
                break
            if i >= len(self.digit_set):
                i -= len(self.digit_set)
                if not hw:
                    hw = True
                    code += "$"
            elif hw:
                hw = False
                code += "$"
            code += self.digit_set[i]
        if hw:
            code += "$"
        return code

    def decode_mul(self, indice, values):
        ret = {}
        last_index = 0
        vec = []
        for ind,i in enumerate(indice):
            if i[0] != last_index:
                ret[last_index] = self.vec2code(vec)
                vec = []
                last_index = i[0]
            vec.append(values[ind])
        ret[last_index] = self.vec2code(vec)
        return ret

if __name__ == '__main__':
    coder = LabelCoder(VTL_DIGITS_2, VTL_LABEL_LEN_2, VTL_OFFSET_2)
    vec = coder.encode('$111,1111$')
    print vec
    print coder.decode(vec)
    
    