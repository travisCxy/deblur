#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',',
          '+', '-', '*', '/', '(', ')', '=', '@', '!', '#', '>', '<', '%', ':', '~',
          #'[yuan]', '[jiao]', '[fen]', '[shi1]', '[miao]', '[xiao]', '[zhong]',
          '元', '角', '分', '时', '秒', '小', '钟',
          #'[yi]', '[wan]', '[qian]', '[bai]', '[shi2]', '[ge]',
          '亿', '万', '千', '百', '十', '个',
          #'[yi2]', '[er]', '[san]', '[si]', '[wu]', '[liu]', '[qi]', '[ba]', '[jiu]', '[de]',
          '一', '二', '三', '四', '五', '六', '七', '八', '九', '得',
          #'[ling]', '[fu]', '[zheng]', '[dian]', '[shang]', '[xia]', '[she]', '[shi3]', '[du2]', # templa
          '零', '负', '正', '点', '上', '下', '摄', '氏', '度',
          #'[du3]', '[xie]', '[zuo]', '[you]',
          '读', '写', '作', '又',
          #'[zhi]', '[cheng]', '[chu]', '[yi3]', '[deng]', '[yu]', '[jia]', '[jian]', '[sheshidu]',
          '之', '乘', '除', '以', '等', '于', '加', '减',
          #'[liang]', '[yu2]',
          '两', '余',
          #'[jiao2]', '[dun2]', '[rui]', '[zhi2]', '[zhou]',
          '钝', '锐', '直', '周',
          #'[nian]', '[yue]', '[ri]', '[xing]', '[qi2]', '[tian]', '[zhe]', '[cheng2]',
          '年', '月', '日', '星', '期', '天', '折', '成',
          #'[chang]', '[mian]', '[ji]',
          '长', '宽', '面', '积',

          #'[shu]', '[bei]', '[yin]', '[shang2]', '[xing2]',
          '数', '被', '因', '商', '形',
          #'[xing3]', '[bian]', '[ban]', '[jing]', '[ti]',
          '行', '边', '半', '径', '体',
          #'[he]', '[cha]', '[gao]', '[di]', '[de2]',
          '和', '差', '高', '底', '的',
          
          #'[mi]', '[hao]', '[li2]', '[gong]', '[li3]',  # '[mi^2]', '[mi^3]',
          '米', '毫', '厘', '公', '里',
          #'[ke]', '[dun]', '[ping]', '[fang]', '[qing]', '[li4]', '[sheng]',
          '克', '吨', '平', '方', '顷', '立', '升',

          '[sheshidu]', '[jiao2]',
          '[mm]', '[cm]', '[dm]', '[m]', '[hm]', '[km]',

          '[ml]', '[l]',
          '[mg]', '[g]', '[kg]', '[t]',
          '[pai]', '[du]',  # circle
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
          'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
          'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          '[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]',

          '[', ']', '{', '}',
          '[dot0]', '[dot1]', '[dot2]', '[dot3]', '[dot4]', '[dot5]', '[dot6]', '[dot7]', '[dot8]', '[dot9]',
          ]

MAX_SEQ_LEN = 30
FRAC_SYMBOL = '~'
NUM_CLASSES = 2 * len(DIGITS) + 1

_code_map = {}
for i in range(len(DIGITS)):
    _code_map[DIGITS[i]] = i

SAME_CODES = [('m', '[m]'), ('l', '[l]'), ('g', '[g]'), ('t', '[t]')]
for pair in SAME_CODES:
    _code_map[pair[0]] = _code_map[pair[1]]


def parse_frac(s):
    l1 = ""
    l2 = ""
    l3 = ""
    p = 0
    # print(s)
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
            l2 += FRAC_SYMBOL + "$" + s[i + 6:j] + "$"
            l3 += FRAC_SYMBOL + "$" + s[j + 2:k] + "$"
        else:
            l3 += FRAC_SYMBOL + s[j + 2:k]
            l2 += FRAC_SYMBOL + s[i + 6:j]
        p = k + 1
    l1 += s[p:]
    if l2 == "":
        l2 = FRAC_SYMBOL
    if l3 == "":
        l3 = FRAC_SYMBOL
    return l1, l2, l3


def get_code(s):
    return _code_map[s]


def operation_chars_in_unit(unit):
    ops = ['+', '-', "*", '/', ',', '$']
    for op in ops:
        if op in unit:
            return True
    if len(unit) == 2:
        return True
    return False


def parse_tokens(s):
    tokens = []

    unit = None
    for c in s:
        if c == '[':
            if unit is not None:
                for u in unit:
                    tokens.append(u)
            unit = c
        elif c == ']':
            if unit is None:
                tokens.append(c)
            else:
                unit = unit + c
                if operation_chars_in_unit(unit):
                    for u in unit:
                        tokens.append(u)
                else:
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
    tokens = parse_tokens(s)
    for c in tokens:
        if c == '$':
            hw = not hw
            continue
        n = _code_map[c]
        if hw:
            n += len(DIGITS)

        vec.append(n)
    vec.extend([-1] * (MAX_SEQ_LEN - len(vec)))
    return vec


def get_bbox(im):
    m1 = np.sum(im, axis=0)
    t = np.where(m1 > 255 * 3)[0]
    x1 = t[0]
    x2 = t[-1]
    m2 = np.sum(im, axis=1)
    t = np.where(m2 > 255 * 3)[0]
    y1 = t[0]
    y2 = t[-1]
    return x1, x2 - x1, y1, y2 - y1


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


def vec2code_mul(indice, values):
    ret = {}
    last_index = 0
    vec = []
    for ind, i in enumerate(indice):
        if i[0] != last_index:
            ret[last_index] = vec2code(vec)
            vec = []
            last_index = i[0]
        vec.append(values[ind])
    ret[last_index] = vec2code(vec)
    return ret


def frac_exp(v1, v2, v3):
    if v1 is None:
        return None

    if v1.find(FRAC_SYMBOL) == -1:
        return v1
    p = 0
    ret = ""
    arr1 = v2.split(FRAC_SYMBOL)
    arr2 = v3.split(FRAC_SYMBOL)
    cnt = 0
    while True:
        i = v1.find(FRAC_SYMBOL, p)
        if i == -1:
            break
        ret += v1[p:i] + "\\frac{"
        cnt += 1

        if cnt < len(arr1) and cnt < len(arr2):
            f1 = arr1[cnt]
            f2 = arr2[cnt]
            if v1.count('$', 0, i) % 2 == 1:
                f1 = f1.replace("$", "")
                f2 = f2.replace("$", "")
            ret += f1 + "}{" + f2 + "}"
        p = i + 1
    if p < len(v1):
        ret += v1[p:]
    return ret


def reencoding(s):
    l1, l2, l3 = parse_frac(s)
    v1 = vec2code(code2vec(l1))
    v2 = vec2code(code2vec(l2))
    v3 = vec2code(code2vec(l3))
    return frac_exp(v1, v2, v3)


def contains_unknown_token(s):
    l1, l2, l3 = parse_frac(s)
    t1 = parse_tokens(l1)
    t2 = parse_tokens(l2)
    t3 = parse_tokens(l3)
    return 'x' in t1 or 'x' in t2 or 'x' in t3 or 'y' in t1 or 'y' in t2 or 'y' in t3


if __name__ == '__main__':
    label = '\\frac{3}{8}+\\frac{11}{20}=\\frac{($15$)}{40}+\\frac{($22$)}{40}=\\frac{($15$)+($22$)}{40}=\\frac{($37$)}{40}'
    l1, l2, l3 = parse_frac(label)
    v1 = code2vec(l1)
    v2 = code2vec(l2)
    v3 = code2vec(l3)
    print(v1)
    print(v2)
    print(v3)

