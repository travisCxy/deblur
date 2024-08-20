# -*- coding: utf-8 -*-

DBC = [
    u'０' , u'１' , u'２' , u'３' , u'４' ,
    u'５' , u'６' , u'７' , u'８' , u'９' ,
    u'Ａ' , u'Ｂ' , u'Ｃ' , u'Ｄ' , u'Ｅ' ,
    u'Ｆ' , u'Ｇ' , u'Ｈ' , u'Ｉ' , u'Ｊ' ,
    u'Ｋ' , u'Ｌ' , u'Ｍ' , u'Ｎ' , u'Ｏ' ,
    u'Ｐ' , u'Ｑ' , u'Ｒ' , u'Ｓ' , u'Ｔ' ,
    u'Ｕ' , u'Ｖ' , u'Ｗ' , u'Ｘ' , u'Ｙ' ,
    u'Ｚ' , u'ａ' , u'ｂ' , u'ｃ' , u'ｄ' ,
    u'ｅ' , u'ｆ' , u'ｇ' , u'ｈ' , u'ｉ' ,
    u'ｊ' , u'ｋ' , u'ｌ' , u'ｍ' , u'ｎ' ,
    u'ｏ' , u'ｐ' , u'ｑ' , u'ｒ' , u'ｓ' ,
    u'ｔ' , u'ｕ' , u'ｖ' , u'ｗ' , u'ｘ' ,
    u'ｙ' , u'ｚ' , u'－' , u'　' , u'：' ,
    u'．' , u'，' , u'／' , u'％' , u'＃' ,
    u'！' , u'＠' , u'＆' , u'（' , u'）' ,
    u'＜' , u'＞' , u'＂' , u'＇' , u'？' ,
    u'［' , u'］' , u'｛' , u'｝' , u'＼' ,
    u'｜' , u'＋' , u'＝' , u'＿' , u'＾' ,
    u'＄' , u'～' , u'｀' , u'；'
]

SBC = [ # banjiao
    u'0', u'1', u'2', u'3', u'4',
    u'5', u'6', u'7', u'8', u'9',
    u'A', u'B', u'C', u'D', u'E',
    u'F', u'G', u'H', u'I', u'J',
    u'K', u'L', u'M', u'N', u'O',
    u'P', u'Q', u'R', u'S', u'T',
    u'U', u'V', u'W', u'X', u'Y',
    u'Z', u'a', u'b', u'c', u'd',
    u'e', u'f', u'g', u'h', u'i',
    u'j', u'k', u'l', u'm', u'n',
    u'o', u'p', u'q', u'r', u's',
    u't', u'u', u'v', u'w', u'x',
    u'y', u'z', u'-', u' ', u':',
    u'.', u',', u'/', u'%', u'#',
    u'!', u'@', u'&', u'(', u')',
    u'<', u'>', u'"', u'\'',u'?',
    u'[', u']', u'{', u'}', u'\\',
    u'|', u'+', u'=', u'_', u'^',
    u'$', u'~', u'`', u';'
]

CODE_MAP = {}
for k, v in zip(DBC, SBC):
    CODE_MAP[k]=v


def str_uniform(org_str):
    ret = u""

    assert isinstance(org_str, str)
    
    for c in org_str:
        r = CODE_MAP.get(c, c)
        ret += r

    return ret



