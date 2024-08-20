import re
import json

# ┏┓
# \mathrm
# \text

symbol2token_dict = {}
for idx, line in enumerate(open("/mnt/server_data2/data/seq_latex/latex2token.txt")):
    symbol, token = line.strip().split("\t")
    symbol2token_dict[symbol] = token

token2symbol_dict = {}
for idx, line in enumerate(open("/mnt/server_data2/data/seq_latex/token2latex.txt")):
    token, symbol = line.strip().split("\t")
    token2symbol_dict[token] = symbol


# data = {}
# data["symbol2token"] = symbol2token_dict
# data["token2symbol"] = token2symbol_dict
# with open("latex_simplify_rule.json", "w") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)
# assert 0
def simplify_single_char_elements(latex_str):
    # 正则表达式匹配 \sqrt{...}
    pattern_sqrt = r"\\sqrt\{([^}]*)\}"

    # 正则表达式匹配 _{...} 和 ^{...}
    pattern_subsup = r"(_|\^)\{([^}]*)\}"

    # 正则表达式匹配 \frac{...}{...}
    pattern_frac = r"\\frac\{([^}]*)\}\{([^}]*)\}"

    # 使用正则表达式替换函数
    def replace_func(match, is_sqrt=False):
        if is_sqrt:
            content = match.group(1)
            if len(content) == 1:
                return f"\\sqrt{{{content}}}".replace('{', '').replace('}', '')
            else:
                return match.group(0)
        else:
            operator = match.group(1)
            content = match.group(2)
            if len(content) == 1:
                return f"{operator}{{{content}}}".replace('{', '').replace('}', '')
            else:
                return match.group(0)

    # 使用正则表达式替换函数
    def replace_frac(match):
        numerator = match.group(1)
        denominator = match.group(2)

        # 如果分子或分母只有一个字符，去掉花括号
        if len(numerator) == 1 and len(denominator) == 1:
            return f"\\frac{numerator}{denominator}"
        elif len(numerator) == 1:
            return f"\\frac{numerator}{{{denominator}}}"
        elif len(denominator) == 1:
            return f"\\frac{{{numerator}}}{denominator}"
        else:
            return match.group(0)

    # 替换所有的 \frac{...}{...}
    latex_str = re.sub(pattern_frac, replace_frac, latex_str)

    # 替换所有的 \sqrt{...}
    latex_str = re.sub(pattern_sqrt, lambda m: replace_func(m, True), latex_str)

    # 替换所有的 _{...} 和 ^{...}
    latex_str = re.sub(pattern_subsup, lambda m: replace_func(m), latex_str)

    return latex_str


def replace_braces(latex_str):
    # 使用正则表达式匹配非转义的大括号
    # 先替换左大括号
    latex_str = re.sub(r'(?<!\\)\{', 'ι', latex_str)
    # 再替换右大括号
    latex_str = re.sub(r'(?<!\\)\}', '┓', latex_str)
    # 替换转义的大括号
    latex_str = latex_str.replace('\{', '{').replace('\}', '}')
    return latex_str


def encode_latex(latex_str):
    latex_str = simplify_single_char_elements(latex_str)

    for symbol, token in symbol2token_dict.items():
        latex_str = latex_str.replace(symbol, token)
    latex_str = replace_braces(latex_str)
    return latex_str


def decode_latex(latex_str):
    latex_str = latex_str.replace('{', '\{').replace('}', '\}')
    for token, symbol in token2symbol_dict.items():
        latex_str = latex_str.replace(token, symbol)
    latex_str = latex_str.replace('ι', '{').replace('┓', '}')
    return latex_str
# -*-coding:utf-8-*-
