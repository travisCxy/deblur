import itertools
from uniform_utils import str_uniform

from projects.cht import common
import cv2
from zhconv import convert
from projects.cht.latex_encode import decode_latex
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


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_rec_eval(gtruth_str, pre_str, bDebug=False):
    ed = levenshtein(gtruth_str, pre_str)
    acc = max(0.0, 1 - float(ed) / len(gtruth_str))

    if bDebug:
        print("gtruth_str, pre_str, ed, acc: ", gtruth_str, pre_str, ed, acc)

    return ed, acc


def remove_dollar(l):
    if len(l) < 3:
        return l
    if l[0] == '$' and l[-1] == '$':
        return l[1:-1]
    return l

def cast_simple_character(ch):
    ret = ''
    for c in ch:
        if c == convert(c, 'zh-cn') or c in ['「', '」']:
            continue
        ret += c
    return ret


class Evaluator:
    def __init__(self):
        self.total_acc = 0.0
        self.total_count = 0
        self.miss_count = 0
        self.ac_count = 0
        #self.fout = open('error.txt', 'a')
        pass

    def record_step(self, outputs, inputs, test_cht=False, test_encode=False):
        preds = outputs["strings"]
        gt_labels = inputs["labels"]
        images = inputs["images"]
        for ind, image, pred in zip(itertools.count(), images.numpy(), preds):
            l = common.attvec2code(gt_labels[ind])
            l = str_uniform(l)
            l = remove_dollar(l)

            top1_pred = pred[0].numpy().decode('utf-8')
            top1_pred = remove_dollar(top1_pred)
            if test_encode:
                l = decode_latex(l)
                top1_pred = decode_latex(top1_pred)

            if test_cht:
                l = cast_simple_character(l)
                top1_pred = convert(top1_pred, 'zh-hk')
                top1_pred = cast_simple_character(top1_pred)
            if l == '':
                continue
            l = ignore_hw(l)
            if l == '':
                continue
            top1_pred = ignore_hw(top1_pred)
            _, acc = word_rec_eval(l, top1_pred)
            self.total_acc += acc
            self.total_count += 1

            if common.PLACE_HOLDER in l:
                self.miss_count += 1

            if common.PLACE_HOLDER not in l and (top1_pred == l or top1_pred.replace(' ','')==l or top1_pred.replace('，',',')==l):
                self.ac_count += 1
            if l!=top1_pred and top1_pred.replace(' ','')!=l and top1_pred.replace(',','')!=l and '{' in l:
                print(l)
                print(top1_pred)
            #     self.fout.write(l + '\n' + top1_pred + '\n')
            #cv2.imshow("debug", image)
            #cv2.waitKey()



    def eval_finished(self, checkpoint_path):
        print("done:", self.total_acc * 100 / self.total_count, self.ac_count * 100.0 / self.total_count,
              self.miss_count * 100 / self.total_count, self.total_count)
        print(2, self.ac_count * 100 / self.total_count)
