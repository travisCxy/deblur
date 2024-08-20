import itertools
from uniform_utils import str_uniform

from projects.ch import common


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


class Evaluator:
    def __init__(self):
        self.total_acc = 0.0
        self.total_count = 0
        pass

    def record_step(self, outputs, inputs):
        preds = outputs["strings"]
        gt_labels = inputs["labels"]
        seq_lens = inputs["seq_lens"]
        images = inputs["images"]
        for ind, image, pred in zip(itertools.count(), images.numpy(), preds):
            l = common.vec2code(gt_labels[ind])
            l = str_uniform(l)
            top1_pred = pred[0].numpy().decode('utf-8')
            _, acc = word_rec_eval(l, top1_pred)
            self.total_acc += acc
            self.total_count += 1

    def eval_finished(self, checkpoint_path):
        print("done:", self.total_acc * 100 / self.total_count, self.total_count)
        print(2, self.total_acc * 100 / self.total_count)