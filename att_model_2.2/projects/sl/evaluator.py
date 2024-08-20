import itertools

from projects.sl import common
from projects.sl.eval_summary import config_summarys


class Evaluator:
    def __init__(self):
        self.summarys = config_summarys()

    def record_step(self, outputs, inputs):
        gt, ret = [], []
        labels = inputs["labels"]
        for ind, vec, label, image in zip(itertools.count(), outputs["strings"].numpy(), labels, inputs["images"]):
            g = common.attvec2code(label)
            gt.append(g)
            pt1 = vec[0].decode('utf-8')
            pt2 = vec[1].decode('utf-8')
            ret.append((pt1, pt2))
        for i in range(len(self.summarys)):
            self.summarys[i].process_result(gt, ret, norm_ws=inputs["norm_ws"])

    def eval_finished(self, checkpoint_path):
        index = checkpoint_path.rfind('-')
        step = checkpoint_path[index + 1:]
        svn_log = "Step: " + step
        for summary in self.summarys:
            summary.display_summary_result()
            svn_log = svn_log + " %s: top1 %.4f top2 %.4f" % (
                summary.summary_text, summary.top1_rate, summary.top2_rate)
        print(svn_log)
        print(len(self.summarys) + 2, self.summarys[0].top2_rate)