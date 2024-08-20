import itertools

from projects.equ import common
from projects.equ.eval_summary import config_summarys


class Evaluator:
    def __init__(self):
        self.summarys = config_summarys()

    def record_step(self, outputs, inputs):
        gts, rets = [], []
        labels = inputs["labels"].numpy()
        images = inputs["images"].numpy()
        #filenames = outputs["filenames"].numpy()

        main_strings = outputs["main_strings"]
        num_strings = outputs["num_strings"]
        den_strings = outputs["den_strings"]

        scale_not_matches, pad_not_matches = [], []
        for ind, mains, nums, dens in zip(itertools.count(), main_strings.numpy(),
                                          num_strings.numpy(), den_strings.numpy()):
            image = images[ind // 2][0]
            gt = labels[ind // 2][0].decode('utf-8')
            pt = common.frac_exp(mains[0].decode('utf-8'), nums[0].decode('utf-8'), dens[0].decode('utf-8'))

            if ind % 2 == 0:
                gts.append(gt)
                if pt != gt:
                    scale_not_matches.append((ind // 2, pt))
            else:
                if pt != gt:
                    pad_not_matches.append((ind // 2, pt))

        for i in range(len(self.summarys)):
            total = self.summarys[i].total
            both_not_matches = self.summarys[i].process_result(gts, scale_not_matches, pad_not_matches)
            if i == 0:
                for ind in both_not_matches:
                    scale_answer = [v[1]
                                    for v in scale_not_matches if v[0] == ind][0]
                    pad_answer = [v[1]
                                  for v in pad_not_matches if v[0] == ind][0]
                    print('%d' % (total + ind),  'label:', gts[ind],
                          'scale:', scale_answer, 'pad:', pad_answer)

    def eval_finished(self, checkpoint_path):
        index = checkpoint_path.rfind('-')
        step = checkpoint_path[index + 1:]
        svn_log = "Step: " + step
        for summary in self.summarys:
            summary.display_summary_result()
            svn_log = svn_log + " %s: scale %.4f pad %.4f union %.4f" % (
                summary.summary_text, summary.scale_rate, summary.pad_rate, summary.union_rate)
        print(svn_log)
        print(len(self.summarys) + 2, self.summarys[0].union_rate)