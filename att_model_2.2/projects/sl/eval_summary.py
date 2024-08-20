from projects.sl import common


class Summary:

    def __init__(self, text):
        self.total = 0
        self.top1_failed = 0
        self.top2_failed = 0
        self.summary_text = text

    def total(self):
        return self.total

    def process_result(self, gt_labels, predicts, **kwargs):
        filtered_batch, filtered_indices = self._filter_batch(gt_labels, **kwargs)
        top1_wrong_predicts = [i for i in filtered_indices if gt_labels[i] not in predicts[i][:1]]
        top2_wrong_predicts = [i for i in filtered_indices if gt_labels[i] not in predicts[i][:2]]
        self.total += len(filtered_batch)
        self.top1_failed += len(top1_wrong_predicts)
        self.top2_failed += len(top2_wrong_predicts)

        return top2_wrong_predicts

    def display_summary_result(self):
        top1_right = self.total - self.top1_failed
        top2_right = self.total - self.top2_failed
        if self.total == 0:
            self.total = 1

        self.top1_rate = top1_right * 1.0 / self.total
        self.top2_rate = top2_right * 1.0 / self.total

        print("%s total:%d %.4f, %.4f" % (self.summary_text, self.total, self.top1_rate, self.top2_rate))

    def _filter_batch(self, batch, **kwargs):
        return batch, range(len(batch))

    def l_filter_not_matches(self, filtered_indices, not_matches):
        return filter(lambda x: x[0] in filtered_indices, not_matches)


class DigitSummary(Summary):

    def __init__(self):
        Summary.__init__(self, "Digit")
        self._digits = common.code2vec("9876543210")

    def _filter_batch(self, batch, **kwargs):
        ret_gt = []
        ret_indices = []
        for ind in range(len(batch)):
            gt = batch[ind]
            vec = common.code2vec(gt)
            keep = True
            for v in vec:
                if v == -1:
                    break

                if v not in self._digits:
                    keep = False
                    break
            if keep:
                ret_gt.append(gt)
                ret_indices.append(ind)
        return ret_gt, ret_indices


class ImgLenSummaryWraper(Summary):
    def __init__(self, summary, img_len=64):
        text = "ShorterImg " + summary.summary_text
        Summary.__init__(self, text)
        self._img_len = img_len
        self._summary = summary

    def _filter_batch(self, batch, **kwargs):
        gts, indices = self._summary._filter_batch(batch, **kwargs)
        norm_ws = kwargs["norm_ws"]
        ret_gts, ret_indices = [], []
        for gt, ind in zip(gts, indices):
            if norm_ws[ind] <= self._img_len:
                ret_gts.append(gt)
                ret_indices.append(ind)
        return ret_gts, ret_indices


def config_summarys():
    summarys = []
    summarys.append(Summary("All:"))
#    summarys.append(DigitSummary())

    return summarys

