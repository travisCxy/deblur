from . import common


class Summary:

    def __init__(self, text):
        self.total = 0
        self.scale_failed = 0
        self.pad_failed = 0
        self.both_failed = 0
        self.summary_text = text

        self.scale_rate = 0.0
        self.pad_rate = 0.0
        self.union_rate = 0.0

    def total(self):
        return self.total

    def process_result(self, batch, scale_not_matches, pad_not_matches):
        filtered_batch, filtered_indices = self._filter_batch(batch)
        filtered_scale_not_matches = self._filter_not_matches(
            filtered_indices, scale_not_matches)
        filtered_pad_not_matches = self._filter_not_matches(
            filtered_indices, pad_not_matches)

        self.total += len(filtered_batch)
        self.scale_failed += len(filtered_scale_not_matches)
        self.pad_failed += len(filtered_pad_not_matches)

        scale_failed_indices = [v[0] for v in filtered_scale_not_matches]
        pad_failed_indices = [v[0] for v in filtered_pad_not_matches]
        both_not_matches = list(filter(
            lambda x: x in pad_failed_indices, scale_failed_indices))
        self.both_failed += len(both_not_matches)
        return both_not_matches

    def display_summary_result(self):
        scale_right = self.total - self.scale_failed
        pad_right = self.total - self.pad_failed
        success = self.total - self.both_failed
        if self.total == 0:
            self.total = 1

        self.scale_rate = scale_right * 1.0 / self.total
        self.pad_rate = pad_right * 1.0 / self.total
        self.union_rate = success * 1.0 / self.total
        print("%s scale: %d  pad: %d, union: %d, total:%d %.4f, %.4f, %.4f" % (self.summary_text, scale_right,
                                                                               pad_right, success, self.total,
                                                                               scale_right * 1.0 / self.total,
                                                                               pad_right * 1.0 / self.total,
                                                                               success * 1.0 / self.total))

    def _filter_batch(self, batch):
        return batch, range(len(batch))

    def _filter_not_matches(self, filtered_indices, not_matches):
        return list(filter(lambda x: x[0] in filtered_indices, not_matches))


class FracSummary(Summary):

    def __init__(self):
        Summary.__init__(self, "Frac")

    def _filter_batch(self, batch):
        ret = []
        indices = []
        for i in range(0, len(batch)):
            if '\\frac' in batch[i]:
                ret.append(batch[i])
                indices.append(i)
        return ret, indices


class DecimalsSummary(Summary):

    def __init__(self):
        Summary.__init__(self, "Decimals")

    def _filter_batch(self, batch):
        ret = []
        indices = []
        for i in range(0, len(batch)):
            if '.' in batch[i]:
                ret.append(batch[i])
                indices.append(i)
        return ret, indices


class UnitSummary(Summary):

    def __init__(self):
        Summary.__init__(self, "Unit")

    def _filter_batch(self, batch):
        ret = []
        indices = []
        for i in range(0, len(batch)):
            if '[' in batch[i]:
                ret.append(batch[i])
                indices.append(i)
        return ret, indices


class EquationSummary(Summary):

    def __init__(self):
        Summary.__init__(self, "Equation")

    def _filter_batch(self, batch):
        ret = []
        indices = []
        for i in range(0, len(batch)):
            if common.contains_unknown_token(batch[i]):
                ret.append(batch[i])
                indices.append(i)
        return ret, indices


def config_summarys():
    summarys = [Summary("All:"), FracSummary(), DecimalsSummary(), UnitSummary(), EquationSummary()]
    return summarys

