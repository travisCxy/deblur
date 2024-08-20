import re

class Summary(object):
    def __init__(self):
        self.total = 0
        self.top_1 = 0
        self.top_3 = 0
        #self.tf_summary = tf.Summary()

    def _is_my_type(self, label, ori_path):
        return True

    def _get_type_str(self):
        return "All"

    def process(self, label, predicts, ori_path):
        if self._is_my_type(label, ori_path):
            self.total += 1
            if label == predicts[0]:
                self.top_1 += 1
            if label in predicts[:3]:
                self.top_3 += 1

    def print_result(self):
        div_num = self.total if self.total > 0 else 1
        print ("%s:" % (self._get_type_str()), "total:", self.total, "right:", self.top_1,
               self.top_3, "rate:", self.top_1 * 1.0 / div_num, self.top_3 * 1.0 / div_num)

#    def write_tf_summary(self):
#        div_num = self.total if self.total > 0 else 1
#        self.tf_summary.value.add(
#            tag="eval/%s_top_1" % self._get_type_str(), simple_value=self.top_1 * 1.0 / div_num)
#        self.tf_summary.value.add(
#            tag="eval/%s_top_3" % self._get_type_str(), simple_value=self.top_3 * 1.0 / div_num)


class DecimalSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return '.' in label

    def _get_type_str(self):
        return "Decimal"


class FracSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return label.find("\\frac") >= 0

    def _get_type_str(self):
        return "Frac"


class EquSetSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return label.find("\\begin") >= 0

    def _get_type_str(self):
        return "Equset"


class TrigonoSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return label.find("\\sin") >= 0 or label.find("\\cos") >= 0 or label.find("\\tan") >= 0 or label.find("\\arc")  >= 0\
                or label.find("\\cot") >= 0

    def _get_type_str(self):
        return "Trigonometry"


class SqrtSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return label.find("\\sqrt") >= 0

    def _get_type_str(self):
        return "Sqrt"


class ComplexSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return label.find("\\int") >= 0 or label.find("\\lim") >= 0 or label.find("\\sum") >= 0 or label.find("\\prod") >= 0

    def _get_type_str(self):
        return "ComplexOthers"


class OfSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        return label.find("of") >= 0

    def _get_type_str(self):
        return "Of"


class MetricSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

    def _is_my_type(self, label, ori_path):
        m = re.search(r"\[[a-zA-Z]{2,10}\]", label)
        if m is None:
            return False
        else:
            return True

    def _get_type_str(self):
        return "Metric"


def config_summarys():
    ret = []
    ret.append(Summary())
    ret.append(FracSummary())
    ret.append(EquSetSummary())
    ret.append(TrigonoSummary())
    ret.append(SqrtSummary())
    ret.append(ComplexSummary())
    ret.append(OfSummary())
    ret.append(MetricSummary())
    return ret
