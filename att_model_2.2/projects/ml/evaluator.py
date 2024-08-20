import itertools


from projects.ml import common
from projects.ml import eval_summary

import cv2


class Evaluator:
    def __init__(self):
        self.total_acc = 0.0
        self.total_count = 0
        self.miss_count = 0
        self.ac_count = 0
        self.summaries = eval_summary.config_summarys()
        pass

    def record_step(self, outputs, inputs):
        preds = outputs["strings"]
        gt_labels = inputs["labels"]
        images = inputs["images"]
        for ind, image, pred in zip(itertools.count(), images.numpy(), preds):
            #print(gt_labels[ind])
            l = common.attvec2code(gt_labels[ind])
            l = l.replace(' ', '')
            tp1 = pred[0].numpy().decode('utf-8')
            tp2 = pred[1].numpy().decode('utf-8')
            tp3 = pred[2].numpy().decode('utf-8')

            for summary in self.summaries:
                summary.process(l, [tp1, tp2, tp3], "")
            if tp1 == l or tp2 == l or tp3 == l:
                self.total_acc += 1
            if tp1 == l: #or tp2 == l or tp3 == l:
                self.ac_count += 1
            self.total_count += 1
            #print("===============")
            #print(l)
            #print(tp1)
            #print(tp2)
            #print(tp3)
            #cv2.imshow("debug", image)
            #cv2.waitKey()


    def eval_finished(self, checkpoint_path):

        print("done:", self.total_acc * 100 / self.total_count, self.ac_count * 100.0 / self.total_count,
              self.miss_count * 100 / self.total_count, self.total_count)
        print(2, self.ac_count * 100 / self.total_count)
