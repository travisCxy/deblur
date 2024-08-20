#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import math

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class HandwritingExecutor(object):
    def __init__(self, triton_client):
        self.triton_client = triton_client


    def run_main(self, batch_imgs, batch_seq_lens):
        infer_ret = self.triton_client.infer_ocr(batch_imgs)
        ret = []
        for strings in infer_ret:
            item = [x.decode('utf-8') for x in strings]
            ret.append(item)
        return ret


class HandwritingIdentify(object):

    def __init__(self, batch_count, img_height, img_max_width, triton_client):
        self.batch_count = batch_count
        self.img_height = img_height
        self.img_max_width = img_max_width

        self.img_widths = [self.img_max_width]

        self.executors = []
        executor = HandwritingExecutor(triton_client)
        for _ in self.img_widths:
            self.executors.append(executor)

    def preprocess_img(self, img):
        scale = 1.0
        h, w = img.shape[:2]
        tw = w * self.img_height / h
        tw = int(min(self.img_max_width - 16, tw * scale)) #  pad a little white, the let the attention output the end token.
        seq_len = int(math.ceil(tw / 4.0))
        tw = seq_len * 4
        tw = max([32, tw])
        resized_img = cv2.resize(img, (tw, self.img_height), interpolation=cv2.INTER_AREA)
        return resized_img, seq_len, tw

    def pad_to_lens(self, imgs, widths, fixed_width):
        max_w = fixed_width
        if max_w is None:
            max_w = max(widths)

        # pad a little white, the let the attention output the end token.
        # sometimes the longest can not end as ususal.
        max_w = max_w + 16
        processed_imgs = []
        for img, w in zip(imgs, widths):
            processed_img = cv2.copyMakeBorder(img, 0, 0, 0, max_w - w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            processed_imgs.append(processed_img)
        return processed_imgs

    def preprocess_imgs(self, imgs):
        indices = []
        group_range = range(len(self.img_widths))
        img_groups = [
            {"imgs": [],
             "seq_lens":[],
             "widths": [],
             "pad_width":self.img_widths[ind],
             "executor": self.executors[ind]
             } for ind in group_range
        ]
        for img in imgs:
            resized_img, seq_len, tw = self.preprocess_img(img)
            for group_ind in group_range:
                if self.img_widths[group_ind] >= tw:
                    group = img_groups[group_ind]
                    group["imgs"].append(resized_img)
                    group["seq_lens"].append(seq_len)
                    group["widths"].append(tw)
                    indices.append((group_ind, len(group["imgs"]) - 1))
                    break

        for group in img_groups:
            if len(group["imgs"]) > 0:
                group["imgs"] = self.pad_to_lens(group["imgs"], group["widths"], 752)

        return indices, img_groups

    def identify_with_batch(self, imgs, seq_lens, executor):
        ret = []
        batch_index = 0
        batch_count = self.batch_count
        while True:
            start_index = batch_index * batch_count
            end_index = min(len(imgs), start_index + batch_count)
            batch_imgs = imgs[start_index:end_index]
            batch_seq_lens = seq_lens[start_index: end_index]

            results = executor.run_main(batch_imgs, batch_seq_lens)
            for result in results:
                ret.append(result)

            batch_index += 1
            if batch_index * batch_count >= len(imgs):
                break
        return ret

    def batch_identify(self, imgs):
        indices, img_groups = self.preprocess_imgs(imgs)
        vs_groups = [[] for _ in range(len(img_groups))]
        for ind in range(len(img_groups)):
            group = img_groups[ind]
            if len(group["imgs"]) > 0:
                vs_groups[ind] = self.identify_with_batch(group["imgs"], group["seq_lens"], group["executor"])

        vs = []
        for index in indices:
            group_ind, ind = index
            vs.append(vs_groups[group_ind][ind])

        return vs