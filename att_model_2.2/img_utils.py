import cv2
import imutils
import random
import math
import numpy as np
from text_image_aug.augment import tia_distort, tia_stretch, tia_perspective
# from text_image_aug.native_augment import tia_distort, tia_stretch, tia_perspective



resize_methods = [cv2.INTER_AREA, cv2.INTER_NEAREST,
                  cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def img_to_fix_size(img, fix_size, use_random=True, method="all", img_ref_size=None):
    if method not in ["scale", "pad", "all"]:
        raise ValueError()

    m1 = cv2.INTER_LINEAR
    if use_random:
        m1 = resize_methods[random.randint(0, 4)]

    if method == "all":
        scale_input = random.randint(0, 10) <= 5
    elif method == "scale":
        scale_input = True
    else:
        scale_input = False

    fix_width, fix_height = fix_size
    if scale_input:
        return cv2.resize(img, (fix_width, fix_height), interpolation=m1)
    else:
        img = imutils.resize(img, height=fix_height, inter=m1)
        if img.shape[1] > fix_width:
            img = imutils.resize(img, width=fix_width, inter=m1)

        h, w = img.shape[:2]
        t, l = 0, 0
        if use_random:
            t = random.randint(0, fix_height - h)
            l = random.randint(0, fix_width - w)

        if img_ref_size is not None and isinstance(img_ref_size, dict):
            min_ys = img_ref_size.get("min_ys", None)
            max_ys = img_ref_size.get("max_ys", None)
            if min_ys is not None:
                min_ys = [(h * y + t) / fix_height for y in min_ys]
                img_ref_size["min_ys"] = min_ys
            if max_ys is not None:
                max_ys = [(h * y + t) / fix_height for y in max_ys]
                img_ref_size["max_ys"] = max_ys

        return cv2.copyMakeBorder(img, t, fix_height - t - h,
                                  l, fix_width - l - w, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def img_to_dynamic_size(img, max_width, fix_height, fix_scale=None, pad_to_max=False, use_random=True):
    m1 = cv2.INTER_LINEAR
    if use_random:
        m1 = resize_methods[random.randint(0, 4)]

    scale = fix_scale
    if fix_scale is None:
        scale = random.choice([1.0, 1.1, 1.2])

    h, w = img.shape[:2]
    w = w * fix_height / h
    w = int(min(max_width, w * scale))
    seq_len = np.int32(math.ceil(w / 4.0))
    w = seq_len * 4

    img = cv2.resize(img, (w, fix_height), interpolation=m1)
    #    print "=======2 img w: %d ======="%w
    if pad_to_max:
        img = cv2.copyMakeBorder(img, 0, 0, 0, max_width - w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img, w, seq_len


def image_aug_wrap(img, prob=0.4, distort=True, stretch=True, perspective=True):
    new_img = img

    h, w = img.shape[:2]
    if distort:
        if random.random() <= prob and h >= 20 and w >= 20:
            new_img = tia_distort(new_img, random.randint(3, 6))

    if stretch:
        if random.random() <= prob and h >= 20 and w >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

    if perspective:
        if random.random() <= prob:
            new_img = tia_perspective(new_img)

    return new_img

