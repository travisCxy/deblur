import time, cv2
from word.rotated_rect_utils import three_points_to_lefttop_rightbottom_theta
from word.type_cls import TypeCls
from collections import defaultdict
import numpy as np
class ImgSplitter:

    def __init__(self, triton_client):
        self.triton_client = triton_client

    def detect(self, img, cache, key, for_html=False):
        if key in cache:
            return cache[key]
        h, w, _ = img.shape
        if for_html:
            max_size = max(h, w)
            pad_img = np.ones((max_size + 100, max_size + 100, 3), dtype=np.uint8) * 255
            self.pad_size = max_size + 100
            pad_img[100:100 + h, 100:100 + w] = img
            pad_img = cv2.resize(pad_img, (1000, 1000))
            boxes, scores, clses, calc_boxes, calc_scores, calc_clses = self.triton_client.infer_det(pad_img)
        else:
            if h > 1000 or w > 1000:
                if h > w:
                    img = cv2.resize(img, (round(w * 1000 / h), 1000))
                else:
                    img = cv2.resize(img, (1000, round(h * 1000 / w)))
            boxes, scores, clses, calc_boxes, calc_scores, calc_clses = self.triton_client.infer_det(img)

        ret_yyt_boxes, ret_yyt_clses, ret_calc_boxes, ret_calc_clses= [], [], [], []
        for ind, score in enumerate(scores):
            if score > 0.5:
                ret_yyt_boxes.append(boxes[ind])
                ret_yyt_clses.append(clses[ind])
        for ind, score in enumerate(calc_scores):
            if score > 0.8:
                if calc_clses[ind] == 1:
                    b = calc_boxes[ind]
                    x1, y1 = int(b[0] * w), int(b[1] * h)
                    x2, y2 = int(b[2] * w), int(b[3] * h)
                    x3, y3 = int(b[4] * w), int(b[5] * h)
                    rect_w_pow = pow((x1 - x2), 2) + pow((y1-y2), 2)
                    rect_h_pow = pow((x2 - x3), 2) + pow((y2 - y3), 2)
                    if rect_h_pow >= rect_w_pow:
                        continue
                # shushi to yyt image.
                if calc_clses[ind] == 2:
                    ret_yyt_boxes.append(calc_boxes[ind])
                    ret_yyt_clses.append(3)

                ret_calc_boxes.append(calc_boxes[ind])
                ret_calc_clses.append(calc_clses[ind])
        cache["yyt"] = (ret_yyt_boxes, ret_yyt_clses)
        cache["calc"] = (ret_calc_boxes, ret_calc_clses)
        return cache[key]

    def detect_regions(self, img, data, cache, key, for_html=False):
        h, w = img.shape[:2]
        start_time = time.time()
        regions, clses = self.detect(img, cache, key, for_html)
        end_time = time.time()
        split_cost = end_time - start_time

        outter_regions = []
        region_3points = []
        for r, cls in zip(regions, clses):
            x1, y1, x2, y2, x3, y3 = r
            if not for_html:
                x1 = w * x1
                x2 = w * x2
                x3 = w * x3
                y1 = h * y1
                y2 = h * y2
                y3 = h * y3
            else:
                x1 = self.pad_size * x1 - 100
                x2 = self.pad_size * x2 - 100
                x3 = self.pad_size * x3 - 100
                y1 = self.pad_size * y1 - 100
                y2 = self.pad_size * y2 - 100
                y3 = self.pad_size * y3 - 100
            outter_region = three_points_to_lefttop_rightbottom_theta((x1, y1, x2, y2, x3, y3), True)
            x1, y1, x2, y2, theta = outter_region
            x1 = max([0, x1])
            x2 = min([w, x2])
            y1 = max([0, y1])
            y2 = min([h, y2])
            outter_regions.append([x1, y1, x2, y2, theta])
            region_3points.append([x1,y1,x2,y2,x3,y3])
        # outter_regions, outter_clses = format_converter.fix_one_line_topic_issue(outter_regions, outter_clses)
        org_regions = []

        # in some extreme case, nms can not correctly remove the duplicated rect if the two rect is every close.
        # here is an example.(the data is from the kernel logging.
        # RotatedRect(Point2f(458.339f, 949.402f), Size2f(618.002f, 26.8202f), -0.00337255f);  center, size, angle
        # RotatedRect(Point2f(458.394f, 949.401f), Size2f(618.069f, 26.8189f), -0.00356456f);
        # the intersection points:(only three point with area about 0)
        # (268.291, 935.997), (149.337, 936.01), (767.339, 935.974)
        # note: the data comes from log, so the data is not the accurate value. 
        # the bug it's hard to debug, because of  the accurate value.  
        duplicate_sets = defaultdict(set)
        for reg, cls, reg_3p in zip(outter_regions, clses, region_3points):
            if key == "yyt":
                cls = int(TypeCls.convert_inner(int(cls)))
            else:
                cls = int(cls)
            reg_tuple = tuple(reg)
            if reg_tuple in duplicate_sets[cls]:
                continue
            duplicate_sets[cls].add(reg_tuple)
            org_regions.append({"region": reg[:4], "rotation": reg[4], "cls": cls, 'region_3point':reg_3p})
        data["regions"] = org_regions
        return data, {"detect_cost": split_cost}


if __name__ == '__main__':
    #    def __init__(self, gpu_memory_fraction, models_path):
    
#  def __init__(self,
#               input_saved_model_dir=None,
#               input_saved_model_tags=None,
#               input_saved_model_signature_key=None,
#               input_graph_def=None,
#               nodes_blacklist=None,
#               session_config=None,
#               max_batch_size=1,
#               max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
#               precision_mode=TrtPrecisionMode.FP32,
#               minimum_segment_size=3,
#               is_dynamic_op=False,
#               maximum_cached_engines=1,
#               use_calibration=True):
    tf.compat.v1.disable_v2_behavior()
    converter = trt.TrtGraphConverter(input_saved_model_dir="/mnt/data2/heping/servers/yyt_server_2.1/models/det/export",
                          input_saved_model_tags=["tag"],
                          input_saved_model_signature_key="model",
                          minimum_segment_size=110,
                          is_dynamic_op=True,
                          maximum_cached_engines=20,
                          use_calibration=False)
    converter.convert()
    converter.save("/mnt/data2/heping/servers/yyt_server_2.1/models/det/export_trt")
    
    ImgSplitter(0.4, "/share/hwhelper_yyt_new/models")
    print ("done")

