import time, cv2
from word.rotated_rect_utils import three_points_to_lefttop_rightbottom_theta, resize_short_side, makesure_3pointbox_in_wh
from word.type_cls import TypeCls
from collections import defaultdict
from word.data_utils import remove_cover_regions,one_side_iou
from word.sort_regions import sort_regions
# import pdb


def region3point_to_bbox(region_3point):
    tl = region_3point[0:2]
    tr = region_3point[2:4]
    br = region_3point[4:6]
    right_vec = [br[0] - tr[0], br[1] - tr[1]]
    bl = [tl[0] + right_vec[0], tl[1] + right_vec[1]]
    return [tl, tr, br, bl]


#类别标签
# 2-文本
# 7-图
# 8-表格
class TableDetector:

    def __init__(self, triton_client):
        self.triton_client = triton_client

    def detect(self, img, t=0.25):
        h, w, _ = img.shape
        img1 = resize_short_side(img, 1200)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        boxes, scores, clses = self.triton_client.infer_table_det(img1)
        ret_precent_list = []          
        ret_boxes = []
        ret_clses = []
        ret_scores = []
        for ind, score in enumerate(scores):
            cls = int(clses[ind])
            if score > t or (score > 0.1 and cls in [1,2,9]):
                b = []
                for v in list(boxes[ind]):
                    b.append(min(max(v,0),1.0)) 
                box = (
                        b[0] * w, b[1] * h, b[2] * w, b[3] * h, b[4] * w, b[5] * h                        
                        )
                box = makesure_3pointbox_in_wh(box, w,h)
                ret_boxes.append(box)
                ret_clses.append(cls)
                ret_scores.append(score)
                ret_precent_list.append(b)
        regions = []
        result = {}
        for box, cls, score, b in zip(ret_boxes, ret_clses, ret_scores, ret_precent_list):
            cls0 = int(cls)
            region = {
                    "region_3point" :  list(map(lambda d: round(d, 6), box)),
                    "cls": cls0,
                    "det_score" : round(float(score), 4),
                    "region_percent": list(map(lambda d: round(float(d), 6), b)),
                }
            regions.append(region)
        regions.sort(key = lambda res : (res["cls"], res["region_3point"][1], res["region_3point"][0]))
        regions = remove_cover_regions(regions, h, w)
        result["regions0"] = regions
        return result 

    def preprocess_ocr_result(self, res, need_sort_regions=True):
        orientation = 0
        regions = res['regions0']
        for i, region in enumerate(regions):
            region['index'] = i
            # print(region)
        # pdb.set_trace()
        table_bboxes = [
            region3point_to_bbox(region['region_3point']) 
            for region in regions if region['cls'] in [8, 12]
        ]
        
        text_regions = [region for region in regions if region['cls'] in [2, 9]] # [oc, hc]
        if need_sort_regions:
            text_regions = sort_regions(text_regions)    
        text_cells = [
            {
                # 'content': region['result'][0],
                'content': '',
                'bbox': region3point_to_bbox(region['region_3point']),
                # 'confidence': region['confidence'],
                'confidence': 1.,
                'index': region['index']
            }
            for region in text_regions
        ]
        return table_bboxes, text_cells, orientation  



if __name__ == '__main__':
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

