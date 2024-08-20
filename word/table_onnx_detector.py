import time, cv2
import numpy as np
from word.rotated_rect_utils import three_points_to_lefttop_rightbottom_theta, resize_short_side, makesure_3pointbox_in_wh
from word.data_utils import remove_cover_regions,one_side_iou
import pdb

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188]).astype(np.float32).reshape(-1, 3)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

#类别标签
#0-table
class TableDetector:
    def __init__(self, triton_client):
        self.triton_client = triton_client
        self.CLASSES = ['table']

    def detect(self, img, input_size=(960, 960), t=0.25):
        img_pad, ratio = self.preprocess(img, input_size)
        outputs = self.triton_client.infer_table_onnx_det(img_pad).copy()
        boxes, scores, clses = self.postprocess(outputs, input_size, ratio)
        results = self.gen_format_table(img, boxes, scores, clses)
        return results

    def gen_format_table(self, img, boxes, scores, clses, score_thr=0.25):
        if boxes is None:
            results = {"regions0":[]}
            return results

        h, w, _ = img.shape
        ret_precent_list = []          
        ret_boxes = []
        ret_clses = []
        ret_scores = []
        for ind, score in enumerate(scores):
            cls = int(clses[ind])
            if score > score_thr:
                b = boxes[ind]
                # b = []
                # for v in list(boxes[ind]):
                #     b.append(min(max(v,0),1.0)) 
                box = (b[0] , b[1] , b[2], b[1], b[2], b[3])
                b = (b[0] / w, b[1] / h, b[2] / w, b[1] / h, b[2] / w, b[3] / h)
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

    def preprocess(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = np.expand_dims(padded_img.transpose(swap), 0)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self, outputs, img_size, ratio, p6=False):
        outputs = outputs[0]
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        else:
            final_boxes, final_scores, final_cls_inds = None, None, None 
        return final_boxes, final_scores, final_cls_inds

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img

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

