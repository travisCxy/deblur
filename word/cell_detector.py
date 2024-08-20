from copy import deepcopy

import numpy as np
import cv2
from word.external.custom_nms.nms import soft_nms_13
from word.external.custom_postprocess.post_process import post_process, cells2html, bbox2rect


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:13].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 8)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret



def bbox_2_region(bbox):
    region = []
    for uu in bbox:
        for ww in uu:
            region.append(ww)
    return region 

class CellDetector:
    def __init__(self, triton_client):
        self.triton_client = triton_client
        self.mean = [0.7880491011577201, 0.7913458210223719, 0.7967626895497831]
        self.std = [0.2028187320661555, 0.19855421916531857, 0.19655613837855995]
        self.num_classes = 1

    def inference(self, img, meta_data):
        img, meta = self.preprocess(img)
        dets = self.triton_client.infer_cell_det(img)
        dets = self.postprocess(dets, meta)
        detections = [dets]
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        soft_nms_13(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        cells, confidence = post_process(results[1], deepcopy(meta_data))
        return cells, confidence 

    def detect(self, img, meta_data, offset=5):
        h, w, _ = img.shape
        orig_img = img.copy()
        # for text_cell in meta_data['text_cells']:
        #     x1,y1,x2,y2 = text_cell['bbox'][0][0],text_cell['bbox'][0][1],text_cell['bbox'][1][0],text_cell['bbox'][1][1]
        #     orig_img[y1:y2,x1:x2,:]=255
        #cv2.imwrite('tmp.jpg', cropped_img)
        tables = []
        confidences = []
        for table_bbox in meta_data['table_bboxes']:
            table = {}
            table_rect = bbox2rect(table_bbox)
            min_x = int(max(0, table_rect[0] - offset))
            max_x = int(min(w-1, table_rect[2] + offset))
            min_y = int(max(0, table_rect[1] - offset))
            max_y = int(min(h-1, table_rect[3] + offset))
            table['rect'] = [min_x, min_y, max_x, max_y]
            cropped_img = orig_img[min_y:max_y+1, min_x:max_x+1]
            meta_data.update(
                {
                    'table_bbox': table_bbox,
                    'table_rect': [
                        max(0, table_rect[0]), max(0, table_rect[1]),
                        min(w-1, table_rect[2]), min(h-1, table_rect[3])
                    ],
                    'table_rect_ex': [
                        min_x, min_y,
                        max_x, max_y
                    ]
                }
            )
            cells, confidence = self.inference(cropped_img, meta_data)
            for i in range(len(cells)):
                texts = cells[i]['texts']
                new_texts = []
                for j in range(len(texts)):
                    for k in range(len(texts[j])):
                        texts[j][k]['bbox'] = bbox_2_region(texts[j][k]['bbox'])
                        new_texts.append(texts[j][k])
                cells[i]['texts'] = new_texts
                cells[i]['startrow'] = cells[i]['logical_coordinates']['startrow']
                cells[i]['endrow'] = cells[i]['logical_coordinates']['endrow']
                cells[i]['startcol'] = cells[i]['logical_coordinates']['startcol']
                cells[i]['endcol'] = cells[i]['logical_coordinates']['endcol']
                cells[i]['bbox'] = bbox_2_region(cells[i]['bbox'])
                del cells[i]['logical_coordinates'] 
            table['cells'] = cells
            table['row_num'] = -1
            table['col_num'] = -1
            for cell in cells:
                table['row_num'] = max(table['row_num'], cell['endrow'] + 1)
                table['col_num'] = max(table['col_num'], cell['endcol'] + 1)
            if (table['row_num'] == 1 and table['col_num']==1) or (table['row_num']==-1) or (table['col_num']==-1):
                continue
            tables.append(table)
            confidences.append(confidence)
        return tables, confidences

    def preprocess(self, image, input_size=(512,512), fix_res=True):
        height, width = image.shape[0:2]
        input_h, input_w = input_size
        if fix_res:
            inp_height, inp_width = input_h, input_w
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (height | 31) + 1
            inp_width = (width | 31) + 1
            c = np.array([width // 2, height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        meta = {'c': c, 's': s, 
                'out_height': inp_height // 4, 
                'out_width': inp_width // 4}
        return images, meta

    def postprocess(self, dets, meta, scale=1):
        dets = dets.reshape(1,-1,dets.shape[2])
        dets = multi_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 13)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]
# tdir = '/home/ateam/xychen/projects/CenterNet/src'
# #类别标签
# # 2-文本
# # 7-图
# # 8-表格
# class CellDetector:
#     def __init__(self):
#         args = ['multi_pose', '--load_model', '/home/ateam/xychen/projects/CenterNet/exp/multi_pose/dla_2x/model_best.pth', '--K', '1000', '--nms']
#         opt = opts().parse(args)
#         opt = opts().update_info_and_set_heads(opt)
#         self.model = MultiPoseDetector(opt)

#     def inference(self, cropped_img, meta_data):
#         ret = self.model.run(cropped_img)['results'][1]
#         # pdb.set_trace()
#         cells, confidence = post_process(ret, deepcopy(meta_data))
#         return cells, confidence

#     def detect(self, img, meta_data, offset=5):
#         h, w, _ = img.shape
#         tables = []
#         confidences = []
#         for table_bbox in meta_data['table_bboxes']:
#             table = {}
#             table_rect = bbox2rect(table_bbox)
#             min_x = max(0, table_rect[0] - offset)
#             max_x = min(w-1, table_rect[2] + offset)
#             min_y = max(0, table_rect[1] - offset)
#             max_y = min(h-1, table_rect[3] + offset)
#             table['rect'] = [min_x, min_y, max_x, max_y]
#             cropped_img = img[min_y:max_y+1, min_x:max_x+1]
#             meta_data.update(
#                 {
#                     'table_bbox': table_bbox,
#                     'table_rect': [
#                         max(0, table_rect[0]), max(0, table_rect[1]),
#                         min(w-1, table_rect[2]), min(h-1, table_rect[3])
#                     ],
#                     'table_rect_ex': [
#                         min_x, min_y,
#                         max_x, max_y
#                     ]
#                 }
#             )
#             cells, confidence = self.inference(cropped_img, meta_data)
#             for i in range(len(cells)):
#                 texts = cells[i]['texts']
#                 new_texts = []
#                 for j in range(len(texts)):
#                     for k in range(len(texts[j])):
#                         texts[j][k]['bbox'] = bbox_2_region(texts[j][k]['bbox'])
#                         new_texts.append(texts[j][k])
#                 cells[i]['texts'] = new_texts
#                 cells[i]['startrow'] = cells[i]['logical_coordinates']['startrow']
#                 cells[i]['endrow'] = cells[i]['logical_coordinates']['endrow']
#                 cells[i]['startcol'] = cells[i]['logical_coordinates']['startcol']
#                 cells[i]['endcol'] = cells[i]['logical_coordinates']['endcol']
#                 cells[i]['bbox'] = bbox_2_region(cells[i]['bbox'])
#                 del cells[i]['logical_coordinates'] 
#             table['cells'] = cells
#             table['row_num'] = -1
#             table['col_num'] = -1
#             for cell in cells:
#                 table['row_num'] = max(table['row_num'], cell['endrow'] + 1)
#                 table['col_num'] = max(table['col_num'], cell['endcol'] + 1)
#             tables.append(table)
#             confidences.append(confidence)
#         return tables, confidences