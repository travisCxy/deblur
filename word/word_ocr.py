from word import imsplitter, equIdentify, cell_detector, publay_detector, table_onnx_detector, img_detector2, square_cell_detector

from word.type_cls import TypeCls
from word import rotated_rect_utils
from word import format_converter
from word.table_detector import region3point_to_bbox
from word.data_utils import merge_det_res,merge_square_cell_res
from word.topic_builder import TopicBuilder
import os
import time
import cv2
import numpy as np
import json 
import random
import logging
from zhconv import convert

def remove_dollar(l):
    if len(l) < 3:
        return l
    if l[0] == '$' and l[-1] == '$':
        return l[1:-1]
    return l


def ignore_hw(s):
    ret = ""
    ignore = False
    for ind, c in enumerate(s):
        if c == '$':
            if ind - 1 < 0 or s[ind - 1] != '\\':
                ignore = not ignore
                continue
            else:
                if not ignore:
                    ret += c
        else:
            if not ignore:
                ret += c
    return ret


def cast_region(ret, threshold=0.65):
    regions = ret['regions']
    pics = []
    lines = []
    others = []
    for region in regions:
        if region['cls'] == 3:
            pics.append(region)
        elif region['cls'] in [1,2,10]:
            lines.append(region)
        else:
            others.append(region)
    output = []
    for line in lines:
        cx1, cy1, cx2, cy2 = line['region']
        flag = False
        for tmp in pics:
            bx1, by1, bx2, by2 = tmp['region']
            if (cx1 > bx1 and cy1 > by1 and cx2 < bx2 and cy2 < by2) or iou([bx1, by1, bx2, by2], [cx1, cy1, cx2, cy2]) > threshold:
                flag = True
                break
        if not flag:
            output.append(line)
    ret['regions'] = output + pics + others
    return ret


def iou_1d(l1, l2, real_iou=True):
    range1 = l1[1] - l1[0]
    range2 = l2[1] - l2[0]
    xmin = max(l1[0], l2[0])
    xmax = min(l1[1], l2[1])
    if xmax<=xmin:
        return 0
    if real_iou==False:
        return (xmax-xmin)/range2
    else:
        return (xmax-xmin) / (range1 + range2 - (xmax-xmin))

def generate_topic_with_lines(topics, regions):
    if topics == []:
        return []
    lines = []
    for region in regions:
        if region["cls"] in [1, 2, 3, 10] and 'result' in region and region['result'] != []:
            lines.append(region)
    used = [False] * len(lines)
    for topic in topics:
        topic['lines'] = []
        for i, line in enumerate(lines):
            if used[i]:
                continue
            if iou(topic["region"], line["region"]) > 0.75:
                topic["lines"].append(line)
                used[i] = True
    for topic in topics:
        topic["lines"].sort(key=lambda x: x["region"][1])
        merged_lines = []
        cur_line = []
        for line in topic["lines"]:
            cur_bbox = line['region']
            cur_h = (cur_bbox[3] + cur_bbox[1])/2
            cur_word_h = (cur_bbox[3] - cur_bbox[1])
            if cur_line==[]:
                cur_line.append(line)
                last_h = cur_h
                last_bbox = cur_bbox
                continue
            #若当前内容的y坐标和上一内容的y坐标差小于上一内容的高度值的一半，则将当前内容合并至上一内容
            if cur_h-last_h<cur_word_h*0.7 and iou_1d([last_bbox[0],last_bbox[2]], [cur_bbox[0], cur_bbox[2]], False)<0.2 and iou_1d([cur_bbox[0], cur_bbox[2]], [last_bbox[0],last_bbox[2]], False)<0.2:
                cur_line.append(line)
            else:
                cur_line = sorted(cur_line, key=lambda x:x['region'][0])
                merged_lines.append(cur_line)
                cur_line = [line]
                last_h = cur_h
            last_bbox = cur_bbox
        if cur_line!=[]:
            cur_line = sorted(cur_line, key=lambda x:x['region'][0])
            merged_lines.append(cur_line)
        topic["result"] = []
        for line in merged_lines:
            topic["result"].append(" ".join([remove_dollar(x["result"][0]) for x in line]))
        del topic["lines"]
    topics.sort(key=lambda x: x["region"][1])
    return topics

def iou(bbox1, bbox2, real_iou=False):
    para_x1, para_y1, para_x2, para_y2 = bbox1
    x1,y1,x2,y2 = bbox2
    inter_x1 = max(para_x1, x1)
    inter_y1 = max(para_y1, y1)
    inter_x2 = min(para_x2, x2)
    inter_y2 = min(para_y2, y2)
    inter_area = max(0, (inter_y2-inter_y1)) * max(0, (inter_x2-inter_x1))
    cur_area = (x2-x1)*(y2-y1)
    if real_iou:
        out_x1 = min(para_x1, x1)
        out_y1 = min(para_y1, y1)
        out_x2 = max(para_x2, x2)
        out_y2 = max(para_y2, y2)
        cur_area = (out_y2-out_y1)*(out_x2-out_x1)
    
    ratio = inter_area * 1. / cur_area
    return ratio


class DataTransfer(object):

    def __init__(self, data):
        self._data = data

    def to_array(self):
        if isinstance(self._data, dict):
            return [self._data]
        if isinstance(self._data, list):
            return self._data
        return None

    def org_form(self):
        return self._data

def draw_bbox(image, cells, color=None):
    for cell in cells['cells']:
        #
        bbox = cell['bbox']
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=int).reshape((-1,2))
            cv2.polylines(
                image, 
                [bbox], 
                isClosed=True, 
                color=tuple([random.randint(0, 200) for _ in range(3)]) if color is None else color, 
                thickness=3
            )
    return image

def cast_overlap_region(det, cls=[1]):
    regions = det['regions']
    region_sorted = sorted(regions,key=lambda x: (x['region'][2] - x['region'][0]) * (x['region'][3] - x['region'][1]), reverse=True)

    casted_idx = []
    for idx in range(len(region_sorted)):
        if region_sorted[idx]['cls'] not in cls:
            continue
        if idx in casted_idx:
            continue
        cx1, cy1, cx2, cy2 = region_sorted[idx]['region']
        for j in range(idx + 1, len(region_sorted)):
            if region_sorted[j]['cls'] not in cls:
                continue
            bx1, by1, bx2, by2 = region_sorted[j]['region']
            if iou([cx1, cy1, cx2, cy2], [bx1, by1, bx2, by2]) > 0.7:
                casted_idx.append(j)
    new_regions = []
    for i, region in enumerate(region_sorted):
        if i not in casted_idx:
            new_regions.append(region)
    new_regions = sorted(new_regions, key=lambda x:x['cls'])
    det['regions'] =  new_regions
    return det

def get_cell_pics(tables, ret):
    all_pics = []
    others = []
    used = []
    for region in ret['regions']:
        if region['cls']==3:
            all_pics.append(region)
            used.append(False)
        else:
            others.append(region)

    for table in tables:
         for cell in table['cells']:
            cell_bbox = cell['bbox']
            for i, pic in enumerate(all_pics):
                if used[i]:
                    continue
                pic_bbox = pic["region"]
                iou_ = iou([cell_bbox[0], cell_bbox[1], cell_bbox[4], cell_bbox[5]], pic_bbox)
                if iou_ > 0.9:
                    cell['texts'].append(pic)
                    used[i] = True
                    #break
    for i in range(len(all_pics)):
        if not used[i]:
            others.append(all_pics[i])
    ret['regions'] = others
    return tables, ret

def split_pic_text(regions, tables):
    pics = []
    new_regions = []
    a = []
    for region in regions:
        a.append(region['index'])
    table_region_indexs = []
    for table in tables:
        for cell in table['cells']:
            for text in cell['texts']:
                table_region_indexs.append(text['index'])
    table_region_indexs.sort()
    for region in regions:
        index = region['index']
        if region['cls'] == 3:
            pics.append(region)
            continue
        if index not in table_region_indexs:
            new_regions.append(region)
    return pics, new_regions

def cast_bad_table(tables):
    new_tables = []
    for table in tables:
        if table['col_num'] ==1:
            continue
        row_num = table['row_num']
        col_num = table['col_num']
        cells = table['cells']
        cell_index = []
        for i in range(row_num):
            tmp = []
            for j in range(col_num):
                tmp.append(-1)
            cell_index.append(tmp)
        for idx in range(len(cells)):
            startrow = cells[idx]['startrow']
            endrow = cells[idx]['endrow']
            startcol = cells[idx]['startcol']
            endcol = cells[idx]['endcol']
            for i in range(startrow, endrow + 1):
                for j in range(startcol, endcol + 1):
                    cell_index[i][j] = idx
        # if row_num>=2 and col_num>=2 and cell_index[row_num-1][col_num-1] == -1 and cell_index[row_num-2][col_num-1]!=-1 and cell_index[row_num-1][col_num-2]!=-1:
            # new_cell = cells[cell_index[row_num-2][col_num-1]].copy()
            # new_cell['startrow'] = row_num-1
            # new_cell['endrow'] = row_num-1
            # new_cell['startcol'] = col_num-1
            # new_cell['endcol'] = col_num-1
            # new_cell['bbox'][0] = cells[cell_index[row_num-2][col_num-1]]['bbox'][0]
            # new_cell['bbox'][2] = cells[cell_index[row_num-2][col_num-1]]['bbox'][2]
            # new_cell['bbox'][1] = cells[cell_index[row_num-1][col_num-2]]['bbox'][1]
            # new_cell['bbox'][3] = cells[cell_index[row_num-1][col_num-2]]['bbox'][3]
            # new_cell['bbox'][4] = cells[cell_index[row_num-1][col_num-2]]['bbox'][4]
            # cells.append(new_cell)
            # cell_index[row_num-1][col_num-1] = len(cells)
        cell_index_np = np.array(cell_index)
        cast_col = False
        cast_row = False
        if sum(cell_index_np[:,0]==-1)==row_num:
            col_num -= 1
            table['col_num'] -= 1
            cast_col = True
            for cell in table['cells']:
                cell['startcol'] -= 1
                cell['endcol'] -= 1
        if sum(cell_index_np[0,:]==-1)==col_num:
            row_num -= 1
            table['row_num'] -= 1
            cast_row = True
            for cell in table['cells']:
                cell['startrow'] -= 1
                cell['endrow'] -= 1
        n_cells = row_num*col_num
        if n_cells<=1:
            continue
        n_invalid_cell = 0
        for i in range(row_num):
            for j in range(col_num):
                if cell_index[i][j]==-1:
                    n_invalid_cell+=1
        if cast_col:
            n_invalid_cell -= row_num
        if cast_row:
            n_invalid_cell -= col_num
        if cast_row and cast_col:
            continue
        if 100.*n_invalid_cell/n_cells>5 and n_cells>=8 and n_invalid_cell>=2:
            continue
        new_tables.append(table)
    region_sorted = sorted(new_tables, key=lambda x: (x['rect'][2] - x['rect'][0]) * (x['rect'][3] - x['rect'][1]),
                           reverse=True)

    casted_idx = []
    for idx in range(len(region_sorted)):
        if idx in casted_idx:
            continue
        cx1, cy1, cx2, cy2 = region_sorted[idx]['rect']
        for j in range(idx + 1, len(region_sorted)):
            bx1, by1, bx2, by2 = region_sorted[j]['rect']
            if iou([cx1, cy1, cx2, cy2], [bx1, by1, bx2, by2]) > 0.3:
                casted_idx.append(j)
    new_regions = []
    for i, region in enumerate(region_sorted):
        if i not in casted_idx:
            new_regions.append(region)
    return new_regions

class WordOcr:
    def __init__(self, triton_client):
        self._spliter = imsplitter.ImgSplitter(triton_client)
        self._identifier = equIdentify.HandwritingIdentify(50, 40, 768, triton_client)
        # self.table_detector = table_detector.TableDetector(triton_client)
        self.table_detector = table_onnx_detector.TableDetector(triton_client)
        self.img_detector2 = img_detector2.ImgDetector2(triton_client)
        self.square_cell_detector = square_cell_detector.SquareCellDetector(triton_client)
        self.cell_detector = cell_detector.CellDetector(triton_client)
        self.publay_detector = publay_detector.PublayDetector(triton_client)
        self.index = 1

    def identify_table(self, img, json_path, vis_path):
        h, w = img.shape[:2]
        res = self.table_detector.detect(img)
        table_bboxes, text_cells, orientation  = self.table_detector.preprocess_ocr_result(res)
        meta_data = {'table_bboxes':table_bboxes,
                    'text_cells': text_cells,
                    'orientation': orientation}
        tables, confidences = self.cell_detector.detect(img, meta_data)

        if len(table_bboxes)!=0:
            json_dict = self.generate_table_json(tables)
            with open(json_path, 'w') as f:
                f.write(json.dumps(json_dict))
            self.visualize_cell_result(img, table_bboxes, tables, vis_path)
            return True 
        return False 

    #检测识别+表格识别+文档段落识别，针对文档
    def identify_document(self, img):
        pub_res = self.publay_detector.detect(img)
        body = self.identify_paper(img)

        regions = body['regions']
        new_regions = pub_res
        for region in regions:
            if region['cls']==4:
                continue 
            #bbox = region['region']
            new_regions.append(region)
        body['regions'] = new_regions
        return body

    def identify_cell(self, img):
        res = self.cell_detector.inference(img)
        return  res

    #最老的版本
    def identify(self, img):
        h, w = img.shape[:2]
        data, cost, cache = {}, {}, {}
        #检测
        ret, cost_1 = self._spliter.detect_regions(img, data, cache, "yyt")
        cost.update(cost_1)
        empty_line_topics = format_converter.find_empty_line_topic(ret["regions"])
        for topic in empty_line_topics:
            x1, y1, x2, y2 = topic["region"]
            if x2 - x1 >= 5 * (y2 - y1):
                ret["regions"].append({
                    "cls": TypeCls.LINE,
                    "region": list(topic["region"]),
                    "rotation": topic["rotation"]
                })
        #识别
        ret, cost_3 = self.identify_regions(img, ret)
        cost.update(cost_3)
        ret, cost_4 = self.construct_topics(data)
        cost.update(cost_4)
        calc, _ = self._spliter.detect_regions(img, {}, cache, "calc")
        ret = {
            "yyt": ret,
            "calc": calc,
            "image_width": w,
            "image_height": h
        }
        return ret, cost

    #检测识别+表格识别，针对试卷
    def identify_paper(self, img, use_det2=True, for_wrong_question=False):

        h, w = img.shape[:2]

        data, cost, cache = {}, {}, {}
        #检测
        ret, cost_1 = self._spliter.detect_regions(img, data, cache, "yyt")
        cost.update(cost_1)
        empty_line_topics = format_converter.find_empty_line_topic(ret["regions"])
        for topic in empty_line_topics:
            x1, y1, x2, y2 = topic["region"]
            if x2 - x1 >= 5 * (y2 - y1):
                ret["regions"].append({
                    "cls": TypeCls.LINE,
                    "region": list(topic["region"]),
                    "rotation": topic["rotation"],
                    "region_3point": topic["region_3point"]
                })
        
        #识别
        ret, cost_3 = self.identify_regions(img, ret)
        if not use_det2:
            table_res = self.table_detector.detect(img)
        else:
            img_det2_res = self.img_detector2.detect(img, for_wrong_question=for_wrong_question)
            ret, table_res = merge_det_res(ret, img_det2_res)
        square_cell_res = self.square_cell_detector.detect(img)
        ret = merge_square_cell_res(ret, square_cell_res)
        # 去除文本包含的情况
        ret = cast_overlap_region(ret, [1, 10])
        table_bboxes, text_cells, orientation = self.preprocess_table_ocr_result(table_res, ret)
        meta_data = {'table_bboxes': table_bboxes,
                     'text_cells': text_cells,
                     'orientation': orientation}
        exception = 0
        try:
            tables, confidences = self.cell_detector.detect(img, meta_data)
        except Exception as ex:
            logging.exception('images_to_word_v3 cell det error: {}'.format(ex))
            tables = []
            exception = 1
        tables = cast_bad_table(tables)
        # 保留单元格内的图片
        tables, ret = get_cell_pics(tables, ret)
        #去除大图内包含小图的情况
        ret = cast_overlap_region(ret, [3])
        #分离图片和文本
        pics, new_regions = split_pic_text(ret['regions'], tables)
        body = {"regions": new_regions,
            "tables": tables,
            "pics":pics,
            "colums":[]}
        return body, exception


    def preprocess_table_ocr_result(self, table_res, det_res, threshold=0.5):
        #若检测到table的存在则将det_res中属于table的内容部分划分到table当中
        tables = []
        orientation = 0
        table_bboxes = []
        index = 0
        for i, region in enumerate(table_res['regions0']):
            if region['det_score']>threshold:
                table_bboxes.append(region3point_to_bbox(region['region_3point']) )
        table_bboxes = sorted(table_bboxes, key=lambda x:(x[2][0]-x[0][0])*(x[2][1]-x[0][1]), reverse=True)
        casted = []
        for i, bbox in enumerate(table_bboxes):
            if i in casted:
                continue
            for j in range(i+1, len(table_bboxes)):
                if j in casted:
                    continue
                if iou([bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]], \
                       [table_bboxes[j][0][0], table_bboxes[j][0][1], table_bboxes[j][2][0], table_bboxes[j][2][1]])>0.75:
                    casted.append(j)
        tmp = []
        for i in range(len(table_bboxes)):
            if i in casted:
                continue
            tmp.append(table_bboxes[i])
        table_bboxes = tmp
        text_regions = []
        for i, region in enumerate(det_res['regions']):
            region['index'] = i 
            if region['cls'] in [1, 10]:
                text_regions.append(region)
            elif region['cls']==3:
                bbox = region['region']
                region_copy = region.copy()
                region_copy['result'] = [""]
                region_copy['cls'] = 1
                for table in table_bboxes:
                    txmin = min(table[0][0], table[1][0], table[2][0], table[3][0])
                    txmax = max(table[0][0], table[1][0], table[2][0], table[3][0])
                    tymin = min(table[0][1], table[1][1], table[2][1], table[3][1])
                    tymax = max(table[0][1], table[1][1], table[2][1], table[3][1])
                    if iou([txmin,tymin, txmax, tymax], bbox)>0.9 and iou([txmin,tymin, txmax, tymax], bbox, real_iou=True)<0.5 :
                        text_regions.append(region_copy)
                        break
        text_cells = [
            {
                'content': region['result'][0],
                'bbox':region3point_to_bbox(region['region_3point']) ,
                'confidence': 1.,
                'index': region['index'],
                'cls':region['cls']
            }
            for region in text_regions
        ]
        return table_bboxes, text_cells, orientation

    def detect(self, img, mode, region_type, subject=None):
        h, w = img.shape[:2]

        data, cost, cache = {}, {}, {}
        ret, cost_1 = self._spliter.detect_regions(img, data, cache, "yyt")
        cost.update(cost_1)

        empty_line_topics = format_converter.find_empty_line_topic(ret["regions"])
        for topic in empty_line_topics:
            x1, y1, x2, y2 = topic["region"]

            if x2 - x1 >= 5 * (y2 - y1):
                ret["regions"].append({
                    "cls": TypeCls.LINE,
                    "region": list(topic["region"]),
                    "rotation": topic["rotation"]
                })

        regions = []
        if mode in [0, 1]:
            ret, cost_3 = self.identify_regions(img, ret)
            cost.update(cost_3)
            if mode == 0:
                regions = ret["regions"]
            else:
                for region in ret["regions"]:
                    if region["cls"] in [1, 2, 3, 10]:
                        regions.append(region)
        elif mode in [2,3]:
            if subject:
                subject = subject.lower()
            if subject in ['chinese', 'english']:
                ret, cost_3 = self.identify_regions(img, ret)
                cost.update(cost_3)
                builder = TopicBuilder(subject)
                rets = builder.split(ret)
                if len(rets) > 0:
                    for ret in rets:
                        regions.extend(builder.build(ret))
                    region_type = 0
            if len(regions) == 0:
                for region in ret["regions"]:
                    if region["cls"] in [4, 5, 6, 7, 8, 9]:
                        regions.append(region)
        if mode==3:
            ret = cast_overlap_region(ret, [1, 10])
            ret, cost_3 = self.identify_regions(img, ret)
            cost.update(cost_3)
            ret = cast_region(ret)
            regions = generate_topic_with_lines(regions, ret["regions"])



        if region_type == 1:
            for item in regions:
                region = item["region"] + [item["rotation"]]
                region = rotated_rect_utils.lefttop_reightbottom_theta_bound_box(region, True)
                item["region"] = region
                item["rotation"] = 0
        ret = {
            "image_width": w,
            "image_height": h,
            "regions": regions
        }
        return ret, cost

    def identify_type_regions(self, data, identify_types, identify_func, crop_func, cost_name):
        transfer = DataTransfer(data)
        start_time = time.time()
        crops, identified_regions = [], []

        #hw_crops = []
        for data_item in transfer.to_array():
            for region in data_item["regions"]:
                if region["cls"] in identify_types:
                    crop = crop_func(region["region"] + [region["rotation"]], data_item)
                    #crop = cv2.imread("/mnt/server_data2/data/mathlens/images/4.0_crop/1790_2.jpg")[:,:,::-1]
                    crops.append(crop)
                    #hw_crop = rotated_rect_utils.lefttop_rightbottom_theta_crop_img(self.hw_img, region["region"] + [region["rotation"]])
                    #hw_crops.append(hw_crop)
                    identified_regions.append(region)

        if len(crops) != 0:
            values = identify_func(crops)
            for region, value in zip(identified_regions, values):
                value[0] = remove_dollar(value[0])
                value[0] = ignore_hw(value[0])
                region["result"] = value
                #print(value[0])
        # root_dir = "/mnt/server_data2/code/projects/shijuanbao_engine_projects/test_data/det_by/0227_crop"
        #
        # predict = open(os.path.join(os.path.dirname(root_dir),  "train_mathlens_0227.txt"), "a")
        # for i in range(len(crops)):
        #     if len(identified_regions[i]["result"][0]) < 4:
        #        continue
        #     result = identified_regions[i]["result"][0]
        #     if '{' not in result or len(result)<20:
        #         continue
        #     if result.count('{') < 3:
        #         continue
        #     img = crops[i]
        #     print(result)
        #     # if self.index%2!=0:
        #     cv2.imwrite(os.path.join(root_dir,  'img_%d'%self.index + ".jpg"), img)
        #     predict.write('img_%d'%self.index + ".jpg" + "\t" + result + "\n")
        #     self.index += 1


        end_time = time.time()
        return transfer.org_form(), {cost_name: end_time - start_time}

    def identify_regions(self, img, data):
        def crop_func(reg, data_item):
            crop = rotated_rect_utils.lefttop_rightbottom_theta_crop_img(img, reg)
            return crop


        identify_type = TypeCls.LINE
        models = {TypeCls.LINE: "std_identify"}
        identify_funcs = {
            TypeCls.LINE: lambda x: self._identifier.batch_identify(x),
        }
        data, cost = self.identify_type_regions(data, [identify_type, TypeCls.STUDENT], identify_funcs[identify_type],
                                                crop_func, models[identify_type])

        return data, cost

    def identify_tmp(self, img, only_ocr=False):
        #self.hw_img = hw_img
        h, w = img.shape[:2]

        data, cost, cache = {}, {}, {}
        # 检测
        if only_ocr:
            ret = {"regions": [{"cls":1, "region":[0,0,w,h], "rotation":0}]}
        else:
            ret, cost_1 = self._spliter.detect_regions(img, data, cache, "yyt", for_html=True)
            cost.update(cost_1)
            empty_line_topics = format_converter.find_empty_line_topic(ret["regions"])
            for topic in empty_line_topics:
                x1, y1, x2, y2 = topic["region"]
                if x2 - x1 >= 5 * (y2 - y1):
                    ret["regions"].append({
                        "cls": TypeCls.LINE,
                        "region": list(topic["region"]),
                        "rotation": topic["rotation"],
                        "region_3point": topic["region_3point"]
                    })

        # 识别
        ret, cost_3 = self.identify_regions(img, ret)
        return ret


    def construct_topics(self, data):
        regions = data["regions"]

        start = time.time()
        #从上到下排序
        regions.sort(key=lambda x: x["region"][1])
        topics, dangle_items = format_converter.to_topic_json_format(regions, throw_when_no_topic_found=False)
        end = time.time()

        data["topics"] = topics
        data["dangle_items"] = dangle_items
        #这里的topics和dangle_items对应的是我所定义的para，text
        return data, {"construct_topics": end - start}

    #识别
    def indentify_regions1(self, result, img):
        regions = result["regions0"]
        
        regions.sort(key = lambda res : res["region_3point"][2] - res["region_3point"][0])#宽度大小
        
        split_map = {}
        # default_list = regions
        # for reg in regions:
        #     cls = reg.get("cls", 0)
        #     default_list += [reg] 
        self.crop_img(img, regions)
        
    def crop_img(self, img, regions):
    #         crops = []
        h, w = img.shape[:2]
        for region in regions:
            try: #切片可能越界失败
                if "crop" in region:
                    continue
                reg = region["region_3point"] 
                x1,y1,x2,y2,x3,y3 = reg
                x1 = int(max(0, x1 - 3))
                x2 = int(min(w, x2 + 3))
                x3 = int(min(w, x3 + 3))
                
                y1 = int(max(0, y1 - 2))
                y2 = int(max(0, y2 - 2))
                y3 = int(min(h, y3 + 2))
                
                if y1 == y2:
    #                     region["region_3point"] = [x1,y1,x2,y2,x3,y3]
                    crop = img[y1:y3, x1:x3]
                else:
                    #inclined box
                    reg = [x1,y1,x2,y2,x3,y3]
                    crop = rotated_rect_utils.three_points_crop_img(img, reg)
                    
                region["crop"] = crop
            except Exception as err:
    #                 traceback.print_exc()
                logging.info('crop_img error: {}'.format(err))
                pass
            
        return 

    def visualize_det_result(self, img, regions):
        img = img.astype(np.uint8)
        for region in regions:
            try:
                bbox = region['region']
            except:
                region_3point = region['region_3point']
                bbox = [region_3point[0],region_3point[1],region_3point[4],region_3point[5]]
            x1,y1,x2,y2 = [int(x) for x in bbox]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cls = region['cls']
            cv2.putText(img, str(cls), (x1, max(0, y1-5)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            if 'contents' in region:
                for content in region['contents']:
                    try:
                        cbbox = content['region']
                    except:
                        region_3point = content['region_3point']
                        cbbox = [region_3point[0],region_3point[1],region_3point[4],region_3point[5]]
                    cx1,cy1,cx2,cy2 = [int(x) for x in cbbox]
                    cv2.rectangle(img, (cx1,cy1), (cx2,cy2), (255,0,0), 2)
                    cls = content['cls']
                    cv2.putText(img, str(cls), (cx1, max(0, cy1-5)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        cv2.imwrite('visualize_img.jpg', img)  

    def visualize_cell_result(self, image, table_bboxes, tables, save_path=None):
        drawed_image = image.copy()
        for i, cells in enumerate(tables):
            drawed_image = draw_bbox(drawed_image, cells)
            #drawed_image = draw_bbox(drawed_image, [{'bbox': table_bboxes[i]}], color=(0,0,255))
        drawed_image = cv2.hconcat([image, drawed_image])
        h, w = drawed_image.shape[:2]
        shrinkratio = min(1600 / w, 1)
        if shrinkratio < 1:
            drawed_image = cv2.resize(drawed_image, (int(w * shrinkratio), int(h * shrinkratio)))
        if save_path!=None:
            cv2.imwrite(save_path, drawed_image)
    
    def generate_table_json(self, tables, confidences, meta_data):
        # tables, confidences = image2tables(image, meta_data)
        table_bboxes = meta_data['table_bboxes']

        json_content = {
            'Tables': [],
            'orientation': meta_data['orientation']
        }
        for i, cells in enumerate(tables):
            table = {
                'cells': [],
                'row_num': -1,
                'col_num': -1,
                'region': [c for p in table_bboxes[i] for c in p],
                'rely_line': True,
                'confidence': confidences[i]
            }

            for cell in cells:
                pcs = cell['bbox']
                lcs = cell['logical_coordinates']
                texts = cell['texts']

                table['row_num'] = max(table['row_num'], lcs['endrow'] + 1)
                table['col_num'] = max(table['col_num'], lcs['endcol'] + 1)

                table['cells'].append({
                    'startrow': lcs['startrow'],
                    'endrow': lcs['endrow'] + 1,
                    'startcol': lcs['startcol'],
                    'endcol': lcs['endcol'] + 1,
                    'region': [c for p in pcs for c in p],
                    'line': 
                    [
                        {
                            'alignments': 'left',
                            'text': ' '.join([col['content'] for col in row])
                        } 
                        for row in texts
                    ],
                    'text_rects': 
                    [
                        [
                            {
                                'raw_text': col['content'],
                                'region': [c for p in col['bbox'] for c in p],
                                'confidence': col['confidence'],
                                'index': col['index']
                            } 
                            for col in row
                        ]
                        for row in texts
                    ],
                    'h_align': 'left',
                })
            
            json_content['Tables'].append(table)

        return json_content


    def paging(self, img):
        data, cost, cache = {}, {}, {}
        ret, cost_1 = self._spliter.detect_regions(img, data, cache, "yyt")
        cost.update(cost_1)

        empty_line_topics = format_converter.find_empty_line_topic(ret["regions"])
        for topic in empty_line_topics:
            x1, y1, x2, y2 = topic["region"]

            if x2 - x1 >= 5 * (y2 - y1):
                ret["regions"].append({
                    "cls": TypeCls.LINE,
                    "region": list(topic["region"]),
                    "rotation": topic["rotation"]
                })
        ret, cost_3 = self.identify_regions(img, ret)
        cost.update(cost_3)
        builder = TopicBuilder("chinese")
        rets = builder.split(ret, item=False)
        rets.sort(key=lambda x: x[0])
        new_rets = []
        for i, r in enumerate(rets):
            if i == 0:
                new_rets.append(r)
            else:
                last_r = new_rets[-1]
                max_x1 = max(r[0], last_r[0])
                min_x2 = min(r[1], last_r[1])
                if r[1]<=last_r[1] or min_x2 - max_x1 >= 0.7*(r[1]-r[0]) or min_x2 - max_x1 >= 0.7*(last_r[1]-last_r[0]):
                    new_rets[-1] = [min(last_r[0], r[0]), max(last_r[1], r[1])]
                else:
                    new_rets.append(r)
        rets = new_rets
        rets.sort(key=lambda x: x[0])
        topics = {}
        for item in ret["regions"]:
            new = True
            for r in rets:
                if r[0] <= item["region"][0] <= r[1] or r[0] <= item["region"][2] <= r[1]:
                    new = False
                    break
            if new:
                k = len(rets)
                for i in range(len(rets)):
                    r = rets[i]
                    if item["region"][2] < r[0]:
                        k = i
                        break
                if k in topics:
                    topics[k][0] = min(topics[k][0], item["region"][0])
                    topics[k][1] = max(topics[k][1], item["region"][2])
                else:
                    topics[k] = [item["region"][0], item["region"][2]]
        ret_widths = []
        for r in rets:
            ret_widths.append(r[1] - r[0])
        rets.extend([(topics[k][0], topics[k][1]) for k in topics])
        rets.sort(key=lambda x: x[0])
        if len(rets)<=1 or len(ret_widths)==0:
            return rets
        #合并宽度很小的页
        merged_rets = []
        merge_last = False
        for i in range(len(rets)):
            ret = list(rets[i])
            ret_width = ret[1] - ret[0]
            if ret_width >= 0.5 * min(ret_widths) and not merge_last:
                merged_rets.append(ret)
            elif ret_width>=0.5 * min(ret_widths) and merge_last:
                ret[0] = rets[i-1][0]
                merged_rets.append(ret)
                merge_last = False
            else:
                merge_last = True
        if merge_last:
            merged_rets[-1][1] = rets[-1][1]
        rets = merged_rets
        return rets

