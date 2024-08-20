import time, math

import numpy as np


def get_bb_box(_box):
    x1, y1, x2, y2, x3, y3 = _box
    x4 = x1 + x3 - x2
    y4 = y1 + y3 - y2
    x_min, x_max = min([x1,x2,x3,x4]), max([x1,x2,x3,x4])
    y_min, y_max = min([y1,y2,y3,y4]), max([y1,y2,y3,y4])
    return [x_min, y_min, x_max, y_max]

def recalc_merge_region(region_b, region_s):
    x11, y11, x12, y12, x13, y13 = region_b
    x21, y21, x22, y22, x23, y23 = region_s
    theta1 = math.degrees(math.atan2(y12 - y11, x12 - x11))
    theta2 = math.degrees(math.atan2(y22 - y21, x22 - x21))
    
    if math.fabs(theta1 - theta2) > 2: #两者角度不能相差太多
        return False, []
    
    if x21 < x11 and x22 > x11 and x22 < x12:#p2p3不动
        x11_n = x21
        y11_n = y11 - ( (x11-x11_n) * (y12 - y11) /(x12-x11) )
        nreg = [x11_n, y11_n, x12, y12, x13, y13]
        return True, nreg
        
    if x21 < x12 and x22 > x12 and x21 > x11:#p1p4不动
        x12_n = x22
        y12_n = y12 + ( (x12_n-x12) * (y12 - y11) /(x12-x11) )
        x14 = x11 + x13 - x12
        y14 = y11 + y13 - y12
        x13_n = x12_n + x14 - x11
        y13_n = y12_n + y14 - y11
        nreg = [x11, y11, x12_n, y12_n, x13_n, y13_n]
        return True, nreg
        
    return False, []


#在det之后,进行去重
#不知道reg的结果
def remove_cover_regions0(regions, h,w):
    start_time = time.time()
    
    regions12_1 = [reg for reg in regions if reg["cls"] in [1,2] ]
    regions_other = [reg for reg in regions if reg["cls"] not in [1, 2] ]
    
    regions12_1.sort(key = lambda res : (res["region_3point"][1], res["region_3point"][0]))
    
    regions2_new = []
    b_cover_list =[]
    for ind, reg in enumerate( regions12_1):
        if ind in b_cover_list:
            continue
        region_3point1 = reg["region_3point"]

        x1_min, y1_min, x1_max, y1_max = get_bb_box(region_3point1)
        mid_y = (y1_min + y1_max) / 2
        y1_h = y1_max - y1_min
        x1_w = x1_max - x1_min
        need_delete = False
        del_num = 0
        
        if x1_w == 0 or y1_h == 0:
            continue
        for ind2 in range(ind+1, len(regions12_1)):
            if ind2 in b_cover_list:
                continue
            
            need_delete = False
            del_num = 0
            reg2 = regions12_1[ind2]
            region_3point2 = reg2["region_3point"]
            x2_min, y2_min, x2_max, y2_max = get_bb_box(region_3point2)
            if mid_y < y2_min:
                continue
            
            if y2_max < y1_min or y1_max < y2_min or x1_max < x2_min or x2_max < x1_min:
                continue
            
            y2_h = y2_max - y2_min
            x2_w = x2_max - x2_min
            if x2_w == 0 or y2_h == 0:
                continue
        
            y_over = min(y1_max, y2_max)  - max(y1_min, y2_min)
            yover_rat = 2 * y_over / (y1_h + y2_h)
            if yover_rat < 0.7: #fix PE-759
                continue
            
            if x1_w < x2_w:
                del_num =1 #maybe del reg1
            else:
                del_num =2 #maybe del reg1
                
            x_over = min(x1_max, x2_max)  - max(x1_min, x2_min)
            xover_rat = x_over / min(x1_w, x2_w)
            xover_rat2 = x_over / min(y1_h, y2_h)
            
            if  xover_rat > 0.98: #全覆盖
                continue #忽略,在第二阶段的时候, 会去重
            if  xover_rat2 < 0.02: #太短 或者 没有覆盖
                continue #忽略 
            
            #recal_new reg
            bmatch, region_3point_b = False, []
            if del_num == 1:
                bmatch, region_3point_b = recalc_merge_region(region_3point2, region_3point1)
            else:
                bmatch, region_3point_b = recalc_merge_region(region_3point1, region_3point2)
                
            if bmatch:
                if del_num == 1:
                    need_delete = True
                    reg2["region_3point0"] = region_3point2
                    reg2["region_3point"] = region_3point_b
                    #set region_3point_b
                    break
                else: #del2
                    b_cover_list.append(ind2)
                    reg["region_3point0"] = region_3point1
                    reg["region_3point"] = region_3point_b
                    continue
                    
        if not need_delete or del_num != 1:
            regions2_new.append(reg)
    
    regions_new = regions2_new + regions_other
    end_time = time.time()
    return regions_new

def remove_cover_regions(regions, h,w):
    # try:
    return remove_cover_regions0(regions, h,w)
    # except Exception as e:
    #     traceback.print_exc()
    #     logging.error("remove_cover_regions error : {}".format(e))    
    #     return regions

#单边iou,计算box1和box2交集占box1的比例
def one_side_iou(box1, box2):
    xmin = max(box1[0], box2[0])
    xmax = min(box1[2], box2[2])
    ymin = max(box1[1], box2[1])
    ymax = min(box1[3], box2[3])
    inter_area = max(0, ymax-ymin) * max(0, xmax-xmin)
    area =  max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    return inter_area/area

def nms(boxes, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
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
    return boxes[keep]


def cast_full_text_pic(pic_res, text_res):
    tmp = []
    for pic in  pic_res:
        inside_texts = []
        pic_in_text_flag = False
        for text in text_res:
            text_bbox, text_result = text
            iou = one_side_iou(text_bbox[:-1], pic[:-1])
            if iou>0.8:
                inside_texts.append(text)
            iou2 = one_side_iou(pic[:-1], text_bbox[:-1])
            if iou2>0.8:
                pic_in_text_flag = True
                break
        if pic_in_text_flag:
            continue
        cur_pic_area = (pic[2]-pic[0]) * (pic[3]-pic[1])
        inside_text_area = 0.
        inside_option = False
        for inside_text in inside_texts:
            text_bbox, text_result = inside_text
            inside_text_area += (text_bbox[2]-text_bbox[0]) * (text_bbox[3]-text_bbox[1])
            for char_ in ['A.', 'B.', 'C.', 'D.',
                          '1.', '2.', '3.', '4.']:
                if len(text_result[0])>=4 and char_ ==text_result[0][:2]:
                    inside_option = True
            for char_ in ['A', 'B', 'C', 'D']:
                if len(text_result[0])>=4 and char_ ==text_result[0][0]:
                    inside_option = True
        text_ratio = inside_text_area/cur_pic_area
        if inside_option and text_ratio > 0.1 and pic[-1]<0.7:
            continue
        # if inside_text_area/cur_pic_area>0.4:
        #     continue
        tmp.append([len(inside_texts), text_ratio, pic])
    new_pic_res = []
    for pic in tmp:
        conf = pic[2][-1]
        text_num = pic[0]
        text_ratio = pic[1]
        if conf<0.25:
            continue
        elif conf<=0.5 and (text_ratio>0.1 or text_num>=8):
            continue
        elif conf>0.5 and (text_ratio>0.4):
            continue
        new_pic_res.append(pic[2])
    return new_pic_res

def merge_det_res(det_res1, det_res2):
    #合并两个检测模型的结果
    # det_res1为seq_line斜框检测模型结果
    # det_res2为yolov7，表格图片下划线检测模型结果
    # 主要进行:
    # 1.合并两个图片的检测结果，nms去重，去除大框中的小框
    # 2.增加表格和下划线结果
    det_pic_res1 = []
    det_text_res = []
    others = []
    for object in det_res1['regions']:
        if object['cls']==3:
            det_pic_res1.append(np.array(object['region']+[1.0]))
        elif object['cls']==1:
            det_text_res.append([object['region'] + [0.9], object['result']])
            others.append(object)
        else:
            others.append(object)
    det_pic_res2 = []
    det_table_res = []
    det_underline_res = []
    for i in range(len(det_res2['clses'])):
        bbox = det_res2['boxes'][i].tolist()
        score = det_res2['scores'][i]
        cls = det_res2['clses'][i]
        bbox.append(score)
        if cls==0:
            det_pic_res2.append(np.array(bbox))
        elif cls==1:
            det_table_res.append(np.array(bbox))
        else:
            det_underline_res.append(np.array(bbox))
    det_pic_res2 = cast_full_text_pic(det_pic_res2, det_text_res)
    det_pic = np.array(det_pic_res1+det_pic_res2)
    if len(det_pic)!=0:
        det_pic = nms(det_pic, 0.5)
    new_regions = []
    for i in range(det_pic.shape[0]):
        region = det_pic[i].tolist()
        new_regions.append({'region':region[:4],
                            'cls':3, 'rotation':0,
                            'region_3point':[region[0], region[1], region[2],region[3],region[2],region[3]]
                            })
    new_regions.extend(others)
    for i in range(len(det_underline_res)):
        region = det_underline_res[i].tolist()
        new_regions.append({'region': region[:4],
                            'cls': 5, 'rotation': 0,
                            'region_3point': [region[0], region[1], region[2], region[3], region[2], region[3]]
                            })
    ret = {'regions':new_regions}
    table_res = []
    for i in range(len(det_table_res)):
        region = det_table_res[i].tolist()
        table_res.append({'region': region[:4],
                            'cls': 6, 'det_score':region[4],'rotation': 0,
                            'region_3point': [region[0], region[1], region[2], region[3], region[2], region[3]]
                            })
    return ret, {'regions0':table_res}


def merge_square_cell_res2(det_res1, square_cell_res):
    det_pic_res1 = []
    others = []
    regions = det_res1['regions']
    delete = [False]*len(regions)
    for object in det_res1['regions']:
        if object['cls']==3:
            det_pic_res1.append(np.array(object['region']+[1.0]))
        else:
            others.append(object)

    square_cell_regions = []
    for i in range(len(square_cell_res['clses'])):
        if square_cell_res['clses'][i]!=3:
            continue
        bbox = square_cell_res['boxes'][i].tolist()
        inside_pic = False
        for j in range(len(det_pic_res1)):
            pic = det_pic_res1[j]
            iou = one_side_iou(pic[:-1], bbox)
            if iou>0.5:
                inside_pic = True
                break
        if  not inside_pic:
            square_cell_regions.append({'region':bbox,
                                        'cls':3, 'rotation':0,
                                        'region_3point':[bbox[0], bbox[1], bbox[2],bbox[3],bbox[2],bbox[3]]
                                        })
    regions2 = []
    regions1 = []
    for k, region in enumerate(regions):
        if region['cls']==1:
            text_x1, text_y1, text_x2, text_y2 = region['region']
            split_bboxes = []
            for square_cell in square_cell_regions:
                square_cell_x1, square_cell_y1, square_cell_x2, square_cell_y2 = [int(uu) for uu in square_cell['region']]
                if square_cell_x1>text_x1 and square_cell_y1<=text_y1 and square_cell_x2<text_x2 and square_cell_y2>=text_y2:
                    split_bboxes.append([square_cell_x1, square_cell_y1, square_cell_x2, square_cell_y2])
            if len(split_bboxes)!=0:
                split_bboxes = sorted(split_bboxes, key=lambda x:x[0])
                start_x = text_x1
                for bbox in split_bboxes:
                    end_x = bbox[0]
                    regions2.append({'region':[start_x, text_y1, end_x, text_y2],
                                    'cls':1, 'rotation':region['rotation'], 'region_3point':[start_x, text_y1, end_x, text_y2, end_x, text_y2]})
                    print(start_x, text_y1, end_x, text_y2)
                    start_x = bbox[2]
                regions2.append({'region':[start_x, text_y1, text_x2, text_y2],
                                'cls':1, 'rotation':region['rotation'], 'region_3point':[start_x, text_y1, text_x2, text_y2, text_x2, text_y2]})
                delete[k] = True
    for k, region in enumerate(regions):
        if not delete[k]:
            regions1.append(region)
    regions1.extend(square_cell_regions)
    det_res1['regions'] = regions1
    ret = {'regions':regions2}
    return det_res1, ret


def merge_square_cell_res(det_res1, square_cell_res):
    det_pic_res1 = []
    others = []
    regions = det_res1['regions']
    delete = [False]*len(regions)
    for idx, object in enumerate(det_res1['regions']):
        if object['cls']==3:
            det_pic_res1.append(np.array(object['region']+[1.0]+[idx]))
        else:
            others.append(object)

    pic_to_suqare_cell = []
    for i in range(len(square_cell_res['clses'])):
        if square_cell_res['clses'][i]!=3:
            continue
        score = square_cell_res['scores'][i]
        if score < 0.5:
            continue
        bbox = square_cell_res['boxes'][i].tolist()
        inside_pic = False
        for j in range(len(det_pic_res1)):
            pic = det_pic_res1[j]
            iou = one_side_iou(pic[:-2], bbox)
            if iou>0.5:
                inside_pic = True
                pic_to_suqare_cell.append(pic[-1])
                break
        inText = False
        for k, region in enumerate(regions):
            if region['cls'] == 1 :
                text_x1, text_y1, text_x2, text_y2 = region['region']
                square_cell_x1, square_cell_y1, square_cell_x2, square_cell_y2 = bbox
                inter_y1 = max(text_y1, square_cell_y1)
                inter_y2 = min(text_y2, square_cell_y2)
                iou_y = (inter_y2-inter_y1)/(text_y2-text_y1)
                if square_cell_x1 > text_x1 and square_cell_x2 < text_x2 and iou_y>0.7:
                    inText = True
                #    if len(region['result'])!=0:
                #        region['result'][0] = region['result'][0].replace('\@','⿱').replace('@','⿱')
                #    break
        if not inText and not inside_pic:
            det_res1['regions'].append({'region':bbox,
                                        'cls':3, 'rotation':0,
                                        'region_3point':[bbox[0], bbox[1], bbox[2],bbox[3],bbox[2],bbox[3]],
                                        'is_square_cell':True
                                        })
    for idx in pic_to_suqare_cell:
        det_res1['regions'][int(idx)]['is_square_cell'] = True
    return det_res1