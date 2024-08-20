# -*-coding:utf-8-*-
import math, cv2
import numpy as np
import wordninja
import re
from lxml import etree
import latex2mathml.converter
            
def move_bbox(bbox, offset_x, offset_y):
    for i, coord in enumerate(bbox):
        if isinstance(coord, list):
            bbox[i][0] -= offset_x
            bbox[i][1] -= offset_y
            continue
        if i%2==0:
            bbox[i] -= offset_x
        else:
            bbox[i] -= offset_y
    return bbox


def bbox_iou(bbox1, bbox2, real_iou=False):
    para_x1, para_y1, para_x2, para_y2 = bbox1
    x1, y1, x2, y2 = bbox2
    inter_x1 = max(para_x1, x1)
    inter_y1 = max(para_y1, y1)
    inter_x2 = min(para_x2, x2)
    inter_y2 = min(para_y2, y2)
    inter_area = max(0, (inter_y2 - inter_y1)) * max(0, (inter_x2 - inter_x1))
    cur_area = (x2 - x1) * (y2 - y1)
    if real_iou:
        out_x1 = min(para_x1, x1)
        out_y1 = min(para_y1, y1)
        out_x2 = max(para_x2, x2)
        out_y2 = max(para_y2, y2)
        cur_area = (out_y2 - out_y1) * (out_x2 - out_x1)
    if cur_area==0:
        return 0
    ratio = inter_area * 1. / cur_area
    return ratio

def cast_region(target_regions, overlap_regions, threshold=0.65):
    output = []
    for region in target_regions:
        cx1, cy1, cx2, cy2 = region['bbox']
        flag = False
        for tmp in overlap_regions:
            bx1, by1, bx2, by2 = tmp['bbox']
            if tmp['type'] == 'paragraph':
                continue
            if (cx1 > bx1 and cy1 > by1 and cx2 < bx2 and cy2 < by2) or bbox_iou([bx1, by1, bx2, by2], [cx1, cy1, cx2, cy2]) > threshold:
                flag = True
                break
        if not flag:
            output.append(region)
    return output

def crop_square_cell(pics, texts):
    new_pics = []
    for region in pics:
        if not region['is_square_cell']:
            new_pics.append(region)
            continue
        cx1, cy1, cx2, cy2 = region['bbox']
        for tmp in texts:
            if tmp['type']!='text':
                continue
            if not is_pinyin(tmp['text']):
                continue
            bx1, by1, bx2, by2 = tmp['bbox']
            iou = bbox_iou([cx1, cy1, cx2, cy2], [bx1, by1, bx2, by2])
            if iou>0.7 and by2<(cy1+cy2)/2 and by2>cy1:
                region['bbox'] = cx1, by2, cx2, cy2
                break
        if region['bbox'][3]-region['bbox'][1]<=2 or region['bbox'][2]-region['bbox'][0]<=2:
            continue
        new_pics.append(region)
    return new_pics

def crop_pics(pics, texts):
    f = False
    new_pics = []
    for region in pics:
        cx1, cy1, cx2, cy2 = region['bbox']
        for tmp in texts:
            if tmp['type']!='text':
                continue
            bx1, by1, bx2, by2 = tmp['bbox']
            iou = bbox_iou([cx1, cy1, cx2, cy2], [bx1, by1, bx2, by2])
            if iou>0.1 and iou<0.7:
                if cy1>by1 and cy1<by2:
                    cy1 = by2
                    f = True
                elif cy2>by1 and cy2<by2:
                    cy2 = by1
                    f = True
                elif cx1>bx1 and cx1<bx2:
                    cx1 = bx2
                    f = True
                else:
                    cx2 = bx1
        region['bbox'] = cx1, cy1, cx2, cy2
        if cx2-cx1<=2 or cy2-cy1<=2 or cx2==0 or cy2==0:
            continue
        new_pics.append(region)
    return new_pics, f

def crop_tables(tables, texts):
    f = False
    new_tables = []
    for region in tables:
        cx1, cy1, cx2, cy2 = region['bbox']
        for tmp in texts:
            if tmp['type']!='text':
                continue
            bx1, by1, bx2, by2 = tmp['bbox']
            iou = bbox_iou([cx1, cy1, cx2, cy2], [bx1, by1, bx2, by2])
            if iou>0.2 and iou<0.7:
                if cy1>by1 and cy1<by2:
                    cy1 = by2
                    f = True
                elif cy2>by1 and cy2<by2:
                    cy2 = by1
                    f = True
        region['bbox'] = cx1, cy1, cx2, cy2
        if cx1>=cx2 or cy1>=cy2 or cx2==0 or cy2==0:
            continue
        new_tables.append(region)
    return new_tables, f

def cast_pics(target_regions, overlap_regions):
    output = []
    for region in target_regions:
        cx1, cy1, cx2, cy2 = region['bbox']
        area1 = (cx2-cx1) * (cy2-cx1)
        flag = False
        for tmp in overlap_regions:
            bx1, by1, bx2, by2 = tmp['bbox']
            area2 = (bx2-bx1)*(by2-by1)
            if (cx1 > bx1 and cy1 > by1 and cx2 < bx2 and cy2 < by2) or \
                    bbox_iou([bx1, by1, bx2, by2], [cx1, cy1, cx2, cy2]) > 0.7:
                flag = True
                if (cx1 > bx1 and cy1 > by1 and cx2 < bx2 and cy2 < by2):
                    for cell in tmp['cells']:
                        cell_bbox = cell['bbox']
                        if bbox_iou([cell_bbox[0], cell_bbox[1], cell_bbox[4], cell_bbox[5]], [cx1, cy1, cx2, cy2])>0.9:
                            flag = False
                            break
                break
        if not flag:
            output.append(region)
    return output

def is_all_english(text):
    for uu in text:
        if ord(uu)>122 or ord(uu)<65 or (ord(uu)>91 and ord(uu)<97):
            return False
    return True

def regular_split(text):
    p = 0
    new_text = ''
    sub_text = ''
    flag = False
    while p<len(text):
        if is_all_english(text[p]) and not flag:
            new_text += sub_text
            sub_text = text[p]
            flag = True
        elif is_all_english(text[p]) and flag:
            sub_text += text[p]
        elif not is_all_english(text[p]) and not flag:
            new_text += text[p]
        else:
            if len(sub_text)>=10:
                text_splits = wordninja.split(sub_text)
                new_text += ' '.join(text_splits)
            else:
                new_text += sub_text
            new_text += text[p]
            sub_text = ''
            flag = False
        p += 1
    if flag:
        if len(sub_text) >= 10:
            text_splits = wordninja.split(sub_text)
            new_text += ' '.join(text_splits)
        else:
            new_text += sub_text
    return new_text


def cast_texts(doc_texts):
    region_sorted = sorted(doc_texts, key=lambda x:(x['region'][2]-x['region'][0])*(x['region'][3]-x['region'][1]), reverse=True)
    casted_idx = []
    for idx  in range(len(region_sorted)):
        if region_sorted[idx]['cls']!=1:
            continue
        if idx in casted_idx:
            continue
        cx1, cy1, cx2, cy2 = region_sorted[idx]['region']
        for j in range(idx+1, len(region_sorted)):
            if region_sorted[j]['cls'] != 1:
                continue
            bx1, by1, bx2, by2 = region_sorted[j]['region']
            if bbox_iou([cx1, cy1, cx2, cy2], [bx1, by1, bx2, by2]) > 0.5:
                casted_idx.append(j)
    #print(casted_idx)
    new_region = []
    for i, region in enumerate(region_sorted):
        if i not in casted_idx:
            new_region.append(region)
    return new_region

def get_ratoted_box(bbox, rotation):
    if rotation == 0:
        return bbox
    x1, y1, x3, y3 = bbox
    w = x3 - x1
    h = y3 - y1
    x2 = x3
    y2 = round(y1 + (w) * math.tan(rotation * 2 * 3.1415 / 360), 1)
    x3 = round(x2 - (h) * math.tan(rotation * 2 * 3.1415 / 360), 1)
    y3 = y2 + h
    x4 = x3 - w
    y4 = y1 + h

    xmin = max(0, min(x1, x4))
    ymin = max(0, min(y1, y2))
    xmax = max(x2, x3)
    ymax = max(y3, y4)
    return xmin, ymin, xmax, ymax

def strip_overlap_char(text, char, num):
    new_text = ''
    char_s = ''
    f = False
    for uu in text:
        if uu!=char and not f:
            new_text+=uu
        elif uu!=char and f:
            f = False
            if len(char_s) >= num:
                new_text+=char*num
            else:
                new_text+=char_s
            char_s = ''
            new_text += uu
        elif not f:
            char_s += uu
            f = True
        else:
            char_s += uu
    if char_s!='':
        if len(char_s) >= num:
            new_text += char * num
        else:
            new_text += char_s
    return new_text


def merge_latex_formula(line_text):
    #return line_text
    formula_char = ['1', '2', '3', '4', '5', '6', '7', '8' ,'9', '0', '.', '＞', '＜', '＝', '(', ')', '÷', '+', '-', '×', '＝']
    # ord(uu) > 122 or ord(uu) < 65 or (ord(uu) > 91 and ord(uu) < 97)
    for i in range(65, 92):
        formula_char.append(chr(i))
    for i in range(97, 123):
        formula_char.append(chr(i))
    if len(line_text) <=1:
        return line_text
    rets = []
    cur_text = ''
    idx = 0
    while idx<len(line_text):
        line_text[idx] = line_text[idx].replace('(\u3000\u3000)', '()')
        if (len(line_text[idx])>0 and line_text[idx][0]=='^') or (idx<len(line_text)-1 and len(line_text[idx+1])>0 and line_text[idx+1][0]=='^'):
            if cur_text!='':
                rets.append(cur_text)
                cur_text = ''
            rets.append(line_text[idx])
            idx += 1
            continue
        if 'latex' in cur_text and 'latex' in line_text[idx]:
            cur_text += line_text[idx]
            idx += 1
        elif 'latex' in cur_text and 'latex' not in line_text[idx]:
            j = 0
            while j<len(line_text[idx]):
                if line_text[idx][j] in formula_char:
                    j += 1
                else:
                    break
            cur_text += line_text[idx][:j]
            rets.append(cur_text)
            cur_text = line_text[idx][j:]
            idx += 1
        elif 'latex' not in cur_text and 'latex' in line_text[idx]:
            j = len(cur_text)-1
            while j>-1:
                if cur_text[j] in formula_char:
                    j -= 1
                else:
                    break
            if j>=0:
                rets.append(cur_text[:j+1])
            cur_text = cur_text[j+1:] + line_text[idx]
            idx += 1
        else:
            cur_text += line_text[idx]
            idx += 1
    rets.append(cur_text)
    return rets

def replace_text(text):
    #['(', '（'], [')', '）'],
    replace_pair = [ ['○', '〇'], ['□', '囗'], ['>', '＞'], ['<', '＜'], ['=', '＝'],['①','① '], ['②', '② '], ['③', '③ '],\
                    ['④','④ '],['⑤','⑤ '],['⑥','⑥ '],['⑦','⑦ '],['⑧','⑧ '],['⑨','⑨ '],['⑪', '⑪ '],['⑫', '⑫ '],['⑬','⑬ '],['⑭','⑭ '],\
                     ['⑮', '⑮ '], ['()', '()'], ['(', '（'], [')', '）'], ['\\sqrt', '¤']]

   #
    #, ['…','·']]
    for char_pair in replace_pair:
        text = text.replace(char_pair[0], char_pair[1])
    if '···' in text:
        text = strip_overlap_char(text, '·', 3)
    if '_ _ _' in text:
        text = text.replace(' _ _ ', '')
    if '___' in text:
        text = strip_overlap_char(text, '_', 2)
    if '@@' in text:
        text = strip_overlap_char(text, '@', 2)
    if '...' in text:
        text = strip_overlap_char(text, '.', 3)
    if '------' in text:
        text = strip_overlap_char(text, '-', 3)
    if '（）（）（）（）' in text:
        text = text.replace('（）', '')
    return text

def replace_mark(text):
    en_char = [',', '?']
    cn_char = ['，', '？']
    for i in range(len(en_char)):
       text = text.replace(en_char[i], cn_char[i])
    return text

def is_need_group(line):
    if len(line['sentences'])>1:
        return False
    title_head = [ '① ', '② ', '③ ', '④ ', '⑤ ','⑥ ','⑦ ', '⑧ ', '⑨ ', '()', '(\u3000', 'I.', 'II.', 'III.', 'IV.','V.', 'VI.']

    for i in range(1, 100):
        title_head.append(str(i) + ' ')
        title_head.append(str(i) + '.')
        title_head.append(str(i) + '、')
        title_head.append(str(i) + ':')
        title_head.append(str(i) + '：')
        title_head.append(str(i) + ')')
        title_head.append(str(i) + '）')
        title_head.append('(' + str(i) + ')')
        title_head.append('（' + str(i) + '）')
    for char_ in ['A', 'B', 'C', 'D']:
        title_head.append(str(char_) + ' ')
        title_head.append(str(char_) + '.')
        title_head.append(str(char_) + '、')
        title_head.append(str(char_) + '：')
        title_head.append(str(char_) + ':')
        title_head.append('('+str(char_) + ')')
        title_head.append(str(char_) + ')')
        title_head.append('（' + str(char_) + '）')
        title_head.append(str(char_) + '）')

    for char_ in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        title_head.append(str(char_) + '、')
        title_head.append('('+str(char_) + ')')
        title_head.append('（' + str(char_) + '）')

    text = line['sentences'][0]['text']
    if len(text)>=2 and text[:2] in title_head:
        if text[:2] in ['1.', '2.', '3.', '4.','5.','6.','7.','8.', '9.'] and len(text)>=3 and text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return True
        return False
    if len(text)>=3 and text[:3] in title_head:
        return False
    return True

def is_topic(line):
    if len(line['sentences'])>1:
        return False
    topic_feature = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '（']
    text = line['sentences'][0]['text']
    if text!='' and text[0] in topic_feature:
        return True
    return False

def is_english_string(text):
    num_char = len(text)
    num_en_char = 0
    if num_char==0:
        return False
    for uu in text:
        if (ord(uu) <= 122 and ord(uu) >=97) or (ord(uu) <= 90 and ord(uu) >= 65) or uu in [' ', '.']:
            num_en_char+=1
    return (1.*num_en_char/num_char)>0.4

def is_same_para(cur_line, last_line, colum_bbox, is_english_paper=False):
    end_mark = ['.', "。", "？", "!", '\"', ']', ';', '?', ')', '）',':','：']
    colum_start, colum_end = colum_bbox
    if not is_need_group(cur_line):
        return False
    if len(last_line['sentences']) > 1:
        return False
    last_line_text = last_line['sentences'][0]['text']
    cur_line_text = cur_line['sentences'][0]['text']
    if len(last_line_text)==0 or len(cur_line_text)==0:
         return False
    bbox1 = last_line['bbox']
    bbox2 = cur_line['bbox']
    pix_char_width1 = (bbox1[2] - bbox1[0]) / len(last_line_text)
    pix_char_width2 = (bbox2[2] - bbox2[0]) / len(cur_line_text)
    if not is_need_group(last_line):
        if last_line_text[-1] in end_mark and colum_end-bbox1[2]>pix_char_width1:
            return False
        if cur_line_text[0] in ['A', 'B', 'C', 'D'] and colum_end-bbox2[0]<(colum_end-colum_start)*0.8:
            return False
    if (last_line_text[-1] in end_mark and cur_line_text[-1] in end_mark) or (last_line_text[0] in ['A', 'B', 'C', 'D'] and cur_line_text[0] in ['A', 'B', 'C', 'D']):
        return False
    if len(last_line_text) <= 8:
        return False
    if colum_end-bbox1[2]>(colum_end-colum_start)*0.5:
        return False

    if pix_char_width1>3*pix_char_width2 or pix_char_width2>3*pix_char_width1:
        return False
    t = pix_char_width2
    if is_english_paper and is_english_string(last_line_text) and is_english_string(cur_line_text):
        t = 2*t
    #  --------line1-------
    #----------line2---------
    if bbox1[0]>=bbox2[0]+t and bbox1[2]+t<=bbox2[2]:
        return False
    #--------line1-------。
    #    ----line2----
    if bbox1[0]+t<bbox2[0] and bbox1[2]>bbox2[2]+t and last_line_text[-1] in end_mark:
        return False
    #--------line1-------
    #   ----line2----------
    if bbox1[0]+t<bbox2[0] and bbox1[2]+t<=bbox2[2]:
        return False
    k1 = abs(bbox1[0]-bbox2[0])<1*t  and bbox1[2]+2*t >bbox2[2] and (last_line_text[-1] not in end_mark or cur_line_text[-1] not in end_mark)
    k2 = bbox1[0]-bbox2[0]>1*t and bbox1[0]-bbox2[0]<3.5*t and not (last_line_text[-1]  in end_mark and colum_end-bbox1[2]>pix_char_width1)
    #
    k3 = bbox1[0]-bbox2[0]<1*t and bbox1[0]-bbox2[0]>-3.5*t and not (last_line_text[-1]  in end_mark and colum_end-bbox1[2]>pix_char_width1)  and  \
         (not is_need_group(last_line)  or last_line_text[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '（'])
    return k1 or k2 or k3


def is_same_para_error_question(cur_line, last_line, colum_bbox, is_english_paper=False):
    # end_mark = ['.', "。", "？", "!", '\"', ']', ';', '?', ')', '）',':','：']
    end_mark = ['.', "。", "？", "!", '\"',  '?', '分)', '分）', '.\"', "。\"", "？\"", "!\"", '?\"']
    colum_start, colum_end = colum_bbox
    last_line_text = last_line['sentences'][0]['text']
    cur_line_text = cur_line['sentences'][0]['text']
    if not is_need_group(cur_line):# and end_with_mark(last_line):
        return False
    if len(last_line['sentences']) > 1:
        return False
    if len(last_line_text)==0 or len(cur_line_text)==0:
        return False
    bbox1 = last_line['bbox']
    bbox2 = cur_line['bbox']
    pix_char_width1 = (bbox1[2] - bbox1[0]) / len(last_line_text)
    pix_char_width2 = (bbox2[2] - bbox2[0]) / len(cur_line_text)
    if not is_need_group(last_line):
        if end_with_mark(last_line) and colum_end-bbox1[2]>pix_char_width1:
            return False
        if cur_line_text[0] in ['A', 'B', 'C', 'D'] and colum_end-bbox2[0]<(colum_end-colum_start)*0.8:
            return False
    if (end_with_mark(last_line) and end_with_mark(cur_line)) or (
            last_line_text[0] in ['A', 'B', 'C', 'D'] and cur_line_text[0] in ['A', 'B', 'C', 'D']):
        return False
    if len(last_line_text) <= 8:
        return False
    if colum_end-bbox1[2]>(colum_end-colum_start)*0.5:
        return False

    if '_' not in last_line_text and '_' not in cur_line_text and (
            pix_char_width1 > 3 * pix_char_width2 or pix_char_width2 > 3 * pix_char_width1):
        return False
    t = pix_char_width2
    if is_english_paper and is_english_string(last_line_text) and is_english_string(cur_line_text):
        t = 3 * t
    #  --------line1-------
    #----------line2---------
    if bbox1[0]>=bbox2[0]+t and bbox1[2]+t<=bbox2[2]:
        return False
    #--------line1-------。
    #    ----line2----
    if bbox1[0]+t<bbox2[0] and bbox1[2]>bbox2[2]+t and last_line_text[-1] in end_mark:
        return False
    #--------line1-------
    #   ----line2----------
    if bbox1[0]+t<bbox2[0] and bbox1[2]+t<=bbox2[2]:
        return False
    if abs(bbox1[0]-bbox2[0])>180:
        return False
    if bbox2[2] - bbox1[2]>2*t:
        return False
    return True


def is_para(cur_line, next_line, is_english_paper=False):
    if is_topic(cur_line):
        return  False
    if is_section(cur_line):
        return False
    if is_choice_option(cur_line):
        return False
    next_line_text = next_line['sentences'][0]['text']
    cur_line_text = cur_line['sentences'][0]['text']
    bbox1 = next_line['bbox']
    bbox2 = cur_line['bbox']
    pix_char_width1 = (bbox1[2] - bbox1[0]) / len(next_line_text)
    pix_char_width2 = (bbox2[2] - bbox2[0]) / len(cur_line_text)
    t = pix_char_width2
    if is_english_paper and is_english_string(next_line_text) and is_english_string(cur_line_text):
        t = 2 * t
    if '_' not in next_line_text and '_' not in cur_line_text and (
            pix_char_width1 > 3 * pix_char_width2 or pix_char_width2 > 3 * pix_char_width1):
        return False
    space = bbox2[0]-bbox1[0]
    if space < 3*t or space>5*t:
        return False
    return True


def is_same_para_error_question2(cur_line, last_line, is_para=False):
    bbox1 = last_line['bbox']
    bbox2 = cur_line['bbox']
    if len(last_line['sentences']) > 1:
        return False
    if abs(bbox1[0]-bbox2[0])>180:
        return False
    if is_choice_option(cur_line):
        return False
    if end_with_mark(last_line):
        if abs(bbox1[0]-bbox2[0])>10 or abs(bbox1[2]-bbox2[2])>10:
            return False
        if not is_para:
            return False
    if (bbox1[2]-bbox1[0])<(bbox2[2]-bbox2[0]-24) and bbox1[2]<bbox2[2]-24:
        return False
    return True

def end_with_mark(line):
    end_mark = ['.', "。", "？", "!", '\"',  '?', '分)', '分）', '.\"', "。\"", "？\"", "!\"", '?\"']
    text = line['sentences'][0]['text']
    if len(text)==0:
        return False
    if len(text) >= 1 and text[-1] in end_mark:
        return True
    if len(text) >= 2 and text[-2:] in end_mark:
        return True
    return False


def is_same_para2(line, last_line, is_english_paper=False):
    # if cur_paragraph['lines']==[]:
    #     return True
    # last_line = cur_paragraph['lines'][-1]
    bbox1 = line['bbox']
    bbox2 = last_line['bbox']
    if end_with_mark(last_line):
        if is_topic_para(last_line, line, is_english_paper):
            return True
        if abs(bbox1[0]-bbox2[0])>5 or abs(bbox1[2]-bbox2[2])>5:
            return True
    if abs(bbox1[0]-bbox2[0])>180:
        return True
    if is_choice_option(line):
        return True
    if bbox2[2]-bbox2[0]<bbox1[2]-bbox1[0]-24 and bbox2[2]<bbox1[2]-24:
        return True
    return False


def is_section(line):
    title_head = []
    for char_ in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        title_head.append(str(char_) + '、')
        title_head.append('('+str(char_) + ')')
        title_head.append('（' + str(char_) + '）')
    text = line['sentences'][0]['text']
    if len(text)>=2 and text[:2] in title_head:
        return True
    if len(text)>=3 and text[:3] in title_head:
        return True
    return False

def is_choice_option(line):
    choice_mark = ['A', 'B', 'C', 'D']
    text = line['sentences'][0]['text']
    if len(text)>=1 and text[0] in choice_mark:
        return True
    return False

def is_topic_para(line, next_line, is_english_paper=False):
    bbox1 = line['bbox']
    bbox2 = next_line['bbox']
    cur_text = line['sentences'][0]['text']
    next_text = next_line['sentences'][0]['text']
    if len(cur_text)==0 or len(next_text)==0:
        return False
    topic_feature = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '(', '（']
    if cur_text[0] in topic_feature:
        return False
    pix_char_width1 = (bbox1[2] - bbox1[0]) / len(cur_text)
    pix_char_width2 = (bbox2[2] - bbox2[0]) / len(next_text)
    if '_' not in cur_text and '_' not in next_text and (
        pix_char_width1 > 3 * pix_char_width2 or pix_char_width2 > 3 * pix_char_width1):
        return False
    space = bbox1[0] - bbox2[0]
    if not is_english_paper or (not is_english_string(cur_text) and not is_english_string(next_text)):
        if space<pix_char_width1*1.5 or space>pix_char_width1*3:
            return False
    else:
        if space<pix_char_width1*3 or space>pix_char_width1*5:
            return False
    return True

# def is_same_para(cur_line, last_line, colum_bbox, is_english_paper=False):
#     end_mark = ['.', "。", "？", "!", '\"', ']', ';', '?', ')', '）',':','：']
#     colum_start, colum_end = colum_bbox
#     if not is_need_group(cur_line):
#         return False
#     if len(last_line['sentences']) > 1:
#         return False
#     last_line_text = last_line['sentences'][0]['text']
#     cur_line_text = cur_line['sentences'][0]['text']
#     bbox1 = last_line['bbox']
#     bbox2 = cur_line['bbox']
#     pix_char_width1 = (bbox1[2] - bbox1[0]) / len(last_line_text)
#     pix_char_width2 = (bbox2[2] - bbox2[0]) / len(cur_line_text)
#     if not is_need_group(last_line):
#         if last_line_text[-1] in end_mark and colum_end-bbox1[2]>pix_char_width1:
#             return False
#         if cur_line_text[0] in ['A', 'B', 'C', 'D'] and colum_end-bbox2[0]<(colum_end-colum_start)*0.8:
#             return False
#     if (last_line_text[-1] in end_mark and cur_line_text[-1] in end_mark) or (last_line_text[0] in ['A', 'B', 'C', 'D'] and cur_line_text[0] in ['A', 'B', 'C', 'D']):
#         return False
#     if len(last_line_text) <= 8:
#         return False
#     if colum_end-bbox1[2]>(colum_end-colum_start)*0.5:
#         return False
#
#     if '_' not in last_line_text  and '_' not in cur_line_text and (pix_char_width1>3*pix_char_width2 or pix_char_width2>3*pix_char_width1):
#         return False
#     t = pix_char_width2
#     if is_english_paper and is_english_string(last_line_text) and is_english_string(cur_line_text):
#         t = 2*t
#     #  --------line1-------
#     #----------line2---------
#     if bbox1[0]>=bbox2[0]+t and bbox1[2]+t<=bbox2[2]:
#         return False
#     #--------line1-------。
#     #    ----line2----
#     if bbox1[0]+t<bbox2[0] and bbox1[2]>bbox2[2]+t and last_line_text[-1] in end_mark:
#         return False
#     #--------line1-------
#     #   ----line2----------
#     if bbox1[0]+t<bbox2[0] and bbox1[2]+t<=bbox2[2]:
#         return False
#     k1 = abs(bbox1[0]-bbox2[0])<1*t  and bbox1[2]+2*t >bbox2[2] and (last_line_text[-1] not in end_mark or cur_line_text[-1] not in end_mark)
#     k2 = bbox1[0]-bbox2[0]>1*t and bbox1[0]-bbox2[0]<3.5*t and not (last_line_text[-1]  in end_mark and colum_end-bbox1[2]>pix_char_width1)
#     k3 = bbox1[0]-bbox2[0]<1*t and bbox1[0]-bbox2[0]>-3.5*t and not (last_line_text[-1]  in end_mark and colum_end-bbox1[2]>pix_char_width1)  and last_line_text[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '（']
#     return k1 or k2 or k3

def is_math_caculate_line(line):
    math_mark = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '×', '＞',  '＜',  '＝', '.', '÷']
    for sentence in line['sentences']:
        text = sentence['text']
        num_char = len(text)
        num_math_char = 0
        for char_ in text:
            if char_ in math_mark:
                num_math_char += 1
        if num_math_char<num_char*0.6:
            return False
    return True

def add_font_size(omml_string, font_size):
    # run_font_xml = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:i/><w:sz w:val="%s"/></w:rPr>'%(2*(font_size))
    # math_font_xml = '<m:ctrlPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsiaTheme="SimSun"/><w:i/><w:sz w:val="%s"/></w:rPr></m:ctrlPr>'%(2*(font_size))
    run_font_xml = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="%s"/></w:rPr>' % (2 * (font_size))
    math_font_xml = '<m:ctrlPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsiaTheme="SimSun"/><w:sz w:val="%s"/></w:rPr></m:ctrlPr>' % (2 * (font_size))
    omml_string = omml_string.replace('<m:r>', '<m:r>'+run_font_xml)
    omml_string = omml_string.replace('omml<m:oMath ', 'omml<m:oMath '+'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ')
    if 'm:rad' in omml_string:
        omml_string = omml_string.replace('</m:radPr>', math_font_xml+'</m:radPr>')
    if '<m:fPr>' in omml_string:
        omml_string= omml_string.replace('<m:fPr>', '<m:fPr>'+ math_font_xml)
    if '<m:r>' in omml_string:
        omml_string= omml_string.replace('<m:r>', '<m:r><m:rPr><m:sty m:val="p"/></m:rPr>')
    omml_string = omml_string.replace('（）', '（         ）')
    return omml_string

def get_line_bbox(line):
    x1 = float('inf')
    y1 = float('inf')
    x2 = 0
    y2 = 0
    for sentence in line['sentences']:
        bbox = sentence['bbox']
        x1 = min(x1, bbox[0])
        y1 = min(y1, bbox[1])
        x2 = max(x2, bbox[2])
        y2 = max(y2, bbox[3])
    return [x1, y1, x2, y2]

def write_json(data, file_path):
    import json
    jdata = json.dumps(data['doc_info'], ensure_ascii=False)
    f = open(file_path, 'w')
    f.write(jdata)

def pt2cm(size):
    size_cm = size*0.03528
    return size_cm

def cm2pt(size):
    size_pt = size/0.03528
    return size_pt

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def is_all_number(strs):
    for _char in strs:
        if _char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '/']:
            return False
    return True

def is_chinese(strs):
    for _char in strs:
        if  '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def my_round(num):
    a = num - num//1
    if a>0.75:
        return round(num)
    else:
        return int(num)


def get_mid_num(nums):
    nums.sort()
    if len(nums)%2==0:
        mid = int(len(nums)/2) - 1
    else:
        mid = len(nums)//2
    return nums[mid]

def get_max_count_num(nums):
    count = {}
    for num in nums:
        if num not in count.keys():
            count[num] = 1
        else:
            count[num] += 1
    max_count = 0
    max_num = nums[0]
    for num,count in count.item():
        if count > max_count:
            max_count = count
            max_num = num
    return max_num

def is_pinyin(string):
    # 使用正则表达式匹配拼音字母（含声调字符）
    pattern = '^[abcdefghijklmnopqrstuwxyzBCDFGHJKLMNPQRSTWXYZāáǎàēéěèīíǐìōóǒòūúǔùüǖǘǚǜ ]+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False

def char_parse(img, text):
    detail = ""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    words = []
    i = 0
    threshold = 0
    while i < img.shape[1]:
        if sum(img[:, i]) > 0:
            x1 = i
            i += 1
            x2 = 0
            while i < img.shape[1]:
                if sum(img[:, i]) == 0:
                    x2 = i - 1
                    break
                i += 1
            if i == img.shape[1]:
                x2 = img.shape[1] - 1
            if x2 > x1:
                words.append([x1, x2])
                detail += "%d,%d " % (x1, x2)
                img[:, x1] = 255
                img[:, x2] = 255
        i += 1
    detail = detail[:-1] + ";"
    for x1, x2 in words:
        i = 0
        y1 = 0
        while i < img.shape[0]:
            if sum(img[i, x1 + 1:x2]) > 0:
                y1 = i
                break
            i += 1
        img[y1, x1:x2] = 255
        i = img.shape[0] - 1
        y2 = i
        while i >= 0:
            if sum(img[i, x1 + 1:x2]) > 0:
                y2 = i
                break
            i -= 1
        img[y2, x1:x2] = 255
        detail += "%d,%d " % (y1, y2)
    return detail


def sqrt(s):
    return '\sqrt{%s}'%s

def power(s):
    a, b = s.split('^')[:2]
    return '%s^{%s}'%(a,b)

def frac(a,b):
    return '\dfrac{%s}{%s}'%(a,b)

def get_power(string):
    string_split = math_formula_parse1(string)
    indexs = [0]*len(string_split)
    for i,uu in enumerate(string_split):
        if uu == '':
            continue
        if "^" in uu:
            indexs[i] = 1
    new_string = ''
    i = 0
    while i <len(string_split):
        uu = string_split[i]
        if i<len(string_split)-1 and indexs[i+1]==1:
            new_string += power(uu+string_split[i+1])
            i += 2
        else:
            new_string += uu
            i += 1
    return new_string



def get_sqrt(text):
    res = ''
    i = 0
    while i<len(text):
        uu = text[i]
        if uu == '¤':
            j = i + 1
            if j > len(text) - 1:
                break
            if text[j] != '（':
                res += sqrt(text[j])
                i = j + 1
                continue
            else:
                tmp = ''
                while j < len(text):
                    if text[j] != '）':
                        tmp += text[j]
                    else:
                        tmp += text[j]
                        break
                    j += 1
                res += sqrt(tmp)
                i = j + 1
                continue
        else:
            res += uu
        i += 1
    return res



def math_formula_parse3(text):
    idx = 0
    while idx<len(text) and text[idx]!='¤':
        idx += 1
    if idx==len(text)-1:
        return text
    part1 = text[:idx]
    part2 = text[idx:]
    if '^' in part1:
        part1 = get_power(part1)
    if '^' in part2:
        part2 = get_power(part2)
    part2 =  get_sqrt(part2)
    sqrt_string = part1+part2
    return ['latex'+sqrt_string]

#parse fraction
def math_formula_parse2(text):
    idx = 0
    while idx<len(text) and text[idx]!='/':
        idx += 1
    if idx==len(text)-1:
        return text
    numerator = text[:idx]
    if numerator!='' and numerator[0]=='（' and numerator[-1]=='）':
        numerator = numerator[1:-1]
    if '^' in numerator:
        numerator = get_power(numerator)
    if '¤' in numerator:
        numerator =  get_sqrt(numerator)
    denominator = text[idx + 1:]
    if denominator!='' and denominator[0]=='（' and denominator[-1]=='）':
        denominator = denominator[1:-1]
    if '^' in denominator:
        denominator = get_power(denominator)
    if '¤' in denominator:
        denominator = get_sqrt(denominator)
    if denominator.replace('\u3000','')=='':
        denominator = '（' + chr(12288) * 3 + '）'
    if numerator.replace('\u3000','')=='':
        numerator = '（' + chr(12288) * 3 + '）'
    fraction = frac(numerator, denominator)
    return ['latex'+fraction]

def sup_sub_parse(text):
    if text.count('_')!=1 or text.count('^')!=1:
        return text
    index1 = text.index('_')
    index2 = text.index('^')
    if index1>index2:
        return text
    if index1==0 or index2==0:
        return text
    if index1==len(text)-1 or index2==len(text)-1:
        return text
    fsub = False
    fsup = False
    element = ''
    sub = ''
    sup = ''
    for i in range(len(text)):
        if not fsub and not fsup and text[i]!='_':
            element += text[i]
        elif not fsub and text[i]=='_':
            fsub = True
        elif fsub and text[i]!='^':
            sub += text[i]
        elif fsub and text[i]=='^':
            fsup = True
            fsub = False
        elif fsup:
            sup += text[i]
    if element!='' and sub!='' and sup!='':
        return ['latex'+element+'_{'+sub+'}^{'+sup+'}']
    else:
        return text
#parse up down
def math_formula_parse1(text):
    if '_' in text and '^' in text and (text[-1]=='-' or text[-1]=='+'):
        ret = sup_sub_parse(text)
        if ret!=text:
            return ret

    split_char = ['^', '_']
    text_splits = []
    flag = False
    i = 0
    text_split = ''
    while i<len(text):
        char_ = text[i]
        if char_ in split_char:
            if char_ == '_' :
                if i<len(text)-1 and text[i+1]!='_' and i>0 and text[i-1]!='_':
                    char_ = '^^'
                else:
                    text_split += char_
                    i += 1
                    continue
            j = i+1
            if j>len(text)-1:
                break
            text_splits.append(text_split)
            if text[j]!='（':
                if char_ == '_':
                    char_ = '^^'
                tmp = ''
                while j<len(text):
                    if text[j] != '`' and text[j] in ['0','1','2','3','4','5','6','7','8','9'] :
                        tmp+=text[j]
                    elif text[j] != '`' and text[j] in ['+', '-'] and char_=='^':
                        tmp+=text[j]
                    elif text[j] != '`' and text[j] in ['a', 'A'] and char_=='^':
                        tmp+=text[j]
                    else:
                        break
                    j += 1
                if tmp=='':
                    char_=''
                    tmp = text[j]
                text_splits.append(char_+tmp)
                i = j+1
                text_split = ''
                continue
            else:
                tmp = ''
                while j<len(text):
                    if text[j]!='）':
                        tmp+=text[j]
                    else:
                        tmp += text[j]
                        break
                    j += 1
                text_splits.append(char_ + tmp)
                text_split = ''
                i = j+1
                continue
        else:
            text_split += char_
        i += 1
    if text_split!='':
        text_splits.append(text_split)
    return text_splits

def sqrt_parse(text):
    #text = text.replace('（', '(').replace('）', ')')
    formula_text = ''
    text_splits  = []
    i = 0
    while i<len(text):
        char_ = text[i]
        if char_ == '¤':
            if formula_text != '':
                text_splits.append(formula_text)
            j = i + 1
            if j > len(text) - 1:
                break
            if text[j] != '（':
                text_splits.append(char_ + text[j])
                i = j + 1
                formula_text = ''
                continue
            else:
                tmp = ''
                while j < len(text):
                    if text[j] != '）':
                        tmp += text[j]
                    else:
                        tmp += text[j]
                        break
                    j += 1
                text_splits.append(char_ + tmp)
                formula_text = ''
                i = j
        else:
            formula_text += char_
        i = i + 1
    if formula_text!='':
        text_splits.append(formula_text)
    new_text_splits = []
    for text_split in text_splits:
        if '¤' not in text_split:
            new_text_splits.append(text_split)
        else:
            new_text_splits.append('latex'+get_sqrt(text_split))
    return new_text_splits

def is_alphabet(char_):
    if ord(char_)>=65 and ord(char_)<=90:
        return True
    elif ord(char_)>=97 and ord(char_)<=122:
        return True
    return False

def is_chemical_str(text):
    count1 = 0
    count2 = 0
    for i, char_ in enumerate(text):
        if is_alphabet(char_):
            if ord(char_)>=65 and ord(char_)<=90:
                count1 += 1
            else:
                count2 += 1
    if count2 <= count1 and count1+count2>=4:
        return True
    return False

def math_formula_parse(text):
    text_splits  = []
    if '^' not in text and '_' not in text and '/' not in text and '¤' not in text:
        return [text.replace('`', '')]
    if '`' not in text and '¤' in text:
        return sqrt_parse(text)
    if '`）_' in text:
        text = re.sub('（([\dA-Za-z`_]+)`）', r'(\1`)', text)
    formula_text = ''
    flag = False
    orig_text_splits = []
    for i, char_ in enumerate(text):
        if char_ == '`' and not flag:
            if formula_text!='':
                if '¤' not in formula_text and '^' not in formula_text:
                    text_splits.append(formula_text)
                elif '^' in formula_text:
                    text_splits.extend(math_formula_parse1(formula_text))
                else:
                    text_splits.extend(sqrt_parse(formula_text))
                orig_text_splits.append(formula_text)
            formula_text = ''
            flag = True
        elif char_=='`' and flag:
            if '/' in formula_text:
                texts = math_formula_parse2(formula_text)
            elif '¤' in formula_text:
                texts = math_formula_parse3(formula_text)
            else:
                texts = math_formula_parse1(formula_text)
            text_splits.extend(texts)
            flag = False
            formula_text = ''
        else:
            formula_text += char_
    if '¤' not in formula_text:
        text_splits.append(formula_text)
    else:
        text_splits.extend(sqrt_parse(formula_text))
    new_text_splits = []
    text = ''
    for uu in text_splits:
        if uu =='':
            continue
        if uu[0]!='^' and 'latex' not in uu:
            text += uu
        else:
            new_text_splits.append(text)
            new_text_splits.append(uu)
            text = ''
    if text!='':
        new_text_splits.append(text)
    return new_text_splits

def cast_invalid_cell1(cell_index, invalid_row, invalid_col):
    new_cell_index = []
    for i in range(len(cell_index)):
        if i in invalid_row:
            continue
        new_cell_index.append(cell_index[i])
    cell_index = new_cell_index.copy()
    new_cell_index = []
    for i in range(len(cell_index)):
        tmp = []
        for j in range(len(cell_index[0])):
            if j in invalid_col:
                continue
            tmp.append(cell_index[i][j])
        new_cell_index.append(tmp)
    cell_index = new_cell_index
    return cell_index

def cast_invalid_cell(row_num, col_num, cell_index):
    invalid_row = []
    for i in range(row_num):
        invalid = True
        for j in range(col_num):
            if cell_index[i][j] != -1:
                invalid = False
                break
        if invalid:
            invalid_row.append(i)
    invalid_col = []
    for j in range(col_num):
        invalid = True
        for i in range(row_num):
            if cell_index[i][j] != -1:
                invalid = False
                break
        if invalid:
            invalid_col.append(j)
    cell_index = cast_invalid_cell1(cell_index, invalid_row, invalid_col)
    return cell_index

def cast_merged_row_col(cell_index, cells):
    cell_index_array = np.array(cell_index)
    row_num, col_num = cell_index_array.shape
    col_merge_matrixs1 = np.zeros_like(cell_index_array)
    col_merge_matrixs2 = np.zeros_like(cell_index_array)
    for i in range(row_num):
        for j in range(col_num):
            cur_index = cell_index_array[i][j]
            if j>0:
                last_index = cell_index_array[i][j-1]
                if last_index==cur_index:
                    col_merge_matrixs1[i][j] = 1
            if j<col_num-1:
                next_index =  cell_index_array[i][j + 1]
                if next_index == cur_index:
                    col_merge_matrixs2[i][j] = 1
    col_merge_sum = np.sum(col_merge_matrixs1, axis=0)
    merged_col_index = []
    for id, value in enumerate(col_merge_sum):
        if value == row_num:
            merged_col_index.append(id)

    row_merge_matrixs = np.zeros_like(cell_index_array)
    for i in range(row_num):
        for j in range(col_num):
            if (i > 0 and cell_index_array[i][j] == cell_index_array[i-1][j]) or \
                    (i < row_num - 1 and cell_index_array[i][j] == cell_index_array[i + 1][j]):
                row_merge_matrixs[i][j] = 1
    row_merge_sum = np.sum(row_merge_matrixs, axis=1)

    merged_row_index = []
    for id, value in enumerate(row_merge_sum):
        if value == col_num:
            merged_row_index.append(id)
    return merged_row_index, merged_col_index








if __name__ == "__main__":
    a = regular_split("()1skate")
    print(a)