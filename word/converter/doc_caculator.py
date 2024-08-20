# -*-coding:utf-8-*-
import os, sys
import cv2
import json
from fontTools.ttLib import TTFont
from PIL import ImageFont
from doc_utils import *

class Types:
    def __init__(self):
        self.text = 1
        self.answer = 2
        self.pic = 3
        self.para = 4
        self.student = 10
        self.blank = 11
label_types = Types()
#'':0.11
class doc_caculator:
    def __init__(self):
        self.size_map = {'a':0.156, 'b':0.175, 'c':0.156, 'd':0.175, 'e':0.155, 'f':0.113, 'g':0.175, 'h':0.175, 'i':0.101, 'j':0.0975, 'k':0.175, 'l':0.0974, 'm':0.273, 'n':0.175, 'o':0.175, \
        'p':0.175, 'q':0.175, 'r':0.118, 's':0.136, 't':0.0975, 'u':0.175, 'v':0.175, 'w':0.255, 'x':0.175, 'y':0.175, 'z':0.156, 'A':0.255, 'B':0.235,'C':0.247, 'D':0.255, 'E':0.215, 'F':0.196,\
        'G':0.255, 'H':0.255, 'I':0.118, 'J':0.137, 'K':0.255, 'L':0.215, 'M':0.3125, 'N':0.255, 'O':0.255, 'P':0.197, 'Q':0.255, 'R':0.234, 'S':0.196, 'T':0.216,  'U':0.255, 'V':0.255, 'X':0.255, 'Y':0.255,\
        'Z':0.216, 'W':0.333, ' ':0.088, ',':0.086, '1':0.1607, '2':0.175, '3':0.175, '4':0.175, '5':0.175, '6':0.175, '7':0.175, '8':0.175, '9':0.175, '0':0.175, '/':0.098, '.':0.086,\
        '“':0.352, '”':0.352, '∠':0.352, '°':0.352, '·':0.352, '＝':0.352, 'π':0.352, '×':0.352, '÷':0.352
        }
        self.my_type = {1:'text', 2:'answer', 3:'pic', 4:'paragraph', 10:'textbox', 5:'underline'}

    def process(self, img, meta_data, prefix, work_dir=''):
        self.work_dir = work_dir
        if os.path.exists(os.path.join(work_dir, 'tmp')) is False:
            os.makedirs(os.path.join(work_dir, 'tmp'))
        self.prefix = prefix
        self.set_font()
        height, width, _ = img.shape
        image_info = {}
        image_info['img'] = img
        image_info['ori_height'] = height
        image_info['ori_width'] = width
        meta_data['image_info'] = image_info
        meta_data = self.set_page_global_size(meta_data)
        meta_data = self.cast_invalid_regions(meta_data)
        meta_data = self.split_colums(meta_data)
        meta_data = self.merge_lines(meta_data)
        meta_data = self.calculate_font_size(meta_data)
        meta_data = self.group_paragraphs(meta_data)
        meta_data = self.get_line_left_indent(meta_data)
        meta_data = self.get_line_text(meta_data)
        meta_data = self.calculate_font_space(meta_data)
        #meta_data = self.calculate_paragraph_attributes(meta_data)
        meta_data = self.calculate_paragraph_attributes2(meta_data)
        meta_data = self.generate_formula(meta_data)
        meta_data = self.generate_pictures(meta_data)
        meta_data = self.generate_text_boxs(meta_data)
        meta_data = self.generate_tables(meta_data)
        meta_data = self.set_index(meta_data)
        return meta_data


    def set_font(self):
        font = TTFont(os.path.join(self.work_dir, 'font', 'times.ttf'))
        self.en_uniMap = font['cmap'].tables[0].ttFont.getBestCmap().keys()
        font = TTFont(os.path.join(self.work_dir, 'font', 'SimSun-01.ttf'))
        self.cn_uniMap = font['cmap'].tables[0].ttFont.getBestCmap().keys()

    def set_page_global_size(self, meta_data):
        img_h = meta_data['image_info']['ori_height']
        img_w = meta_data['image_info']['ori_width']
        is_horizontal_paper = False
        page_height = 29.7
        page_width = 21
        page_top_bottom_margin = 2.54
        page_left_right_margin = 1.91
        valid_page_width = page_width - page_left_right_margin * 2
        valid_page_height = page_height - page_top_bottom_margin * 2 - 0.35  # 0.35为众向余量
        # 判断是否为横向文本
        if img_w > img_h:
            is_horizontal_paper = True
            page_height = 21
            page_width = 29.7
            page_top_bottom_margin = 1.91
            page_left_right_margin = 2.54
            valid_page_width = page_width - page_left_right_margin * 2
            valid_page_height = page_height - page_top_bottom_margin * 2 - 0.35
        img_left_margin = img_w
        img_right_margin = 0
        img_top_margin = img_h
        img_bottom_margin = 0
        # 计算有效内容占据的区域和图片边缘之间的边距
        for region in meta_data['regions']:
            bbox = region['region']
            rotation = region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            img_left_margin = min(img_left_margin, x1)
            img_right_margin = max(img_right_margin, x2)
            img_top_margin = min(img_top_margin, y1)
            img_bottom_margin = max(img_bottom_margin, y2)

        for region in meta_data['pics']:
            bbox = region['region']
            rotation = region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            img_left_margin = min(img_left_margin, x1)
            img_right_margin = max(img_right_margin, x2)
            img_top_margin = min(img_top_margin, y1)
            img_bottom_margin = max(img_bottom_margin, y2)

        for table in meta_data['tables']:
            bbox = table['rect']
            x1, y1, x2, y2 = bbox
            img_left_margin = min(img_left_margin, x1)
            img_right_margin = max(img_right_margin, x2)
            img_top_margin = min(img_top_margin, y1)
            img_bottom_margin = max(img_bottom_margin, y2)

        img_right_margin = img_w - img_right_margin
        img_bottom_margin = img_h - img_bottom_margin
        img_right_margin = min(img_right_margin, img_left_margin)
        img_bottom_margin = min(img_bottom_margin, img_top_margin)
        valid_img_w = img_w - img_left_margin - img_right_margin
        # 对于竖排文本
        if not is_horizontal_paper:
            page_hw_ratio = page_height / page_width
            img_hw_ratio = img_h / img_w
            # 若高宽比小于1.412
            if img_hw_ratio < page_hw_ratio:
                pad_h = int(img_w * (page_hw_ratio - img_hw_ratio))
                img_h += pad_h
            else:
                # 若高宽比大于1.412
                if img_hw_ratio - page_hw_ratio < 0.1:
                    pad_w = (img_h / page_hw_ratio - img_w)
                    img_w += pad_w
                else:
                    page_width = max(2*page_left_right_margin+0.1, page_height / (img_hw_ratio + 0.01))
                    valid_page_width = page_width - page_left_right_margin * 2
        else:
            page_wh_ratio = page_width / page_height
            img_wh_ratio = img_w / img_h
            if img_wh_ratio < page_wh_ratio:
                pad_w = int(page_wh_ratio * img_h - img_w)
                img_w += pad_w
            else:
                pad_h = (img_w / page_wh_ratio - img_h)
                img_h += pad_h
        valid_img_h = img_h - img_top_margin - img_bottom_margin
        valid_img_w = img_w - img_left_margin - img_right_margin
        image_info = meta_data['image_info']
        image_info['img_top_margin'] = img_top_margin
        image_info['img_left_margin'] = img_left_margin
        image_info['img_bottom_margin'] = img_bottom_margin
        image_info['img_right_margin'] = img_right_margin
        image_info['valid_img_h'] = valid_img_h
        image_info['valid_img_w'] = valid_img_w

        # real_page_top_bottom_margin = img_top_margin/valid_img_h * valid_page_height;
        # if real_page_top_bottom_margin>page_top_bottom_margin:
        #     page_top_bottom_margin = real_page_top_bottom_margin
        #     valid_page_height = page_height - page_top_bottom_margin * 2
        page_info = {}
        page_info['valid_page_width'] = valid_page_width
        page_info['valid_page_height'] = valid_page_height
        page_info['page_top_bottom_margin'] = page_top_bottom_margin
        page_info['page_left_right_margin'] = page_left_right_margin
        page_info['page_width'] = page_width
        page_info['page_height'] = page_height

        meta_data['doc_info'] = {}
        meta_data['doc_info']['doc_page_info'] = page_info
        meta_data['image_info'] = image_info
        self.valid_img_h = valid_img_h
        self.valid_img_w = valid_img_w
        self.valid_page_width = valid_page_width
        self.valid_page_height = valid_page_height
        return meta_data

    def cast_invalid_regions(self, meta_data):
        #First : Read all regions detected by ocr model
        regions = meta_data['regions']
        pics = meta_data['pics']
        tables = meta_data['tables']

        doc_texts = []
        doc_pics = []
        doc_textboxs = []
        doc_tables = []
        img_left_margin = meta_data['image_info']['img_left_margin']
        img_top_margin = meta_data['image_info']['img_top_margin']
        n_chars = 0
        n_en_chars = 0
        regions = cast_texts(regions)
        for region in regions:
            bbox = region['region']
            rotation = region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            rx1, ry1, rx2, ry2 = move_bbox([x1, y1, x2, y2], img_left_margin, img_top_margin)
            cls = region['cls']
            if cls == 10 and (ry2-ry1)>(rx2-rx1):
                continue
            if 'result' not in region.keys():
                result = ''
            else:
                result = region['result'][0]
            if result == "@":
                continue
            n_chars += len(result)
            for uu in result:
                if (uu >= 'a' and uu <= 'z') or (uu >= 'A' and uu <= 'Z'):
                    n_en_chars += 1
            result = result.replace("\\@", "@")
            result = replace_text(result)
            cur_sentence = {}
            cur_sentence['bbox'] = [rx1, ry1, rx2, ry2]
            cur_sentence['type'] = self.my_type[cls]
            cur_sentence['text'] = result
            if cls==label_types.text and (ry2-ry1)>2*(rx2-rx1):
                continue
            if cls == label_types.student:
                doc_textboxs.append(cur_sentence)
            else:
                doc_texts.append(cur_sentence)
        self.is_english_doc = False
        if n_chars!=0 and  n_en_chars * 1. / n_chars > 0.4:
            self.is_english_doc = True


        for pic_region in pics:
            bbox = pic_region['region']
            rotation = pic_region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            rx1, ry1, rx2, ry2 = move_bbox([x1, y1, x2, y2], img_left_margin, img_top_margin)
            result = ''
            cls = pic_region['cls']
            cur_sentence = {}
            cur_sentence['bbox'] = [rx1, ry1, rx2, ry2]
            cur_sentence['type'] = self.my_type[cls]
            cur_sentence['text'] = result
            cur_sentence['url'] = pic_region['url']
            doc_pics.append(cur_sentence)

        for table_region in tables:
            bbox = table_region['rect']
            rx1, ry1, rx2, ry2 = move_bbox(bbox, img_left_margin, img_top_margin)
            table_region['bbox'] = [rx1, ry1, rx2, ry2]
            table_region['type'] = 'table'
            doc_tables.append(table_region)

        #Second : Cast invalid regions
        doc_pics = cast_region(doc_pics, doc_tables)
        doc_tables = cast_region(doc_tables, doc_pics)

        doc_texts = cast_region(doc_texts, doc_pics)
        doc_pics, f = crop_pics(doc_pics, doc_texts)
        doc_textboxs = cast_region(doc_textboxs, doc_pics+doc_texts)

        doc_info = meta_data['doc_info']
        doc_info['doc_texts'] = doc_texts
        doc_info['doc_textboxs'] = doc_textboxs
        doc_info['doc_pics'] = doc_pics
        doc_info['doc_tables'] = doc_tables
        meta_data['doc_info'] = doc_info
        return meta_data

    def split_colums(self, meta_data):
        doc_texts = meta_data['doc_info']['doc_texts']
        doc_textboxs = meta_data['doc_info']['doc_textboxs']
        doc_texts = sorted(doc_texts, key=lambda x:x['bbox'][3])
        doc_paragraphs = []
        for doc_text in doc_texts:
            if doc_text['type']=='paragraph':
                doc_text['lines'] = []
                doc_paragraphs.append(doc_text)

        new_doc_texts = []
        #1.merge text of para
        for doc_text in doc_texts:
            if doc_text['type'] != 'text':
                continue
            bbox = doc_text['bbox']
            #这里若当前word的图片插入以浮动式图片为嵌入形式，则图片和文本分开操作
            # if self.is_english_paper:
            #     doc_text['text'] = doc_text['text'].replace("（", "(").replace("）", ")")
            x1, y1, x2, y2 = bbox
            in_para = False
            for idx in range(len(doc_paragraphs)):
                para_x1, para_y1, para_x2, para_y2 = doc_paragraphs[idx]['bbox']
                inter_x1 = max(para_x1, x1)
                inter_y1 = max(para_y1, y1)
                inter_x2 = min(para_x2, x2)
                inter_y2 = min(para_y2, y2)
                inter_area = max(0, (inter_y2-inter_y1)) * max(0, (inter_x2-inter_x1))
                cur_area = (x2-x1)*(y2-y1)
                ratio = inter_area * 1. / cur_area
                if ratio >0.75:
                    doc_paragraphs[idx]['lines'].append(doc_text)
                    in_para = True
                    break
            if not in_para:
                new_doc_texts.append(doc_text)
        doc_texts = new_doc_texts
        doc_paragraphs = sorted(doc_paragraphs, key=lambda x:x['bbox'][0])
        #2.split colums
        doc_colums = []
        pages_x = []
        colum_ranges = []
        x_thresh = 0
        # 根据是否存在并列关系的段落来判断是否存在分栏
        for idx, doc_para in enumerate(doc_paragraphs):
            bbox = doc_para['bbox']
            cur_x1 = bbox[0]
            column_id = len(pages_x) - 1
            # 新建一栏的规则为
            if column_id < 0 or pages_x[column_id] + x_thresh < cur_x1:
                x_thresh = (bbox[2] - bbox[0]) * 3. / 4
                doc_colums.append([])
                pages_x.append(cur_x1)
                colum_ranges.append([bbox[0], bbox[2]])
                column_id = len(pages_x) - 1
            else:
                x_thresh = max(x_thresh, (bbox[2] - pages_x[-1]))
            doc_para['colum_id'] = column_id
            doc_colums[column_id].append(doc_para)
            colum_ranges[column_id][1] = max(colum_ranges[column_id][1], bbox[2])
        # 如果当前内容无段落
        no_para = False
        if len(doc_paragraphs) == 0:
            no_para = True
            doc_colums = [[]]
            colum_ranges = [[0, meta_data['image_info']['valid_img_w']]]
        # 将段落之外的内容和图片划分一下栏
        txbx_threshold = 0.8
        for idx, doc_text in enumerate(doc_texts):
            bbox = doc_text['bbox']
            if no_para:
                doc_text['colum_id'] = 0
                doc_colums[0].append(doc_text)
                continue
            index = 0
            max_iou = 0
            min_iou = 1.
            ious = []
            for idx, crange in enumerate(colum_ranges):
                cx1, cx2 = crange
                lx1, lx2 = bbox[0], bbox[2]
                max_left = max(cx1, lx1)
                min_right = min(lx2, cx2)
                iou_1d = (min_right - max_left) / (lx2 - lx1)
                if iou_1d > max_iou:
                    max_iou = iou_1d
                    index = idx
                ious.append(iou_1d)
            cx1, cx2 = colum_ranges[index]
            lx1, lx2 = bbox[0], bbox[2]
            #if len(ious)<=1 or min(ious)==0. and max_iou>txbx_threshold:
            # 若被判断为处于原文中间的行，则调整其坐标移动至归属列的中间
            # if lx2>cx2 and min(lx2, cx2)-max(cx1, lx1)<=0.6*width:
            if (max_iou <= txbx_threshold or (lx2 - lx1) >= 1.2 * (cx2 - cx1) or min(ious)>0.05) and len(doc_colums) >= 2:
                doc_text['type'] = 'textbox'
                doc_textboxs.append(doc_text)
                continue
            doc_text['colum_id'] = index
            doc_colums[index].append(doc_text)
        if doc_colums==[[]]:
            doc_colums = []

        meta_data['doc_info']['doc_sections'] = []
        doc_section = {}
        doc_section['doc_colums'] = doc_colums
        meta_data['doc_info']['doc_sections'].append(doc_section)
        meta_data['doc_info']['doc_textboxs'] = doc_textboxs
        del meta_data['doc_info']['doc_texts']
        return meta_data

    def merge_lines(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        doc_text_boxs = meta_data['doc_info']['doc_textboxs']
        for colum_id, doc_colum in enumerate(doc_colums):
            doc_colum = self.cast_part_para(doc_colum)
            new_doc_colum = []
            doc_ind_lines = []
            #merge para lines
            for doc_content in doc_colum:
                if doc_content['type'] == 'paragraph':
                    doc_lines, text_boxs = self.merge_sentences(doc_content['lines'])
                    doc_content['lines'] = doc_lines
                    new_doc_colum.append(doc_content)
                    doc_text_boxs.extend(text_boxs)
                else:
                    doc_ind_lines.append(doc_content)
            #merge independent lines
            doc_ind_lines, text_boxs = self.merge_sentences(doc_ind_lines)
            doc_text_boxs.extend(text_boxs)
            new_doc_ind_lines = []
            #cast textbox
            for doc_ind_line in doc_ind_lines:
                y1, y2 = doc_ind_line['bbox'][1],doc_ind_line['bbox'][3]
                x1, x2 = doc_ind_line['bbox'][0],doc_ind_line['bbox'][2]
                flag = False
                for doc_content in new_doc_colum:
                    if doc_content['type'] != 'paragraph':
                        continue
                    py1, py2 = doc_content['bbox'][1], doc_content['bbox'][3]
                    px1, px2 = doc_content['bbox'][0], doc_content['bbox'][2]
                    inter_y1 = max(y1, py1)
                    inter_y2 = min(y2, py2)
                    inter_x1 = max(x1, px1)
                    inter_x2 = min(x2, px2)
                    if inter_y2>=inter_y1 and inter_x2<=inter_x1:
                        flag = True
                        break
                if flag:
                    doc_ind_line['type'] = 'textbox'
                    doc_text_boxs.extend(doc_ind_line['sentences'])
                else:
                    new_doc_ind_lines.append(doc_ind_line)
            new_doc_colum.extend(new_doc_ind_lines)
            new_doc_colum = sorted(new_doc_colum, key=lambda x: x['bbox'][1])
            doc_colums[colum_id] = new_doc_colum
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        meta_data['doc_info']['doc_textboxs'] = doc_text_boxs
        return meta_data

    def merge_sentences(self, sentences):
        if len(sentences)==0:
            return [], []
        sentences = sorted(sentences, key=lambda x:x['bbox'][3])
        doc_lines = []
        doc_line = {}
        doc_line['type'] = 'line'
        cur_line = []
        last_h = (sentences[0]['bbox'][3] + sentences[0]['bbox'][1])/2
        #遍历所有内容
        for sentence in sentences:
            cur_h = (sentence['bbox'][3] + sentence['bbox'][1])/2
            cur_word_h = (sentence['bbox'][3] - sentence['bbox'][1])
            if cur_line==[]:
                cur_line.append(sentence)
                last_h = cur_h
                continue
            #若当前内容的y坐标和上一内容的y坐标差小于上一内容的高度值的一半，则将当前内容合并至上一内容
            if cur_h-last_h<cur_word_h*0.5:
                cur_line.append(sentence)
            else:
                cur_line = sorted(cur_line, key=lambda x:x['bbox'][0])
                doc_line['sentences'] = cur_line
                doc_line['bbox'] = get_line_bbox(doc_line)
                doc_lines.append(doc_line)
                doc_line = {'type':'line'}
                cur_line = [sentence]
                last_h = cur_h

        if len(cur_line)!=0:
            cur_line = sorted(cur_line, key=lambda x:x['bbox'][0])
            doc_line['sentences'] = cur_line
            doc_line['bbox'] = get_line_bbox(doc_line)
            doc_lines.append(doc_line)

        #cast overlap line to textbox
        doc_lines = sorted(doc_lines, key=lambda x:x['bbox'][1])
        text_boxs = []
        # new_doc_lines = []
        # for idx, doc_line in enumerate(doc_lines):
        #     if idx>0 and idx<len(doc_lines)-1:
        #         cur_y1, cur_y2 = doc_line['bbox'][1], doc_line['bbox'][3]
        #         last_y2, next_y1 = doc_lines[idx-1]['bbox'][3], doc_lines[idx+1]['bbox'][1]
        #         if cur_y1<last_y2 and cur_y2>next_y1:
        #             doc_line['type'] = 'textbox'
        #             text_boxs.append(doc_line)
        #             continue
        #     new_doc_lines.append(doc_line)
        # doc_lines = new_doc_lines
        # #行合并后进行文本空格分离操作
        return doc_lines, text_boxs

    def cast_part_para(self, doc_colum):
        paras = []
        for idx, content in enumerate(doc_colum):
            if content['type']== 'paragraph':
                paras.append((idx, content['bbox']))
        if len(paras) <= 1:
            return doc_colum
        paras = sorted(paras, key=lambda x: x[1][1])
        y1 = paras[0][1][1]
        y2 = paras[0][1][3]
        cast_index = set()
        for u, (idx, pbox) in enumerate(paras[1:]):
            cy1 = pbox[1]
            cy2 = pbox[3]
            if cy1 < y2:
                cast_index.add(idx)
                cast_index.add(paras[u][0])
            y1 = cy1
            y2 = cy2
        res = []
        for idx, content in enumerate(doc_colum):
            if content['type']== 'paragraph' and idx in cast_index:
                for line in content['lines']:
                    line['colum_id'] = content['colum_id']
                    res.append(line)
            else:
                res.append(content)
        return res


    def group_paragraphs(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        if doc_colums==[]:
            return meta_data

        for colum_id, doc_colum in enumerate(doc_colums):
            new_doc_colum = {'name': 'colum', 'colum_space': 0.2}
            new_doc_colum['all_lines'] = []
            colum_x1 = self.valid_img_w
            colum_y1 = self.valid_img_h
            colum_x2 = 0
            colum_y2 = 0
            all_lines = []
            for doc_content in doc_colum:
                colum_x1 = min(colum_x1, doc_content['bbox'][0])
                colum_y1 = min(colum_y1, doc_content['bbox'][1])
                colum_x2 = max(colum_x2, doc_content['bbox'][2])
                colum_y2 = max(colum_y2, doc_content['bbox'][3])
                if doc_content['type'] == 'line':
                    all_lines.append(doc_content)
                    continue
                for doc_line in doc_content['lines']:
                    all_lines.append(doc_line)
            colum_bbox = [colum_x1, colum_y1, colum_x2, colum_y2]
            new_doc_colum['bbox'] = colum_bbox
            new_doc_colum['all_lines'] = all_lines
            doc_colums[colum_id] = new_doc_colum

        for idx, doc_colum in enumerate(doc_colums):
            bbox = doc_colum['bbox']
            doc_colum['page_range'] = [0, 0, (idx + 1) * self.valid_page_width / len(doc_colums),
                                       self.valid_page_height]
            if idx==0:
                bbox[0] = 0
            if idx==len(doc_colums)-1:
                bbox[2] = self.valid_img_w
            img_width_range = bbox[2] - bbox[0]
            x1 = self.trans_to_page_size(bbox[0], cal_w=True)
            page_width_range = self.trans_to_page_size(img_width_range, cal_w=True)
            doc_colum['page_range'][0] = 0.
            doc_colum['page_range'][2] = page_width_range
            edge = 0.
            if idx != 0:
                doc_colum['page_range'][0] = x1
                doc_colum['page_range'][2] = x1 + page_width_range
            if idx != len(doc_colums) - 1:
                next_x1 = self.trans_to_page_size(doc_colums[idx + 1]['bbox'][0], cal_w=True)
                edge = next_x1 - doc_colum['page_range'][2]
            doc_colum['colum_space'] = edge
            doc_colum['paragraphs'] = []

        for idx, doc_colum in enumerate(doc_colums):
            cur_colum_start = doc_colum['page_range'][0]
            cur_colum_end = doc_colum['page_range'][2]
            if colum_id > 0:
                cur_colum_start += doc_colums[colum_id - 1]['colum_space'] / 2
            if colum_id < len(doc_colums) - 1:
                cur_colum_end -= doc_colum['colum_space'] / 2
            all_lines = doc_colum['all_lines']
            #doc_colum['paragraphs'] = []
            all_lines = sorted(all_lines, key=lambda x:x['bbox'][1])
            if all_lines==[]:
                return meta_data
            cur_paragraph = {'type':'paragraph', 'lines':[all_lines[0]]}
            para_bbox = all_lines[0]['bbox'].copy()
            last_bbox = all_lines[0]['bbox'].copy()
            para_font_list = [all_lines[0]['font_size']]
            for idx, line in enumerate(all_lines):
                if idx==0:
                    continue
                cur_bbox = line['bbox']
                if cur_paragraph['lines']==[]:
                    cur_paragraph['lines'].append(line)
                    last_bbox = line['bbox']
                    continue
                font_size_cm = pt2cm(line['font_size'])
                k1 = self.trans_to_page_size(cur_bbox[1] - last_bbox[1], cal_h=True)/font_size_cm
                k2 = self.trans_to_page_size(cur_bbox[1] - last_bbox[3], cal_h=True)/font_size_cm
                last_line_end_diff = (self.trans_to_page_size(cur_bbox[2] - last_bbox[2], cal_w=True))
                cur_line_start_diff = abs(self.trans_to_page_size(cur_bbox[0] - last_bbox[0], cal_w=True))
                k41 = abs(last_line_end_diff/font_size_cm)
                k42 = last_line_end_diff/font_size_cm
                k3 = cur_line_start_diff/font_size_cm
                if k1<2.2 and k2<1 and len(all_lines[idx-1]['sentences'])==1 and ((len(cur_paragraph['lines'])==1 and k3<3) \
                                 or (k3<0.5)) and (k41<0.5 or k42<0) and is_need_group(line) and len(cur_paragraph['lines'][-1]['sentences'][0]['text'])>5:
                    cur_paragraph['lines'].append(line)
                    para_bbox[0] = min(para_bbox[0], cur_bbox[0])
                    para_bbox[1] = min(para_bbox[1], cur_bbox[1])
                    para_bbox[2] = max(para_bbox[2], cur_bbox[2])
                    para_bbox[3] = max(para_bbox[3], cur_bbox[3])
                    para_font_list.append(line['font_size'])
                    if k41>0.5:
                        right_indent = cur_colum_end - self.trans_to_page_size(para_bbox[2], cal_w=True)
                        cur_paragraph['bbox'] = para_bbox
                        para_font_size = get_mid_num(para_font_list)
                        cur_paragraph['font_size'] = para_font_size
                        cur_paragraph['right_indent'] = right_indent
                        doc_colum['paragraphs'].append(cur_paragraph)
                        cur_paragraph = {'type': 'paragraph', 'lines': []}
                        if idx!=len(all_lines)-1:
                            para_bbox = all_lines[idx+1]['bbox'].copy()
                else:
                    cur_paragraph['bbox'] = para_bbox
                    right_indent = cur_colum_end - self.trans_to_page_size(para_bbox[2], cal_w=True)
                    para_font_size = get_mid_num(para_font_list)
                    cur_paragraph['font_size'] = para_font_size
                    cur_paragraph['right_indent'] = right_indent
                    doc_colum['paragraphs'].append(cur_paragraph)
                    cur_paragraph = {'type': 'paragraph', 'lines': [line]}
                    para_bbox = cur_bbox.copy()
                    para_font_list = [line['font_size']]
                last_bbox = cur_bbox.copy()
            if cur_paragraph['lines']!=[]:
                cur_paragraph['bbox'] = para_bbox
                right_indent= cur_colum_end - self.trans_to_page_size(para_bbox[2], cal_w=True)
                para_font_size = get_mid_num(para_font_list)
                cur_paragraph['font_size'] = para_font_size
                cur_paragraph['right_indent'] = right_indent
                doc_colum['paragraphs'].append(cur_paragraph)
            doc_colum['paragraphs'] = sorted(doc_colum['paragraphs'], key=lambda x:x['bbox'][1])
            del doc_colum['all_lines']
            #new_doc_colums.append(new_doc_colum)

        #doc_colums = new_doc_colums
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def calculate_font_size(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        all_font_list = []
        for colum_id, doc_colum in enumerate(doc_colums):
            for doc_content in doc_colum:
                if doc_content['type'] == 'paragraph':
                    para_font_list = []
                    for doc_line in doc_content['lines']:
                        line_font_size = self.calculate_line_font_size(doc_line)
                        doc_line['font_size'] = line_font_size
                        if line_font_size!=0:
                            para_font_list.append(line_font_size)
                            all_font_list.append(line_font_size)
                    if len(para_font_list)!=0:
                        doc_content['font_size'] = get_mid_num(para_font_list)
                    else:
                        doc_content['font_size'] = 0
                else:
                    line_font_size = self.calculate_line_font_size(doc_content)
                    doc_content['font_size'] = line_font_size
                    if line_font_size != 0:
                        all_font_list.append(line_font_size)
        if len(all_font_list) != 0:
            meta_data['doc_info']['font_size'] = get_mid_num(all_font_list)
        else:
            meta_data['doc_info']['font_size'] = 10
        global_font_size = meta_data['doc_info']['font_size']
        for colum_id, doc_colum in enumerate(doc_colums):
            for doc_content in doc_colum:
                if doc_content['type'] == 'paragraph':
                    if doc_content['font_size']-global_font_size<=1 or doc_content['font_size']==0:
                        doc_content['font_size'] = global_font_size
                    for doc_line in doc_content['lines']:
                        doc_line['font_size'] = doc_content['font_size']
                else:
                    if doc_content['font_size']-global_font_size<=1 or doc_content['font_size']==0:
                        doc_content['font_size'] = global_font_size
        meta_data['doc_info']['doc_sections'][0]['doc_colums']  = doc_colums
        return meta_data

    def calculate_line_font_size(self, line):
        real_width = 0.
        real_height = 0.
        base_width = 0.
        base_height = 0.
        used_text_len = 0
        all_text = ''
        for sentence in line['sentences']:
            text = sentence['text']
            text = text.replace('()', '(' + chr(12288) * 2 + ')').replace('（）', '（' + chr(12288) * 2 + '）')
            if "\"" in text and not self.is_english_doc:
                new_text = ''
                f = False
                for uu in text:
                    if uu=='\"' and not f:
                        new_text  += "“"
                        f = True
                    elif uu=='\"' and  f:
                        new_text += "”"
                        f = False
                    else:
                        new_text += uu
                sentence['text'] = new_text
            else:
                sentence['text'] = text
            if "_" in text  or text =='' or ('`' in text and '/' in text) or ('`' in text and '^' in text):
                continue
            text_box_w = sentence['bbox'][2] - sentence['bbox'][0]
            text_box_h = sentence['bbox'][3] - sentence['bbox'][1]
            text = text.replace("\"", "“").replace('-', '➖').replace('+', '➕')
            cur_width, cur_height = self.get_text_size(text, 10)
            used_text_len += len(text)
            all_text += text
            cur_width = cm2pt(cur_width)
            cur_height = cm2pt(cur_height)
            real_width += cm2pt(self.trans_to_page_size(text_box_w, cal_w=True))
            real_height += cm2pt(self.trans_to_page_size(text_box_h, cal_h=True))
            yuan_string = 0
            for i, uu in enumerate(text):
                if uu == '〇':
                    yuan_string += 1
                    continue
            cur_width = cur_width + yuan_string * 10
            if text[-1]=='。':
                cur_width -= 5
            base_width += cur_width
            base_height += cur_height
        if base_width==0:
            return 0
        ratio_w = real_width / base_width
        ratio_h = real_height / base_height * 0.9
        if ratio_w<ratio_h*0.5 and "……" in all_text:
            font_size = my_round(ratio_h * 10)
            # if ratio_w * 10 < 11:
            #     font_size = ratio_w * 10 // 0.5 * 0.5
            for sentence in line['sentences']:
                text_box_w = sentence['bbox'][2] - sentence['bbox'][0]
                text = sentence['text']
                no_point_text = ''
                for uu in text:
                    if uu!='…':
                        no_point_text += uu
                text_box_w = self.trans_to_page_size(text_box_w, cal_w=True)
                current_w = self.get_text_size(no_point_text, font_size)[0]
                num_point = (text_box_w - current_w) / self.get_text_size('·', font_size)[0]
                new_text = strip_overlap_char(text, '…', int(num_point))
                sentence['text'] = new_text
        ratio_w = min(ratio_w, ratio_h)
        font_size = my_round(ratio_w * 10)
        # if ratio_w*10<11:
        #     font_size = ratio_w*10//0.5*0.5
        if used_text_len<=2 and not is_all_chinese(all_text):
            font_size = 0
        if '@' in all_text:
            font_size = 0
        if '…' in all_text:
            font_size = 0
        font_size = max(5, font_size)
        return font_size

    def get_text_size(self, text, font_size=10):
        cur_en_font = ImageFont.truetype(os.path.join(self.work_dir, "font/times.ttf"), font_size)
        cur_cn_font = ImageFont.truetype(os.path.join(self.work_dir, "font/simsun.ttc"), font_size)
        # cur_en_font = ImageFont.truetype(os.path.join(self.work_dir, "font/times.ttf"), 10)
        # cur_cn_font = ImageFont.truetype(os.path.join(self.work_dir, "font/simsun.ttc"), 10)
        en_text = ''
        cn_text = ''
        text_width = 0
        valid_text = text.replace('`', '').replace('^','').replace('\\sqrt', '√')
        for i, uu in enumerate(valid_text):
            if uu in self.size_map.keys():
                text_width += self.size_map[uu] / 10 * font_size
            elif ord(uu) in self.en_uniMap:
                en_text += uu
            else:
                cn_text += uu
        text_width += pt2cm(cur_cn_font.getsize(cn_text)[0] + cur_en_font.getsize(en_text)[0])
        text_height = max(pt2cm(cur_cn_font.getsize(valid_text)[1]), pt2cm(cur_en_font.getsize(valid_text)[1]))
        # text_width += pt2cm(cur_cn_font.getsize(cn_text)[0]/10*font_size + cur_en_font.getsize(en_text)[0]/10*font_size)
        # text_height = max(pt2cm(cur_cn_font.getsize(valid_text)[1]/10*font_size), pt2cm(cur_en_font.getsize(valid_text)[1]/10*font_size))
        return text_width, text_height

    def trans_to_page_size(self, asize, cal_w=False, cal_h=False, margin=0.):
        if cal_h:
            return asize/self.valid_img_h*(self.valid_page_height-margin)
        return asize/self.valid_img_w*(self.valid_page_width-margin)

    def cal_underline_space(self, orig_img, sentence, cur_font_size, offset_x, offset_y):
        x1, y1, x2, y2 = sentence['bbox']
        real_x1 = int(x1 + offset_x)
        real_x2 = int(x2 + offset_x)
        real_y1 = int(y1 + offset_y)
        real_y2 = int(y2 + offset_y)
        sub_img = orig_img[real_y1:real_y2, real_x1:real_x2]
        text = sentence['text']
        char_splits = char_parse(sub_img, text)
        ww, hh = char_splits.split(';')
        ww = ww.split()
        hh = hh.split()
        hh = [uu.split(',') for uu in hh]
        ww = [uu.split(',') for uu in ww]
        heights = []
        widths = []
        for i in range(len(hh)):
            hh[i] = [int(hh[i][0]), int(hh[i][1])]
            ww[i] = [int(ww[i][0]), int(ww[i][1])]
            heights.append(self.trans_to_page_size(hh[i][1] - hh[i][0], cal_h=True))
            widths.append(self.trans_to_page_size(ww[i][1] - ww[i][0], cal_w=True))
        base_height = pt2cm(cur_font_size)
        base_width = pt2cm(cur_font_size)
        underline_space = []
        underline_char_size = 0.5 * base_width
        if self.is_english_doc:
            underline_char_size = 0.7 * base_width
        for i in range(len(widths)):
            if widths[i] > 2 * base_width and heights[i] < 0.3 * base_height:
                underline_space.append([i, my_round(widths[i] / underline_char_size)])
        if underline_space == []:
            for i in range(len(widths)):
                if widths[i] > 2 * base_width:
                    underline_space.append([i, my_round(widths[i] / underline_char_size)])
        return underline_space, len(widths)

    def get_text_width_underline(self, img, sentence, font_size, img_left_margin, img_top_margin):
        text = sentence['text']
        underline_space, n_space = self.cal_underline_space(img, sentence, font_size, img_left_margin,
                                                            img_top_margin)
        uidx = 0
        new_text = ''
        for _char in text:
            if _char != '_':
                new_text += _char
            elif _char == '_' and uidx <= len(underline_space) - 1:
                new_text += underline_space[uidx][1] * '_'
                uidx += 1
            else:
                new_text += '_' * 5
        text = new_text
        return text

    def get_line_left_indent(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        for colum_id, doc_colum in enumerate(doc_colums):
            colum_start = doc_colum['page_range'][0]
            colum_end = doc_colum['page_range'][2]
            last_diff = -1
            for paragraph in doc_colum['paragraphs']:
                for line in paragraph['lines']:
                    line['font_size'] = paragraph['font_size']
                    line_start = self.trans_to_page_size(line['bbox'][0], cal_w=True)
                    diff = round(line_start - colum_start,2)
                    diff = max(0, diff)
                    if abs(diff-last_diff)<=0.1:
                        diff = last_diff
                    line['left_indent'] = max(0, diff)
                    last_diff = diff
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def get_line_text(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        img = meta_data['image_info']['img']
        img_left_margin = meta_data['image_info']['img_left_margin']
        img_top_margin = meta_data['image_info']['img_top_margin']
        for colum_id, doc_colum in enumerate(doc_colums):
            colum_start = doc_colum['page_range'][0]
            colum_end = doc_colum['page_range'][2]
            for paragraph in doc_colum['paragraphs']:
                for line_idx, line in enumerate(paragraph['lines']):
                    left_indent = line['left_indent']
                    font_size = line['font_size']
                    line_text = ''
                    real_width = colum_start + left_indent
                    if len(line['sentences'])==0:
                        line['line_text'] = ''
                        continue
                    elif len(line['sentences'])==1:
                        text = line['sentences'][0]['text']
                        if '_' in text and '`' not in text:
                            text = self.get_text_width_underline(img, line['sentences'][0], font_size, img_left_margin, img_top_margin)
                        line['line_text'] = text
                    else:
                        for idx, sentence in enumerate(line['sentences']):
                            text = sentence['text']
                            if '_' in text and '`' not in text:
                                text = self.get_text_width_underline(img, sentence, font_size, img_left_margin, img_top_margin)
                            if idx==0:
                                line_text+=text
                                cur_width = self.get_text_size(text, font_size)[0]
                                real_width += cur_width
                            elif idx>=1:
                                # space = sentence['bbox'][0] - line['sentences'][idx-1]['bbox'][2]
                                cur_width = self.get_text_size(text, font_size)[0]
                                space = self.trans_to_page_size(sentence['bbox'][0], cal_w=True) - real_width
                                max_space = int((colum_end-real_width-cur_width)/(self.size_map[' ']*font_size/10))
                                num_space = int(space/(self.size_map[' ']*font_size/10))
                                num_space = min(max_space, num_space)
                                real_width += self.get_text_size(' '*num_space, font_size)[0]
                                real_width += cur_width
                                line_text += num_space*' '
                                line_text += text
                        line['line_text'] = line_text
                    if line_idx!=len(paragraph['lines'])-1 and len(line['line_text'])>=1 and  not '\u4e00' <= line['line_text'][-1] <= '\u9fa5' and\
                     len(paragraph['lines'][line_idx+1]['sentences']) >=1 and  paragraph['lines'][line_idx+1]['sentences'][0]['text']!="" \
                            and not '\u4e00' <= paragraph['lines'][line_idx+1]['sentences'][0]['text'][0] <= '\u9fa5':
                        line['line_text'] += ' '
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def calculate_font_space(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        for colum_id, doc_colum in enumerate(doc_colums):
            colum_page_range = doc_colum['page_range']
            colum_start = colum_page_range[0]
            colum_end = colum_page_range[2]

            colum_bbox = doc_colum['bbox']
            colum_bbox_range = colum_bbox[2] - colum_bbox[0]
            colum_bbox_mid = (colum_bbox[2] + colum_bbox[0])/2
            for paragraph in doc_colum['paragraphs']:
                font_size_cm = pt2cm(paragraph['font_size'])
                paragraph['align'] = 'both'
                right_indent = paragraph['right_indent']
                cur_para_end = colum_end - right_indent
                if len(paragraph['lines'])==1:
                    start = paragraph['bbox'][0]
                    end = paragraph['bbox'][2]
                    mid = (start + end)/2
                    if abs(mid-colum_bbox_mid)<=colum_bbox_range*0.05 and (start-colum_bbox[0])>=colum_bbox_range*0.15 \
                            and (colum_end-end)<=0.8*colum_bbox_range:
                        paragraph['align'] = 'center'
                    #continue
                for line in paragraph['lines']:
                    line_end = self.trans_to_page_size(line['bbox'][2], cal_w=True)
                    if len(paragraph['lines'])!=1:
                        diff = (cur_para_end - line_end)/font_size_cm
                        base_end = cur_para_end
                        margin_threshold = 0.01 * (base_end - colum_start)
                    else:
                        diff = (colum_end - line_end) / font_size_cm
                        base_end = cur_para_end
                        #margin_threshold = 0.5*font_size_cm
                        margin_threshold = 0.01 * (base_end - colum_start)
                    # diff = (self.trans_to_page_size(colum_bbox[2], cal_w=True) - line_end)/font_size_cm
                    if diff <=2:
                        line['w_ratio'], line['w_space'] = self.cal_line_font_space(line, [colum_start, base_end], margin_threshold)
                    else:
                        line['w_ratio'], line['w_space'] = 100, 0
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def cal_line_font_space(self, line, base_range, margin_threshold=0.35):
        colum_start = base_range[0]
        colum_end = base_range[1]
        base_width = colum_end - colum_start - margin_threshold
        ratio = 100
        diff = 0
        if len(line['sentences'])==1:
            line_text = line['line_text']
            if line_text == '':
                return 100, 0
            left_indent = line['left_indent']
            base_width -= left_indent
            real_width = self.get_text_size(line_text, line['font_size'])[0]
            if not is_chinese(line_text):
                ratio = int(base_width/real_width*100)
                if ratio>=105:
                    ratio = 105
                elif ratio<=95:
                    ratio = 95
                diff = round(cm2pt((base_width - ratio/100*real_width)/len(line_text)), 2)
            else:
                diff = round(cm2pt((base_width - real_width) / len(line_text)), 2)
        return ratio, diff

    def calculate_paragraph_attributes(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        for colum_id, doc_colum in enumerate(doc_colums):
            paragraphs = doc_colum['paragraphs']
            doc_colum_height = 0
            cache = 0.
            for para_id, paragraph in enumerate(paragraphs):
                font_size = paragraph['font_size']
                para_bbox = paragraph['bbox']
                paragraph['page_height_start'] = doc_colum_height
                a = paragraph['lines'][0]['line_text']
                if len(paragraph['lines'])==1:
                    k = 1.1
                    if self.is_english_doc:
                        k = 1.
                    if '_' in paragraph['lines'][0]['line_text']:
                        k = 1.2
                    if '`' in paragraph['lines'][0]['line_text'] and '/' in paragraph['lines'][0]['line_text']:
                        k = 2.7
                    paragraph['line_space'] = round(k*pt2cm(font_size), 2)
                else:
                    line_space = self.cal_line_space(paragraph['lines'])
                    paragraph['line_space'] = line_space
                cur_para_h = self.trans_to_page_size(para_bbox[1], cal_h=True)
                up_space = 0.8*(paragraph['line_space']-pt2cm(font_size))
                cur_para_h -= up_space
                para_space = cur_para_h - doc_colum_height + cache
                if para_space>=0:
                    paragraph['para_space_before'] = para_space
                else:
                    paragraph['para_space_before'] = 0
                    #cache = para_space
                doc_colum_height = doc_colum_height + paragraph['line_space']*len(paragraph['lines']) + paragraph['para_space_before']
                paragraph['page_height_end'] = doc_colum_height
                paragraph['line_space_rule'] = 'exact'
                int_pad_line = 0
                if para_id!=len(paragraphs)-1:
                    next_para_h = self.trans_to_page_size(paragraphs[para_id+1]['bbox'][1], cal_h=True)
                    # pad_line = (next_para_h - doc_colum_height) / paragraph['line_space']
                    pad_line = (next_para_h - doc_colum_height) / pt2cm(font_size)
                    if pad_line >=2:
                        int_pad_line = int(pad_line)-1
                else:
                    int_pad_line = -1
                doc_colum_height = doc_colum_height + pt2cm(font_size)*int_pad_line
                paragraph['n_pad_line'] = int_pad_line
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data


    def calculate_paragraph_attributes1(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        line_pitch = 0.55
        for colum_id, doc_colum in enumerate(doc_colums):
            paragraphs = doc_colum['paragraphs']
            doc_colum_height = 0
            cache = 0.
            for para_id, paragraph in enumerate(paragraphs):
                font_size = paragraph['font_size']
                if font_size>=14:
                    font_size = 1.1
                para_bbox = paragraph['bbox']
                paragraph['page_height_start'] = doc_colum_height
                a = paragraph['lines'][0]['line_text']
                if len(paragraph['lines'])==1:
                    k = 1.1
                    if self.is_english_doc:
                        k = 1.
                    if '_' in paragraph['lines'][0]['line_text']:
                        k = 1.2
                    if '`' in paragraph['lines'][0]['line_text'] and '/' in paragraph['lines'][0]['line_text']:
                        k = 2
                    line_space = round(k*pt2cm(font_size), 2)
                else:
                    line_space = self.cal_line_space(paragraph['lines'])
                cur_para_h = self.trans_to_page_size(para_bbox[1], cal_h=True)
                up_space = 0.8*(line_space-pt2cm(font_size))
                cur_para_h -= up_space
                para_space = cur_para_h - doc_colum_height + cache
                if para_space<=0:
                    paragraph['para_space_before'] = 0
                    paragraph['line_space'] = line_space
                    paragraph['line_space_rule'] = "exact"
                else:
                    nline = len(paragraph['lines'])
                    if nline==1 and ((para_space + line_space)<=line_pitch or ('`' in paragraph['lines'][0]['line_text'] and '/' in paragraph['lines'][0]['line_text'])):
                        paragraph['line_space'] = (para_space + nline * line_space) / nline
                        paragraph['line_space_rule'] = 'exact'
                        paragraph['para_space_before'] = 0.
                    elif nline==1 and (para_space + line_space)>line_pitch:
                        paragraph['line_space'] = line_pitch
                        paragraph['line_space_rule'] = 'single'
                        real_up_space =  0.8*(line_pitch - pt2cm(font_size))
                        paragraph['para_space_before'] = max(0, (para_space + nline * line_space) - nline * line_pitch + up_space - real_up_space)
                    elif nline!=1 and (para_space + line_space)/nline<=line_pitch:
                        paragraph['line_space'] = (para_space + nline * line_space) / nline
                        paragraph['line_space_rule'] = 'exact'
                        paragraph['para_space_before'] = 0.
                    elif nline!=1 and (para_space + line_space)/nline>line_pitch:
                        paragraph['line_space'] = line_pitch
                        paragraph['line_space_rule'] = 'single'
                        up_space = 0.5 * (line_pitch - pt2cm(font_size))
                        paragraph['para_space_before'] = max(0, (para_space + nline * line_space) - nline * line_pitch - up_space)
                doc_colum_height = doc_colum_height + paragraph['line_space']*len(paragraph['lines']) + paragraph['para_space_before']
                paragraph['page_height_end'] = doc_colum_height
                int_pad_line = 0
                if para_id!=len(paragraphs)-1:
                    next_para_h = self.trans_to_page_size(paragraphs[para_id+1]['bbox'][1], cal_h=True)
                    pad_line = (next_para_h - doc_colum_height) / pt2cm(font_size)
                    if pad_line >=2:
                        int_pad_line = int(pad_line)-1
                doc_colum_height = doc_colum_height + pt2cm(font_size)*int_pad_line
                paragraph['n_pad_line'] = int_pad_line
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def is_down_pic(self, bbox, meta_data, threshold=0.55):
        doc_info = meta_data['doc_info']
        doc_text_boxs = doc_info['doc_textboxs']
        doc_pics = doc_info['doc_pics']
        doc_tables = meta_data['tables']
        float_contents = doc_text_boxs + doc_pics + doc_tables
        for content in float_contents:
            if 'rect' in content.keys():
                cbbox = content['rect']
            else:
                cbbox = content['bbox']
            inter_x1 = max(bbox[0], cbbox[0])
            inter_x2 = min(bbox[2], cbbox[2])
            dis = self.trans_to_page_size(bbox[1]-cbbox[3], cal_h=True)
            if (bbox[1]>=cbbox[3] or  (bbox[1]<cbbox[3] and bbox[3]>cbbox[3])) and inter_x2>inter_x1 and self.trans_to_page_size(bbox[1]-cbbox[3], cal_h=True)<threshold :
                return True, content
        return False, None

    def calculate_paragraph_attributes2(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        #line_pitch = 0.55
        page_pad_space = 0.
        last_doc_colum_height = 0.
        for colum_id, doc_colum in enumerate(doc_colums):
            paragraphs = doc_colum['paragraphs']
            doc_colum_height = 0
            cache = 0.
            colum_pad_space = 0.
            for para_id, paragraph in enumerate(paragraphs):
                para_bbox = paragraph['bbox']
                is_down_pic1, content = self.is_down_pic(para_bbox, meta_data, 0.5)
                font_size = paragraph['font_size']
                if font_size<13:
                    line_pitch = 0.55
                elif font_size>=13:
                    line_pitch = 1.1
                elif font_size>=25:
                    line_pitch = 1.65
                paragraph['page_height_start'] = doc_colum_height
                a = paragraph['lines'][0]['line_text']
                if len(paragraph['lines'])==1:
                    k = 1.1
                    if self.is_english_doc:
                        k = 1.
                    if '_' in paragraph['lines'][0]['line_text']:
                        k = 1.2
                    if '`' in paragraph['lines'][0]['line_text'] and '/' in paragraph['lines'][0]['line_text']:
                        k = 2
                    line_space = round(k*pt2cm(font_size), 2)
                else:
                    line_space = self.cal_line_space(paragraph['lines'])
                cur_para_h = self.trans_to_page_size(para_bbox[1], cal_h=True)
                up_space = 0.8*(line_space-pt2cm(font_size))
                cur_para_h -= up_space
                para_space = cur_para_h - doc_colum_height + cache
                if para_space<=0:
                    paragraph['para_space_before'] = 0
                    paragraph['line_space'] = line_space
                    paragraph['line_space_rule'] = "exact"
                else:
                    nline = len(paragraph['lines'])
                    if nline==1 and ((para_space + line_space)<=line_pitch or ('`' in paragraph['lines'][0]['line_text'] and '/' in paragraph['lines'][0]['line_text'])):
                        paragraph['line_space'] = (para_space + nline * line_space) / nline
                        paragraph['line_space_rule'] = 'exact'
                        paragraph['para_space_before'] = 0.
                    elif nline==1 and (para_space + line_space)>line_pitch:
                        paragraph['line_space'] = line_pitch
                        paragraph['line_space_rule'] = 'single'
                        real_up_space =  0.8*(line_pitch - pt2cm(font_size))
                        if is_down_pic1:
                            paragraph['para_space_before'] = max(0, para_space + up_space - real_up_space)
                            colum_pad_space += line_pitch - line_space
                        else:
                            paragraph['para_space_before'] = max(0, para_space + line_space - line_pitch + up_space - real_up_space)
                    elif nline!=1 and (para_space + nline * line_space)/nline<=line_pitch:
                        paragraph['line_space'] = (para_space + nline * line_space) / nline
                        paragraph['line_space_rule'] = 'exact'
                        paragraph['para_space_before'] = 0.
                    elif nline!=1 and (para_space + nline * line_space)/nline>line_pitch:
                        paragraph['line_space'] = line_pitch
                        paragraph['line_space_rule'] = 'single'
                        real_up_space = 0.8 * (line_pitch - pt2cm(font_size))
                        if is_down_pic1:
                            paragraph['para_space_before'] = max(0, para_space + up_space - real_up_space)
                            colum_pad_space += (line_pitch - line_space) * nline
                        else:
                            paragraph['para_space_before'] = max(0, (para_space + nline * line_space) - nline * line_pitch + up_space - real_up_space)

                doc_colum_height = doc_colum_height + paragraph['line_space']*len(paragraph['lines']) + paragraph['para_space_before']
                paragraph['page_height_end'] = doc_colum_height
                int_pad_line = 0
                if para_id!=len(paragraphs)-1:
                    next_para_h = self.trans_to_page_size(paragraphs[para_id+1]['bbox'][1], cal_h=True)
                    pad_line = (next_para_h - doc_colum_height) / pt2cm(font_size)
                    if pad_line >=2:
                        int_pad_line = int(pad_line)-1
                else:
                    if colum_id!=0 and doc_colum_height<last_doc_colum_height:
                        pad_line = (last_doc_colum_height - doc_colum_height) / pt2cm(font_size)
                        int_pad_line = int(pad_line)
                    else:
                        int_pad_line = -1
                doc_colum_height = doc_colum_height + pt2cm(font_size)*int_pad_line
                paragraph['n_pad_line'] = int_pad_line
            last_doc_colum_height = doc_colum_height
            page_pad_space = max(page_pad_space, colum_pad_space)
        meta_data['doc_info']['doc_page_info']['page_top_bottom_margin'] = max(0.5, meta_data['doc_info']['doc_page_info']['page_top_bottom_margin']*0.8 - page_pad_space*0.6)
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def cal_line_space(self, lines):
        font_size_cm = pt2cm(lines[0]['font_size'])
        last_h = (lines[0]['bbox'][1] + lines[0]['bbox'][3])/2
        h_space_list = []
        has_underline = False
        for line in lines[1:]:
            cur_h = (line['bbox'][1] + line['bbox'][3])/2
            h_space_list.append(cur_h - last_h)
            if '_' in line['line_text']:
                has_underline = True
            last_h = cur_h
        k = 1.1
        if has_underline:
            k = 1.2
        if self.is_english_doc:
            k = 1.
        h_space = sum(h_space_list)/len(h_space_list)
        line_space = round(self.trans_to_page_size(h_space, cal_h=True), 2)
        line_space = max(line_space, k*font_size_cm)
        return line_space

    def latex_to_omml(self, latex_input):
        mathml = latex2mathml.converter.convert(latex_input)
        tree = etree.fromstring(mathml)
        xslt = etree.parse(os.path.join(self.work_dir, 'font', 'MML2OMML.XSL'))
        transform = etree.XSLT(xslt)
        new_dom = transform(tree)
        omml_string = str(new_dom).split('\n')[1]
        return 'omml' + omml_string

    def generate_formula(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        for colum_id, doc_colum in enumerate(doc_colums):
            paragraphs = doc_colum['paragraphs']
            for para_id, paragraph in enumerate(paragraphs):
                for line in paragraph['lines']:
                    font_size = line['font_size']
                    line_text = line['line_text']
                    if '`' not in line_text and '√' not in line_text:
                        line['line_text'] = [{"text":line_text}]
                    else:
                        line_text = math_formula_parse(line_text)
                        omml_line_text = []
                        for uu in line_text:
                            if 'latex' in uu:
                                uu = uu.replace('latex', '')
                                #print(uu)
                                try:
                                    omml_string = self.latex_to_omml(uu)
                                except:
                                    omml_line_text.append(uu)
                                    continue
                                omml_string = add_font_size(omml_string, font_size)
                                omml_line_text.append(omml_string)
                            else:
                                omml_line_text.append(uu)
                        line_text = omml_line_text
                        new_line_text = []
                        for uu in line_text:
                            new_line_text.append({"text":uu})
                        line['line_text'] = new_line_text
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def generate_pictures(self, meta_data):
        doc_pics = meta_data['doc_info']['doc_pics']
        doc_page_info = meta_data['doc_info']['doc_page_info']
        image_info = meta_data['image_info']
        image = image_info['img']
        index = 1
        new_doc_pics = []
        for doc_pic in doc_pics:
            bbox = doc_pic['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            real_x1 = int(x1 + image_info['img_left_margin'])
            real_x2 = int(x2 + image_info['img_left_margin'])
            real_y1 = int(y1 + image_info['img_top_margin'])
            real_y2 = int(y2 + image_info['img_top_margin'])
            if real_y1>=real_y2 or real_x1>=real_x2 or real_x2==0 or real_y2==0:
                continue
            crop_img = image[real_y1:real_y2, real_x1:real_x2]
            # url = os.path.join(self.work_dir, 'tmp', self.prefix+'_pic'+'%03d.jpg'%index)
            url = doc_pic['url']
            cv2.imwrite(url, crop_img)
            page_x1 = self.trans_to_page_size(x1, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y1 = self.trans_to_page_size(y1, cal_h=True) + doc_page_info['page_top_bottom_margin']
            page_x2 = self.trans_to_page_size(x2, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y2 = self.trans_to_page_size(y2, cal_h=True) + doc_page_info['page_top_bottom_margin']
            wcm = page_x2 - page_x1
            hcm = page_y2 - page_y1
            img_wh_ratio = (real_x2-real_x1)/(real_y2-real_y1)
            page_wh_ratio = wcm/hcm
            if page_wh_ratio>img_wh_ratio:
                wcm = hcm*img_wh_ratio
            else:
                hcm = wcm/img_wh_ratio
            doc_pic['url'] = url
            doc_pic['page_bbox'] = [page_x1, page_y1, page_x1+wcm*0.9, page_y1+hcm*0.9]
            new_doc_pics.append(doc_pic)
            index += 1
        meta_data['doc_info']['doc_pics'] = new_doc_pics
        return meta_data

    def generate_text_boxs(self, meta_data):
        doc_textboxs = meta_data['doc_info']['doc_textboxs']
        doc_page_info = meta_data['doc_info']['doc_page_info']
        img = meta_data['image_info']['img']
        img_left_margin = meta_data['image_info']['img_left_margin']
        img_top_margin = meta_data['image_info']['img_top_margin']
        global_font_size = meta_data['doc_info']['font_size']
        for textbox in doc_textboxs:
            bbox = textbox['bbox']
            x1, y1, x2, y2 = bbox
            page_x1 = self.trans_to_page_size(x1, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y1 = self.trans_to_page_size(y1, cal_h=True) + doc_page_info['page_top_bottom_margin']
            page_x2 = self.trans_to_page_size(x2, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y2 = self.trans_to_page_size(y2, cal_h=True) + doc_page_info['page_top_bottom_margin']
            text = textbox['text']
            if '_' in text:
                text = self.get_text_width_underline(img, textbox, global_font_size, img_left_margin, img_top_margin)
            textbox['text'] = text.replace('()', '(  )').replace('（）', '（  ）').replace('`','')
            cur_w, cur_h = self.get_text_size(text, 10)
            real_w, real_h = self.trans_to_page_size(x2-x1, cal_w=True), self.trans_to_page_size(y2-y1, cal_h=True)
            font_size = int(min(real_h / cur_h, real_w / cur_w) * 10)
            line_space = 1.5*font_size
            page_y1 = page_y1-pt2cm(line_space-font_size)*0.5
            textbox['page_bbox'] = [page_x1, page_y1, page_x2, page_y2]
            textbox['font_size'] = font_size
        meta_data['doc_info']['doc_textboxs'] = doc_textboxs
        return meta_data

    def generate_tables(self, meta_data):
        tables = meta_data['doc_info']['doc_tables']
        doc_page_info = meta_data['doc_info']['doc_page_info']
        for aa, table in enumerate(tables):
            #1.
            bbox = table['bbox']
            # table['bbox'] = bbox
            # del table['rect']
            x1, y1, x2, y2 = bbox
            page_x1 = self.trans_to_page_size(x1, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y1 = self.trans_to_page_size(y1, cal_h=True) + doc_page_info['page_top_bottom_margin'] + 0.15
            page_x2 = self.trans_to_page_size(x2, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y2 = self.trans_to_page_size(y2, cal_h=True) + doc_page_info['page_top_bottom_margin'] - 0.15
            table['page_bbox'] = [page_x1, page_y1, page_x2, page_y2]
            #2.
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
            cell_index = cast_invalid_cell(row_num, col_num, cell_index)

            if cell_index==[] or cell_index[0]==[]:
                continue
            table['row_num'] = len(cell_index)
            table['col_num'] = len(cell_index[0])
            all_font_list = []
            #3.
            for cell_id, cell in enumerate(cells):
                startrow = cell['startrow']
                endrow = cell['endrow']
                startcol = cell['startcol']
                endcol = cell['endcol']
                bbox = cell['bbox']
                page_x1 = self.trans_to_page_size(bbox[0], cal_w=True) + doc_page_info['page_left_right_margin']
                page_y1 = self.trans_to_page_size(bbox[1], cal_h=True) + doc_page_info['page_top_bottom_margin']
                page_x2 = self.trans_to_page_size(bbox[2], cal_w=True) + doc_page_info['page_left_right_margin']
                page_y2 = self.trans_to_page_size(bbox[5], cal_h=True) + doc_page_info['page_top_bottom_margin']
                cell['page_bbox'] = [page_x1, page_y1, page_x2, page_y2]
                cell['page_width'] = page_x2 - page_x1
                cell['page_height'] = page_y2 - page_y1
                cell['page_split_width'] = (page_x2 - page_x1) / (endcol + 1 - startcol)
                cell['page_split_height'] = (page_y2 - page_y1) / (endrow + 1 - startrow)
                texts = cell['texts']
                lines, _ = self.merge_sentences(texts)
                lines = sorted(lines, key=lambda x:x['bbox'][1])
                doc_colums = {'name':'cloum', 'bbox':[bbox[0], bbox[1], bbox[2], bbox[5]], 'page_range':cell['page_bbox']}
                paragraphs = []
                for line in lines:
                    paragraph = {'type':'paragraph', 'bbox':line['bbox']}
                    new_line = {'type':'line', 'bbox':line['bbox']}
                    sentences = []
                    for sentence in line['sentences']:
                        sentences.append({
                          'text':sentence['content'],
                          'bbox':sentence['bbox'][:4],
                          'type':'text'
                        })
                    cur_x1 = self.trans_to_page_size(line['bbox'][0], cal_w=True) + doc_page_info['page_left_right_margin']
                    new_line['sentences'] = sentences
                    line_font_size = self.calculate_line_font_size(new_line)
                    new_line['font_size'] = line_font_size
                    new_line['left_indent'] = max(0., cur_x1-page_x1)
                    new_line['w_ratio'] = 100
                    new_line['w_space'] = 0
                    if line_font_size != 0:
                        all_font_list.append(line_font_size)
                    paragraph['lines'] = [new_line]
                    paragraph['font_size'] = line_font_size
                    paragraph['align'] = 'left'
                    paragraph['right_indent'] = 0
                    paragraphs.append(paragraph)

                cell['paragraphs'] = paragraphs
            table_font_size = 8
            if all_font_list!=[]:
                table_font_size = get_mid_num(all_font_list)
            for cell_id, cell in enumerate(cells):
                paragraphs = cell['paragraphs']
                if paragraphs==[]:
                    continue
                for paragraph in paragraphs:
                    if paragraph['font_size']==0 or abs(paragraph['font_size']-table_font_size)<=1:
                        paragraph['font_size'] = table_font_size
                    font_size = paragraph['font_size']
                    for line in  paragraph['lines']:
                        line_text = ''
                        for idx, sentence in enumerate(line['sentences']):
                            text = sentence['text']
                            if '_' in text:
                                text = text.replace('_', '____')
                            if idx == 0:
                                line_text += text
                            elif idx >= 1:
                                space = self.trans_to_page_size(sentence['bbox'][0] - line['sentences'][idx-1]['bbox'][2], cal_w=True)
                                num_space = int(space / self.size_map[' '])
                                line_text += num_space * ' '
                                line_text += text
                        line_text = line_text.replace('`', '')
                        #line_text = math_formula_parse(line_text)
                        new_line_text = []
                        for uu in line_text:
                            new_line_text.append({"text":uu})
                        line['line_text'] = new_line_text
                doc_colum_height = self.trans_to_page_size(paragraphs[0]['bbox'][1], cal_h=True)
                for para_id, paragraph in enumerate(paragraphs):
                    font_size = paragraph['font_size']
                    para_bbox = paragraph['bbox']
                    paragraph['page_height_start'] = doc_colum_height
                    if len(paragraph['lines']) == 1:
                        k = 1.2
                        if '_' in paragraph['lines'][0]['line_text']:
                            k = 1.3
                        paragraph['line_space'] = round(k * pt2cm(font_size), 2)
                    else:
                        line_space = self.cal_line_space(paragraph['lines'])
                        paragraph['line_space'] = line_space
                    cur_para_h = self.trans_to_page_size(para_bbox[1], cal_h=True)
                    up_space = 0.8 * (paragraph['line_space'] - pt2cm(font_size))
                    cur_para_h -= up_space
                    para_space = cur_para_h - doc_colum_height
                    if para_space >= 0:
                        paragraph['para_space_before'] = para_space
                    else:
                        paragraph['para_space_before'] = 0
                        if para_id >= 1:
                            paragraphs[para_id - 1]['para_space_before'] -= para_space
                    doc_colum_height = doc_colum_height + paragraph['line_space'] * len(
                        paragraph['lines']) + paragraph['para_space_before']
                    int_pad_line = 0
                    if para_id != len(paragraphs) - 1:
                        next_para_h = self.trans_to_page_size(paragraphs[para_id + 1]['bbox'][1], cal_h=True)
                        pad_line = (next_para_h - doc_colum_height) / paragraph['line_space']
                        if pad_line >= 2:
                            int_pad_line = int(pad_line) - 1
                    doc_colum_height = doc_colum_height + paragraph['line_space'] * int_pad_line
                    paragraph['n_pad_line'] = int_pad_line
                    paragraph['float_content_id'] = []
                cell['paragraphs'] = paragraphs
            #merged_row_index, merged_col_index = cast_merged_row_col(cell_index, cells)
            table['avg_row_height'] = (table['page_bbox'][3]-table['page_bbox'][1])/table['row_num']
            table['avg_col_width'] = (table['page_bbox'][2]-table['page_bbox'][0])/table['col_num']
            row_heights = []
            for i in range(table['row_num']):
                row_h = 0.
                for j in range(table['col_num']):
                    cell_idx = cell_index[i][j]
                    cell_h = cells[cell_idx]['page_split_height']
                    row_h = max(row_h, cell_h)
                if row_h==0:
                    row_h = table['avg_row_height']
                row_heights.append(row_h)
            col_widths = []
            for i in range(table['col_num']):
                col_w = 0.
                for j in range(table['row_num']):
                    cell_idx = cell_index[j][i]
                    cell_w = cells[cell_idx]['page_split_width']
                    col_w = max(col_w, cell_w)
                if col_w==0:
                    col_w = table['avg_row_height']
                col_widths.append(col_w)
            table['row_heights'] = row_heights
            table['col_widths'] = col_widths
            cell_index_dict = []
            for i in range(len(cell_index)):
                row = {'row':cell_index[i]}
                cell_index_dict.append(row)
            table['cell_index'] = cell_index_dict
        meta_data['doc_info']['doc_tables'] = tables
        return meta_data

    def set_index(self, meta_data):
        doc_info = meta_data['doc_info']
        page_top_bottom_margin = doc_info['doc_page_info']['page_top_bottom_margin']
        page_left_right_margin = doc_info['doc_page_info']['page_left_right_margin']
        doc_colums = doc_info['doc_sections'][0]['doc_colums']
        doc_text_boxs = doc_info['doc_textboxs']
        doc_pics = doc_info['doc_pics']
        doc_tables = doc_info['doc_tables']
        index = 1
        for id, doc_colum in enumerate(doc_colums):
            place_para = {'type':'paragraph', 'lines':[], 'bbox':[doc_colum['bbox'][0], 0, doc_colum['bbox'][2], 0], 'font_size':5, 'align':'both', 'is_place_para':1, \
                          'line_space':0.04, 'line_space_rule':"exact", "para_space_before":0, "page_height_start":0, "page_height_end":0, "right_indent":0., "n_pad_line":0}
            doc_colum['paragraphs'].insert(0, place_para)
            for para in doc_colum['paragraphs']:
                para['index'] = index
                para['float_content_id'] = []
                index += 1

        other_content = doc_text_boxs+doc_pics+doc_tables
        for index, content in enumerate(other_content):
            pic_bbox = content['bbox']
            content['index'] = index
            px1, px2 = pic_bbox[0], pic_bbox[2]
            cid = 0
            if len(doc_colums)==0:
                content['in_page'] = 1
                continue
            for idx, doc_colum in enumerate(doc_colums):
                colum_bbox = doc_colum['bbox']
                cx1, cx2 = colum_bbox[0], colum_bbox[2]
                if px1>=cx1 and px2<=cx2:
                    cid = idx
                    break
            doc_colum = doc_colums[cid]
            colum_width_start = doc_colum['page_range'][0]
            f = False
            for idx, para in enumerate(doc_colum['paragraphs']):
                pbbox = para['bbox']
                if pbbox[3]<=pic_bbox[1] and (idx==len(doc_colum['paragraphs'])-1 or doc_colum['paragraphs'][idx+1]['bbox'][3]>pic_bbox[1]):
                    para['float_content_id'].append(index)
                    content['position_h_from_paragraph'] = max(0, content['page_bbox'][1] - para['page_height_end'] - page_top_bottom_margin)
                    content['position_w_from_colum'] = content['page_bbox'][0] - colum_width_start - page_left_right_margin
                    content['in_page'] = 0
                    f = True
                    break
            if not f:
                content['in_page'] = 1
        # for content in doc_text_boxs:
        #     content['in_page'] = 1
        # for id, doc_colum in enumerate(doc_colums):
        #     for para in doc_colum['paragraphs']:
        #         print(para['float_content_id'])
        return meta_data



if __name__ == "__main__":
    dc = doc_caculator()
    img_file = '20220831-153635.jpg'
    json_file = 'test1.json'
    dc.process(img_file=img_file, json_file=json_file, work_dir='/home/ateam/xychen/projects/image_2_word/word')