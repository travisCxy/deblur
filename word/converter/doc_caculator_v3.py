# -*-coding:utf-8-*-
import os, sys
import cv2
import json
from fontTools.ttLib import TTFont
from PIL import ImageFont
from doc_utils import *
from zhconv import convert

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
class doc_caculator_v3:
    def __init__(self):
        self.size_map = {'a':0.156, 'b':0.175, 'c':0.156, 'd':0.175, 'e':0.155, 'f':0.113, 'g':0.175, 'h':0.175, 'i':0.101, 'j':0.0975, 'k':0.175, 'l':0.0974, 'm':0.273, 'n':0.175, 'o':0.175, \
        'p':0.175, 'q':0.175, 'r':0.118, 's':0.136, 't':0.0975, 'u':0.175, 'v':0.175, 'w':0.255, 'x':0.175, 'y':0.175, 'z':0.156, 'A':0.255, 'B':0.235,'C':0.247, 'D':0.255, 'E':0.215, 'F':0.196,\
        'G':0.255, 'H':0.255, 'I':0.118, 'J':0.137, 'K':0.255, 'L':0.215, 'M':0.3125, 'N':0.255, 'O':0.255, 'P':0.197, 'Q':0.255, 'R':0.234, 'S':0.196, 'T':0.216,  'U':0.255, 'V':0.255, 'X':0.255, 'Y':0.255,\
        'Z':0.216, 'W':0.333, ' ':0.088, ',':0.086, '1':0.1607, '2':0.175, '3':0.175, '4':0.175, '5':0.175, '6':0.175, '7':0.175, '8':0.175, '9':0.175, '0':0.175, '/':0.098, '.':0.086,\
        '“':0.352, '”':0.352, '∠':0.352, '°':0.352, '·':0.352, '＝':0.352, 'π':0.352, '×':0.352, '÷':0.352
        }
        tmp_map = {'a':'āáǎà', 'e':'ēéěè', 'i':'īíǐì','o':'ōóǒòūúǔùüǖǘǚǜ'}
        for k,v in tmp_map.items():
            for i in v:
                self.size_map[i] = self.size_map[k]
        self.my_type = {1:'text', 2:'answer', 3:'pic', 4:'paragraph', 10:'textbox', 5:'underline'}
        self.set_font()

    def process(self, img, meta_data, prefix, work_dir=''):
        self.work_dir = work_dir
        if os.path.exists(os.path.join(work_dir, 'tmp')) is False:
            os.makedirs(os.path.join(work_dir, 'tmp'))
        self.cur_en_font = ImageFont.truetype(os.path.join(self.work_dir, "font/times.ttf"), 12)
        self.cur_cn_font = ImageFont.truetype(os.path.join(self.work_dir, "font/simsun.ttc"), 12)
        self.prefix = prefix
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
        meta_data = self.calculate_paragraph_attributes2(meta_data)
        meta_data = self.generate_formula(meta_data)
        meta_data = self.generate_pictures(meta_data)
        meta_data = self.generate_text_boxs(meta_data)
        meta_data = self.generate_tables(meta_data)
        meta_data = self.set_index(meta_data)
        meta_data = self.process_multi_doc_colums(meta_data)
        return meta_data


    def set_font(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font = TTFont(os.path.join(current_dir, '../..', 'font', 'times.ttf'))
        self.en_uniMap = font['cmap'].tables[0].ttFont.getBestCmap().keys()

    def set_page_global_size(self, meta_data):
        img_h = meta_data['image_info']['ori_height']
        img_w = meta_data['image_info']['ori_width']
        is_horizontal_paper = False
        page_height = 29.7
        page_width = 21
        page_top_bottom_margin = 2.54
        # page_left_right_margin = 1.91
        page_left_right_margin = 2.5
        valid_page_width = page_width - page_left_right_margin * 2
        valid_page_height = page_height - page_top_bottom_margin * 2 - 0.35  # 0.35为众向余量
        # 判断是否为横向文本
        # if img_w > img_h:
        #     is_horizontal_paper = True
        #     page_height = 21
        #     page_width = 29.7
        #     page_top_bottom_margin = 1.91
        #     page_left_right_margin = 2.54
        #     valid_page_width = page_width - page_left_right_margin * 2
        #     valid_page_height = page_height - page_top_bottom_margin * 2 - 0.35
        img_left_margin = img_w
        img_right_margin = 0
        img_top_margin = img_h
        img_bottom_margin = 0

        pad_w = 0
        pad_h = 0
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
                # if img_hw_ratio - page_hw_ratio < 0.4:
                pad_w = (img_h / page_hw_ratio - img_w)
                img_w += pad_w
                #pad_w = pad_w//2
                # else:
                #     page_width = max(2 * page_left_right_margin + 0.1, page_height / (img_hw_ratio + 0.01))
                #     valid_page_width = page_width - page_left_right_margin * 2
        else:
            page_wh_ratio = page_width / page_height
            img_wh_ratio = img_w / img_h
            if img_wh_ratio < page_wh_ratio:
                pad_w = int(page_wh_ratio * img_h - img_w)
                img_w += pad_w
                #pad_w = pad_w//2
            else:
                pad_h = (img_w / page_wh_ratio - img_h)
                img_h += pad_h
        pad_w = 0
        # 计算有效内容占据的区域和图片边缘之间的边距
        for region in meta_data['regions']:
            bbox = region['region']
            rotation = region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            x2 = min(x2, img_w)
            y2 = min(y2, img_h)

            img_left_margin = min(img_left_margin, x1)
            img_right_margin = max(img_right_margin, x2)
            img_top_margin = min(img_top_margin, y1)
            img_bottom_margin = max(img_bottom_margin, y2)
            bbox[0] += pad_w
            bbox[2] += pad_w

        for region in meta_data['pics']:
            bbox = region['region']
            rotation = region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            x2 = min(x2, img_w)
            y2 = min(y2, img_h)

            img_left_margin = min(img_left_margin, x1)
            img_right_margin = max(img_right_margin, x2)
            img_top_margin = min(img_top_margin, y1)
            img_bottom_margin = max(img_bottom_margin, y2)
            bbox[0] += pad_w
            bbox[2] += pad_w

        for table in meta_data['tables']:
            bbox = table['rect']
            x1, y1, x2, y2 = bbox
            x2 = min(x2, img_w)
            y2 = min(y2, img_h)

            img_left_margin = min(img_left_margin, x1)
            img_right_margin = max(img_right_margin, x2)
            img_top_margin = min(img_top_margin, y1)
            img_bottom_margin = max(img_bottom_margin, y2)
            bbox[0] += pad_w
            bbox[2] += pad_w

        img_right_margin = img_w - img_right_margin - pad_w*2
        img_bottom_margin = img_h - img_bottom_margin
        #应付右侧空间过大的情况
        if img_right_margin>0.3*img_w:
            img_right_margin = min(img_right_margin, img_left_margin)
        img_bottom_margin = min(img_bottom_margin, img_top_margin)

        img_bottom_margin = min(img_bottom_margin, img_h*0.1)
        img_top_margin = min(img_top_margin, img_h*0.1)
        valid_img_h = max(0, img_h - img_top_margin - img_bottom_margin)
        valid_img_w = max(0, img_w - img_left_margin - img_right_margin)

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
        page_info['pad_w'] = pad_w
        page_info['pad_h'] = pad_h
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
        doc_underlines = []
        img_left_margin = meta_data['image_info']['img_left_margin']
        img_top_margin = meta_data['image_info']['img_top_margin']
        n_chars = 0
        n_en_chars = 0
        n_cht_chars = 0
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
            if cls==1 and result=='':
                continue
            n_chars += len(result)
            for uu in result:
                if (uu >= 'a' and uu <= 'z') or (uu >= 'A' and uu <= 'Z'):
                    n_en_chars += 1
                if uu!=convert(uu, 'zh-cn'):
                    n_cht_chars += 1
            result = result.replace("\\@", "@")
            result = replace_text(result)
            if cls==1 and result=='':
                continue
            cur_sentence = {}
            cur_sentence['bbox'] = [rx1, ry1, rx2, ry2]
            cur_sentence['type'] = self.my_type[cls]
            if cls==5:
                cur_sentence['text'] = '_'
                cur_sentence['type'] = 'text'
                doc_underlines.append(cur_sentence)
                continue
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
        if self.is_english_doc:
            for doc_text in doc_texts:
                doc_text['text'] = regular_split(doc_text['text'])
        else:
            for doc_text in doc_texts:
                if not is_english_string(doc_text['text']):
                    doc_text['text'] = replace_mark(doc_text['text'])

        if n_chars>1 and n_cht_chars * 1. / n_chars > 0.1:
            for doc_text in doc_texts:
                doc_text['text'] = convert(doc_text['text'], 'zh-hk')
            for doc_textbox in doc_textboxs:
                doc_textbox['text'] = convert(doc_textbox['text'], 'zh-hk')

        for pic_region in pics:
            bbox = pic_region['region']
            rotation = pic_region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            if x2-x1<1 or y2-y1<1:
                continue
            rx1, ry1, rx2, ry2 = move_bbox([x1, y1, x2, y2], img_left_margin, img_top_margin)
            result = ''
            cls = pic_region['cls']
            cur_sentence = {}
            cur_sentence['bbox'] = [rx1, ry1, rx2, ry2]
            cur_sentence['type'] = self.my_type[cls]
            cur_sentence['text'] = result
            if 'is_square_cell' not in pic_region.keys():
                cur_sentence['is_square_cell'] = False
            else:
                cur_sentence['is_square_cell'] = pic_region['is_square_cell']
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


        doc_underlines = cast_region(doc_underlines, doc_tables, 0.5)
        doc_underlines = cast_region(doc_underlines, doc_texts, 0.2)
        doc_underlines = cast_region(doc_underlines, doc_textboxs, 0.2)
        doc_texts.extend(doc_underlines)
        doc_texts = cast_region(doc_texts, doc_pics)
        doc_pics, f = crop_pics(doc_pics, doc_texts)
        doc_tables, f = crop_tables(doc_tables, doc_texts)
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
        doc_texts = doc_texts + doc_textboxs
        doc_texts = sorted(doc_texts, key=lambda x:x['bbox'][3])
        doc_paragraphs = []
        for doc_text in doc_texts:
            if doc_text['type']=='paragraph':
                doc_text['lines'] = []
                doc_paragraphs.append(doc_text)

        new_doc_texts = []
        #1.merge text of para
        for doc_text in doc_texts:
            if doc_text['type'] != 'text' and doc_text['type'] != 'textbox':
                continue
            bbox = doc_text['bbox']
            #这里若当前word的图片插入以浮动式图片为嵌入形式，则图片和文本分开操作
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
                    doc_text['in_topic'] = True
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
            if column_id < 0 or pages_x[column_id] + x_thresh*0.7 < cur_x1:
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

        if len(doc_colums)==1:
            h1_list = []
            h2_list = []
            for para in doc_colums[0]:
                h1_list.append(para['bbox'][1])
                h2_list.append(para['bbox'][3])
            cy1 = min(h1_list)
            cy2 = max(h2_list)
            cx1, cx2 = colum_ranges[0]
            left_overlap_h_range = 0.
            right_overlap_h_range = 0.
            left_texts_indexs = []
            right_texts_indexs = []
            used = [False]*len(doc_texts)
            for idx, doc_text in enumerate(doc_texts):
                bbox = doc_text['bbox']
                lx1, ly1, lx2, ly2 = bbox
                max_left = max(cx1, lx1)
                min_right = min(lx2, cx2)
                iou_1d = (min_right - max_left) / (lx2 - lx1)
                if iou_1d>0.1:
                    continue
                if ly1>=cy1 and ly2<=cy2 and lx2<cx1:
                    left_overlap_h_range+=ly2-ly1
                    left_texts_indexs.append(idx)
                elif ly1>=cy1 and ly2<=cy2 and lx1>cx2:
                    right_overlap_h_range+=ly2-ly1
                    right_texts_indexs.append(idx)
            if left_overlap_h_range/(cy2-cy1)>0.5:
                left_cx1_list = []
                left_cx2_list = []
                for doc_text_id in left_texts_indexs:
                    left_cx1_list.append(doc_texts[doc_text_id]['bbox'][0])
                    left_cx2_list.append(doc_texts[doc_text_id]['bbox'][2])
                colum_ranges.insert(0, [min(left_cx1_list), max(left_cx2_list)])
                left_texts = []
                for doc_text_id in left_texts_indexs:
                    doc_texts[doc_text_id]['in_topic'] = True
                    left_texts.append(doc_texts[doc_text_id])
                    used[doc_text_id] = True
                doc_colums.insert(0, left_texts)
            if right_overlap_h_range/(cy2-cy1)>0.5:
                right_cx1_list = []
                right_cx2_list = []
                for doc_text_id in right_texts_indexs:
                    right_cx1_list.append(doc_texts[doc_text_id]['bbox'][0])
                    right_cx2_list.append(doc_texts[doc_text_id]['bbox'][2])
                colum_ranges.append([min(right_cx1_list), max(right_cx2_list)])
                right_texts = []
                for doc_text_id in right_texts_indexs:
                    doc_texts[doc_text_id]['in_topic'] = True
                    right_texts.append(doc_texts[doc_text_id])
                    used[doc_text_id] = True
                doc_colums.append(right_texts)
            new_doc_texts = []
            for doc_text_id in range(len(used)):
                if used[doc_text_id] == False:
                    new_doc_texts.append(doc_texts[doc_text_id])
            doc_texts = new_doc_texts
        if len(doc_colums)>1:
            pad_h = meta_data['doc_info']['doc_page_info']['pad_h']
            self.valid_img_h -= pad_h
            meta_data['image_info']['valid_img_h'] -= pad_h
        # 如果当前内容无段落
        no_para = False
        if len(doc_paragraphs) == 0:
            no_para = True
            doc_colums = [[]]
            colum_ranges = [[0, meta_data['image_info']['valid_img_w']]]

        # 将段落之外的内容划分一下栏
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
            if max_iou==0:
                continue
            if min(ious)>0.2:
                index = 0
            doc_text['colum_id'] = index
            doc_colums[index].append(doc_text)
        if doc_colums==[[]]:
            doc_colums = []

        meta_data['doc_info']['doc_sections'] = []
        doc_section = {}
        doc_section['doc_colums'] = doc_colums
        meta_data['doc_info']['doc_sections'].append(doc_section)
        # meta_data['doc_info']['doc_textboxs'] = doc_textboxs
        meta_data['doc_info']['doc_textboxs'] = []
        del meta_data['doc_info']['doc_texts']
        return meta_data

    def merge_lines(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        doc_text_boxs = meta_data['doc_info']['doc_textboxs']
        for colum_id, doc_colum in enumerate(doc_colums):
            doc_colum = self.cast_part_para(doc_colum)
            doc_colum = sorted(doc_colum, key=lambda x:x['bbox'][1])
            new_doc_colum = []
            doc_ind_lines = []
            before_has_topic = False
            #merge para lines
            for doc_content in doc_colum:
                if doc_content['type'] == 'paragraph':
                    doc_ind_lines.extend(doc_content['lines'])
                    before_has_topic = True
                else:
                    doc_ind_lines.append(doc_content)
                    if not before_has_topic:
                        doc_content['before_has_topic'] = False
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
        last_bbox = sentences[0]['bbox']
        #遍历所有内容
        for sentence in sentences:
            cur_bbox = sentence['bbox']
            cur_h = (sentence['bbox'][3] + sentence['bbox'][1])/2
            cur_word_h = (sentence['bbox'][3] - sentence['bbox'][1])
            if cur_line==[]:
                cur_line.append(sentence)
                last_h = cur_h
                last_bbox = cur_bbox
                continue
            #若当前内容的y坐标和上一内容的y坐标差小于上一内容的高度值的一半，则将当前内容合并至上一内容
            if cur_h-last_h<cur_word_h*0.7 and iou_1d([last_bbox[0],last_bbox[2]], [cur_bbox[0], cur_bbox[2]], False)<0.2 and iou_1d([cur_bbox[0], cur_bbox[2]], [last_bbox[0],last_bbox[2]], False)<0.2:
                cur_line.append(sentence)
            else:
                cur_line = sorted(cur_line, key=lambda x:x['bbox'][0])
                doc_line['sentences'] = cur_line
                doc_line['bbox'] = get_line_bbox(doc_line)
                doc_lines.append(doc_line)
                doc_line = {'type':'line'}
                cur_line = [sentence]
                last_h = cur_h
            last_bbox = cur_bbox

        if len(cur_line)!=0:
            cur_line = sorted(cur_line, key=lambda x:x['bbox'][0])
            doc_line['sentences'] = cur_line
            doc_line['bbox'] = get_line_bbox(doc_line)
            doc_lines.append(doc_line)

        #cast overlap line to textbox
        doc_lines = sorted(doc_lines, key=lambda x:x['bbox'][1])
        text_boxs = []
        for doc_line in doc_lines:
            doc_line['in_topic'] = False
            for sentence in doc_line['sentences']:
                if 'in_topic' in sentence.keys() and sentence['in_topic'] is True:
                    doc_line['in_topic'] = True
                    break

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
            new_doc_colum = {'name': 'colum', 'colum_space': 0.0}
            new_doc_colum['all_lines'] = []
            colum_x1_list = []
            colum_y1_list = []
            colum_x2_list = []
            colum_y2_list = []

            all_lines = []
            if doc_colum==[]:
                new_doc_colum['bbox'] = [0,0,self.valid_img_w,self.valid_img_h]
                new_doc_colum['float_contents'] = []
                doc_colums[colum_id] = new_doc_colum
                continue
            for doc_content in doc_colum:
                if doc_content['in_topic']==True or len(doc_colums)==1:
                    colum_x1_list.append(doc_content['bbox'][0])
                    colum_y1_list.append(doc_content['bbox'][1])
                    colum_x2_list.append(doc_content['bbox'][2])
                    colum_y2_list.append(doc_content['bbox'][3])
                if doc_content['type'] == 'line':
                    doc_content['before_has_topic'] = True
                    for sentence in doc_content['sentences']:
                        if 'before_has_topic' in sentence.keys():
                            doc_content['before_has_topic'] = sentence['before_has_topic']
                            del sentence['before_has_topic']
                    all_lines.append(doc_content)
                    continue
                for doc_line in doc_content['lines']:
                    all_lines.append(doc_line)
            colum_bbox = [0, 0, self.valid_img_w, self.valid_img_h]
            if colum_x1_list!=[]:
                colum_x1 = min(colum_x1_list)
                colum_y1 = min(colum_y1_list)
                colum_x2 = max(colum_x2_list)
                colum_y2 = max(colum_y2_list)
                colum_bbox = [colum_x1, colum_y1, colum_x2, colum_y2]
            new_doc_colum['bbox'] = colum_bbox
            new_doc_colum['base_bbox'] = [uu+0.1 for uu in colum_bbox]
            new_doc_colum['all_lines'] = all_lines
            doc_colums[colum_id] = new_doc_colum
            doc_colums[colum_id]['float_contents'] = []

        doc_pics = meta_data['doc_info']['doc_pics']
        doc_tables = meta_data['doc_info']['doc_tables']
        other_content = doc_pics + doc_tables
        for index, content in enumerate(other_content):
            pic_bbox = content['bbox']
            content['index'] = index
            px1, px2 = pic_bbox[0], pic_bbox[2]
            if len(doc_colums) == 0:
                content['in_page'] = 1
                continue
            cid = 0
            max_iou = 0
            for idx, doc_colum in enumerate(doc_colums):
                cx1, cx2 = doc_colums[idx]['bbox'][0], doc_colums[idx]['bbox'][2]
                max_left = max(cx1, px1)
                min_right = min(px2, cx2)
                iou_1d = (min_right - max_left) / (px2 - px1)
                if idx==0:
                    iou_1d *= 2
                if iou_1d > max_iou:
                    max_iou = iou_1d
                    cid = idx
            content['colum_id'] = cid
            doc_colums[cid]['float_contents'].append(content)

        for idx, doc_colum in enumerate(doc_colums):
            for fc in doc_colum['float_contents']:
                doc_colums[idx]['bbox'][0] = min(doc_colums[idx]['bbox'][0], fc['bbox'][0])
                doc_colums[idx]['bbox'][1] = min(doc_colums[idx]['bbox'][1], fc['bbox'][1])
                doc_colums[idx]['bbox'][2] = max(doc_colums[idx]['bbox'][2], fc['bbox'][2])
                doc_colums[idx]['bbox'][3] = max(doc_colums[idx]['bbox'][3], fc['bbox'][3])
            for line in doc_colum['all_lines']:
                doc_colums[idx]['bbox'][0] = min(doc_colums[idx]['bbox'][0], line['bbox'][0])
                doc_colums[idx]['bbox'][1] = min(doc_colums[idx]['bbox'][1], line['bbox'][1])
                doc_colums[idx]['bbox'][2] = max(doc_colums[idx]['bbox'][2], line['bbox'][2])
                doc_colums[idx]['bbox'][3] = max(doc_colums[idx]['bbox'][3], line['bbox'][3])

        for idx, doc_colum in enumerate(doc_colums):
            bbox = doc_colum['bbox']
            doc_colum['page_range'] = [0, 0, self.valid_page_width,
                                       self.valid_page_height]
            if idx==0:
                bbox[0] = 0
            doc_colum['paragraphs'] = []

        for idx, doc_colum in enumerate(doc_colums):
            cur_colum_start = doc_colum['page_range'][0]
            cur_colum_end = doc_colum['page_range'][2]
            colum_bbox = doc_colum['bbox']
            colum_bbox_width = colum_bbox[2] - colum_bbox[0]
            colum_page_width = cur_colum_end - cur_colum_start
            all_lines = doc_colum['all_lines']
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
                r1 = k1 < 4 and k2 < 2
                r2 = is_same_para(line, cur_paragraph['lines'][-1], [doc_colum['base_bbox'][0], doc_colum['base_bbox'][2]], self.is_english_doc)
                if r1  and r2 :
                    cur_paragraph['lines'].append(line)
                    para_bbox[0] = min(para_bbox[0], cur_bbox[0])
                    para_bbox[1] = min(para_bbox[1], cur_bbox[1])
                    para_bbox[2] = max(para_bbox[2], cur_bbox[2])
                    para_bbox[3] = max(para_bbox[3], cur_bbox[3])
                    para_font_list.append(line['font_size'])
                    # if k41>0.5:
                    #     #right_indent = cur_colum_end - self.trans_to_page_size(para_bbox[2], cal_w=True)
                    #     right_indent = cur_colum_end - self.trans_size(para_bbox[2], colum_bbox_width, colum_page_width)
                    #     cur_paragraph['bbox'] = para_bbox
                    #     para_font_size = get_mid_num(para_font_list)
                    #     cur_paragraph['font_size'] = para_font_size
                    #     cur_paragraph['right_indent'] = right_indent
                    #     doc_colum['paragraphs'].append(cur_paragraph)
                    #     cur_paragraph = {'type': 'paragraph', 'lines': []}
                    #     if idx!=len(all_lines)-1:
                    #         para_bbox = all_lines[idx+1]['bbox'].copy()
                else:
                    cur_paragraph['bbox'] = para_bbox
                    # right_indent = cur_colum_end - self.trans_to_page_size(para_bbox[2], cal_w=True)
                    right_indent = cur_colum_end - self.trans_size(para_bbox[2], colum_bbox_width, colum_page_width)
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
                # right_indent= cur_colum_end - self.trans_to_page_size(para_bbox[2]-colum_bbox[0], cal_w=True)
                right_indent = cur_colum_end - self.trans_size(para_bbox[2], colum_bbox_width, colum_page_width)
                para_font_size = get_mid_num(para_font_list)
                cur_paragraph['font_size'] = para_font_size
                cur_paragraph['right_indent'] = right_indent
                doc_colum['paragraphs'].append(cur_paragraph)
            doc_colum['paragraphs'] = sorted(doc_colum['paragraphs'], key=lambda x:x['bbox'][1])
            for para in doc_colum['paragraphs']:
                para['before_has_topic'] = True
                for line in para['lines']:
                    if line['before_has_topic'] == False:
                        para['before_has_topic'] = False
                    del line['before_has_topic']
            del doc_colum['all_lines']
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
                        #line_font_size = 12
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
                    #line_font_size = 12
                    doc_content['font_size'] = line_font_size
                    if line_font_size != 0:
                        all_font_list.append(line_font_size)
        if len(all_font_list) != 0:
            meta_data['doc_info']['font_size'] = get_mid_num(all_font_list)
        else:
            meta_data['doc_info']['font_size'] = 10
        global_font_size = 12
        threshold = 2
        for colum_id, doc_colum in enumerate(doc_colums):
            for doc_content in doc_colum:
                if doc_content['type'] == 'paragraph' :
                    doc_content['font_size'] = global_font_size
                    for doc_line in doc_content['lines']:
                        doc_line['font_size'] = doc_content['font_size']
                elif  ('in_topic' in doc_content.keys() and doc_content['in_topic'] is True):
                    doc_content['font_size'] = global_font_size
                else:
                    if abs(doc_content['font_size']-global_font_size)<=threshold or doc_content['font_size']==0:
                        doc_content['font_size'] = global_font_size
        meta_data['doc_info']['doc_sections'][0]['doc_colums']  = doc_colums
        return meta_data

    def calculate_line_font_size(self, line, is_table=False):
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
        if not is_table:
            return 12
        real_width = 0.
        real_height = 0.
        base_width = 0.
        base_height = 0.
        used_text_len = 0
        all_text = ''
        for sentence in line['sentences']:
            text = sentence['text']
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
        ratio_h = real_height / base_height #* 0.9
        if ratio_w<ratio_h*0.5 and "……" in all_text:
            font_size = my_round(ratio_h * 10)
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
        if used_text_len<=2 and not is_all_chinese(all_text):
            font_size = 0
        if '@' in all_text:
            font_size = 0
        if '…' in all_text:
            font_size = 0
        if font_size!=0:
            font_size = max(5, font_size)
        if font_size==13:
            font_size = 12

        return font_size


    def get_text_size(self, text, font_size=10):

        if font_size==12:
            cur_en_font = self.cur_en_font
            cur_cn_font = self.cur_cn_font
        else:
            cur_en_font = ImageFont.truetype(os.path.join(self.work_dir, "font/times.ttf"), font_size)
            cur_cn_font = ImageFont.truetype(os.path.join(self.work_dir, "font/simsun.ttc"), font_size)
        en_text = ''
        cn_text = ''
        text_width = 0
        valid_text = text.replace('`', '').replace('^','').replace('\\sqrt', '¤')
        for i, uu in enumerate(valid_text):
            if uu in self.size_map.keys():
                text_width += self.size_map[uu] / 10 * font_size
            elif ord(uu) in self.en_uniMap:
                en_text += uu
            else:
                cn_text += uu
        text_width += pt2cm(cur_cn_font.getsize(cn_text)[0] + cur_en_font.getsize(en_text)[0])
        text_height = max(pt2cm(cur_cn_font.getsize(valid_text)[1]), pt2cm(cur_en_font.getsize(valid_text)[1]))
        return text_width, text_height

    def trans_to_page_size(self, asize, cal_w=False, cal_h=False, margin=0.):
        if cal_h:
            return asize/self.valid_img_h*(self.valid_page_height-margin)
        return asize/self.valid_img_w*(self.valid_page_width-margin)

    def trans_size(self, asize, scale, factor):
        return asize/scale*factor

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
        underline_char_size = 0.6 * base_width
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
        if underline_space==[]:
            return text

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
        threshold = 0.25
        if len(doc_colums)>1:
            threshold = 0.15
        is_multi_doc = len(doc_colums)>1
        for colum_id, doc_colum in enumerate(doc_colums):
            colum_start = doc_colum['page_range'][0]
            colum_end = doc_colum['page_range'][2]
            colum_bbox = doc_colum['bbox']
            pre_indents1 = [-1, -1, -1]
            pre_indents2 = [-1, -1, -1]
            colum_indents = []
            for paragraph in doc_colum['paragraphs']:
                right_indent_diff = 0.
                for line in paragraph['lines']:
                    line = self.get_pinyin_text(line, meta_data['doc_info']['doc_pics'], multi_colums=is_multi_doc)
                    line['font_size'] = paragraph['font_size']
                    diff = round(self.trans_to_page_size(line['bbox'][0]-colum_bbox[0], cal_w=True),2)
                    diff = max(0, diff)
                    if line['is_pinyin']:
                        line['left_indent'] = diff
                        continue
                    orig_diff = diff+0.
                    min_relative_diff = 1
                    min_relative_diff_idx = -1
                    for ii, pre_indent in enumerate(pre_indents2):
                        relative_diff = abs(pre_indent-diff)
                        if relative_diff<min_relative_diff:
                            min_relative_diff = relative_diff
                            min_relative_diff_idx = ii
                    if min_relative_diff<threshold:
                        pre_indents2[min_relative_diff_idx] = diff
                        diff = pre_indents1[min_relative_diff_idx]
                    else:
                        pre_indents1.pop(0)
                        pre_indents1.append(diff)
                        pre_indents2.pop(0)
                        pre_indents2.append(diff)
                    line['left_indent'] = diff
                    colum_indents.append(diff)
                    if diff>orig_diff:
                        right_indent_diff = max(right_indent_diff, diff-orig_diff)
                paragraph['right_indent'] -= right_indent_diff
                paragraph['right_indent'] = max(0, paragraph['right_indent'])
            for paragraph in doc_colum['paragraphs']:
                for line in paragraph['lines']:
                    if not line['is_pinyin']:
                        line['left_indent'] = line['left_indent']//0.21*0.21
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def get_pinyin_text(self, line, doc_pics, multi_colums=False):
        line['is_pinyin'] = False
        if len(line['sentences'])==0:
            return line
        for sentence in line['sentences']:
            if not is_pinyin(sentence['text']):
                return line

        f = False
        lx1, lx2 = line['bbox'][0], line['bbox'][2]
        for pic in doc_pics:
            if 'is_square_cell' not in pic.keys() or not  pic['is_square_cell']:
                continue
            px1, px2 = pic['bbox'][0], pic['bbox'][2]
            ix1 = max(lx1, px1)
            ix2 = min(lx2, px2)
            if ix1 < ix2 and (ix2 - ix1) / (px2 - px1) < 0.5:
                continue
            if pic['bbox'][1]+2>=line['bbox'][3] and pic['bbox'][1]-line['bbox'][3]<0.5*(line['bbox'][3]-line['bbox'][1]):
                f = True
                break
        if not f:
            return line
        img_wh_ratio = (pic['bbox'][2] - pic['bbox'][0]) / (pic['bbox'][3] - pic['bbox'][1])
        wcm = self.trans_to_page_size(pic['bbox'][2] - pic['bbox'][0], cal_w=True)
        hcm = self.trans_to_page_size(pic['bbox'][3] - pic['bbox'][1], cal_h=True)
        page_wh_ratio = wcm / hcm
        real_wcm = wcm
        if multi_colums:
            if page_wh_ratio > img_wh_ratio:
                real_wcm = hcm * img_wh_ratio
            else:
                real_wcm = hcm * img_wh_ratio
        else:
            if page_wh_ratio > img_wh_ratio:
                real_wcm = hcm * img_wh_ratio
        factor = 0.9 * real_wcm / wcm
        for sentence in line['sentences']:
            width = sentence['bbox'][2] - sentence['bbox'][0]
            page_width = self.trans_to_page_size(width, cal_w=True) * factor
            real_width = self.get_text_size(sentence['text'], 12)[0]
            pad_space = int((page_width - real_width) / (self.size_map[' ']*1.2))
            spaces = sentence['text'].count(' ')
            if spaces>0:
                pad_space = pad_space//spaces
            if pad_space<=0:
                continue
            pad_space = min(6, pad_space)
            sentence['text'] = sentence['text'].replace(' ', ' '*pad_space)
        line['is_pinyin'] = True
        return line

    def get_line_text(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        img = meta_data['image_info']['img']
        img_left_margin = meta_data['image_info']['img_left_margin']
        img_top_margin = meta_data['image_info']['img_top_margin']
        factor = len(doc_colums)
        for colum_id, doc_colum in enumerate(doc_colums):
            colum_start = doc_colum['page_range'][0]
            colum_end = doc_colum['page_range'][2]
            colum_bbox = doc_colum['bbox']
            for paragraph in doc_colum['paragraphs']:
                for line_idx, line in enumerate(paragraph['lines']):
                    left_indent = line['left_indent']
                    font_size = line['font_size']
                    line['use_tab'] = 0
                    line_text = ''
                    real_width = colum_start + left_indent
                    if len(line['sentences'])==0:
                        line['line_text'] = ''
                        continue
                    elif len(line['sentences'])==1:
                        text = line['sentences'][0]['text']
                        if '_' in text and '`' not in text and not is_chemical_str(text):
                            if '_' == text:
                                num_space = min(75, int((line['bbox'][2]-line['bbox'][0])/(colum_bbox[2]-colum_bbox[0])*75))
                                if num_space>=70:
                                    num_space = 72
                                else:
                                    num_space = num_space // 5 * 5
                                    if num_space%10>5:
                                        num_space+=5
                                text = '_' * num_space
                            else:
                                text = self.get_text_width_underline(img, line['sentences'][0], font_size, img_left_margin, img_top_margin)
                        line['line_text'] = text
                    else:
                        if is_math_caculate_line(line):
                            line['use_tab'] = 1
                        for idx, sentence in enumerate(line['sentences']):
                            text = sentence['text']
                            if '_' in text and '`' not in text:
                                if '_' == text:
                                    num_space = min(75, int((sentence['bbox'][2] - sentence['bbox'][0]) / (colum_bbox[2] - colum_bbox[0]) * 75))
                                    if num_space >= 70:
                                        num_space = 72
                                    else:
                                        num_space = num_space // 5 * 5
                                        if num_space % 10 > 5:
                                            num_space += 5
                                    text = '_' * num_space
                                else:
                                    text = self.get_text_width_underline(img, sentence, font_size, img_left_margin, img_top_margin)
                            if idx==0:
                                line_text+=text
                                cur_width = self.get_text_size(text, font_size)[0]
                                real_width += cur_width
                            elif idx>=1:
                                # space = sentence['bbox'][0] - line['sentences'][idx-1]['bbox'][2]
                                cur_width = self.get_text_size(text, font_size)[0]
                                #mark
                                space = self.trans_to_page_size(sentence['bbox'][0]-colum_bbox[0], cal_w=True) - real_width
                                if len(doc_colums)>1:
                                    space = self.trans_to_page_size(sentence['bbox'][0] - line['sentences'][idx-1]['bbox'][2], cal_w=True)
                                max_space = int((colum_end-real_width-cur_width)/(self.size_map[' ']*font_size/10))
                                num_space = int(space/(self.size_map[' ']*font_size/10))
                                if len(doc_colums)==1:
                                    num_space = min(max_space, num_space)
                                num_space = max(2, num_space)
                                if '`' in line['sentences'][idx-1]['text'] and ('sqrt' in line['sentences'][idx-1]['text'] or '/' in line['sentences'][idx-1]['text']):
                                    num_space += 4
                                real_width += self.get_text_size(' '*num_space, font_size)[0]
                                real_width += cur_width
                                line_text += num_space*' '*factor
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
            has_multi_line_para = False
            for pid, paragraph in enumerate(doc_colum['paragraphs']):
                font_size_cm = pt2cm(paragraph['font_size'])
                paragraph['align'] = 'both'
                right_indent = paragraph['right_indent']
                cur_para_end = colum_end - right_indent
                if len(paragraph['lines'])==1:
                    start = paragraph['bbox'][0]
                    end = paragraph['bbox'][2]
                    mid = (start + end)/2
                    # if abs(mid-colum_bbox_mid)<=colum_bbox_range*0.05 and (start-colum_bbox[0])>=colum_bbox_range*0.15 \
                    #         and (colum_end-end)<=0.8*colum_bbox_range and len(paragraph['lines'][0]['line_text'])<=8:
                    if not has_multi_line_para and pid<=3 and abs(mid - colum_bbox_mid) <= colum_bbox_range * 0.1 and paragraph['before_has_topic']==False and not paragraph['lines'][0]['in_topic'] and (start-colum_bbox[0])>=colum_bbox_range*0.05:# and (start-colum_bbox[0])/(colum_bbox[2]-end)>0.9 and (start-colum_bbox[0])/(colum_bbox[2]-end)<1.1:
                        paragraph['align'] = 'center'
                        continue
                    if pid == len(doc_colum['paragraphs']) - 1 and len(paragraph['lines']) == 1 and len(paragraph['lines'][0]['line_text']) <= 10 and '页' in paragraph['lines'][0]['line_text'] and (start-colum_bbox[0])>=colum_bbox_range*0.15:
                        paragraph['cast'] = True
                        continue

                else:
                    has_multi_line_para = True
                    line_ends = []
                    for line in paragraph['lines']:
                        line_end = self.trans_to_page_size(line['bbox'][2], cal_w=True)
                        line_ends.append(line_end)
                    line_ends.sort()
                    if (len(line_ends)>2 and line_ends[-1] - line_ends[1]>0.2) or (len(line_ends)==2 and line_ends[-1] - line_ends[0]>0.2) :
                        paragraph['align'] = 'left'
                w_ratio = 100
                w_spaces = []
                for line in paragraph['lines']:
                    line_end = self.trans_to_page_size(line['bbox'][2], cal_w=True)
                    line_start = self.trans_to_page_size(line['bbox'][0], cal_w=True)
                    colum_start = line_start
                    line['w_ratio'], line['w_space'] = self.cal_line_font_space(line, [colum_start, line_end],0.1)
                    if len(line['sentences'])>1:
                        continue
                    w_spaces.append(line['w_space'])
                    line['w_ratio'] = 100
                if w_spaces!=[]:
                    for line in paragraph['lines']:
                        line['w_space'] = min(1, max(0, min(w_spaces)))
            new_paragraphs = []
            for ppid, paragraph in enumerate(doc_colum['paragraphs']):
                if ppid==len(doc_colum['paragraphs'])-1 and  len(paragraph['lines'])==1  and is_all_number(paragraph['lines'][0]['line_text']) and len(paragraph['lines'][0]['line_text'])<=3:
                    continue
                if 'cast' in paragraph.keys():
                    continue
                if not self.is_english_doc:
                    for line in paragraph['lines']:
                        if 'w_space' not in line.keys():
                            line['w_space'] = 1
                            line['w_ratio'] = 100
                        if line['w_space']==0:
                            line['w_space'] = 1
                        if is_english_string(line['line_text']) or len(line['sentences'])!=1 or '_' in line['line_text']:
                            line['w_space'] = 0
                new_paragraphs.append(paragraph)
            doc_colum['paragraphs'] = new_paragraphs
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def cal_line_font_space(self, line, base_range, margin_threshold=0.35):
        base_width = base_range[1] - base_range[0] - margin_threshold
        ratio = 100
        diff = 0
        underline_width = pt2cm(line['font_size']) * 0.5
        if len(line['sentences'])==1:
            line_text = line['line_text']
            if line_text=="":
                return 100,0
            left_indent = line['left_indent']
            #base_width -= left_indent
            real_width = self.get_text_size(line_text, line['font_size'])[0]
            num_add_count = 0
            if base_width>real_width and "_" in line_text:
                ncount = (base_width-real_width) // underline_width
                nhas_underline = 0
                math_flag = False
                for i, char_ in enumerate(line_text):
                    if char_=='`' and not math_flag:
                        math_flag = True
                    elif char_=='`' and  math_flag:
                        math_flag = False
                    if not math_flag and char_=='_' and (i==0 or line_text[i-1]!='_'):
                        nhas_underline += 1
                if nhas_underline!=0:
                    per_place_add_count = int(ncount//nhas_underline)
                    more_count = ncount - nhas_underline*per_place_add_count
                    new_line_text = ""
                    count = 0
                    math_flag = False
                    for i, char_ in enumerate(line_text):
                        if char_=='`' and not math_flag:
                            math_flag = True
                        elif char_=='`' and  math_flag:
                            math_flag = False
                        if not math_flag and char_=='_' and (i==0 or line_text[i-1]!='_'):
                            new_line_text += char_
                            new_line_text += per_place_add_count*'_'
                            num_add_count += per_place_add_count
                            if count<more_count:
                                new_line_text+='_'
                                num_add_count += 1
                                count += 1
                        else:
                            new_line_text += char_
                    line['line_text'] = new_line_text

            new_line_text2 = ""
            math_flag = False
            for i, char_ in enumerate(line['line_text']):
                if char_ == '`' and not math_flag:
                    math_flag = True
                elif char_ == '`' and math_flag:
                    math_flag = False
                if not math_flag and char_ == '_' and (i == 0 or line['line_text'][i - 1] != '_'):

                    j = i + 1
                    while j < len(line['line_text']) and line['line_text'][j] == '_':
                        j += 1
                    cur_count = j - i
                    if cur_count == 1:
                        continue
                    new_line_text2 += char_
                    if cur_count < 5:
                        new_line_text2 += (5 - cur_count) * '_'
                        num_add_count += 5 - cur_count
                else:
                    new_line_text2 += char_
            real_width = real_width + num_add_count * 0.21 #0.21是下划线的宽度
            line['line_text'] = new_line_text2
            line_text = new_line_text2
            diff = round(cm2pt((base_width - real_width) / len(line_text)), 2)
        return ratio, diff

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
        page_pad_space = 0.
        last_doc_colum_height = 0.
        for colum_id, doc_colum in enumerate(doc_colums):
            colum_bbox = doc_colum['bbox']
            paragraphs = doc_colum['paragraphs']
            doc_colum_height = 0
            colum_pad_space = 0.
            last_para_end_h = 0.
            for para_id, paragraph in enumerate(paragraphs):
                para_bbox = paragraph['bbox']
                font_size = paragraph['font_size']
                if font_size<13:
                    line_pitch = 0.55
                elif font_size>=13:
                    line_pitch = 1.1
                elif font_size>=25:
                    line_pitch = 1.65
                if font_size==12:
                    line_pitch = 0.55*1.25
                paragraph['page_height_start'] = doc_colum_height
                a = paragraph['lines'][0]['line_text']
                cur_para_h = self.trans_to_page_size(para_bbox[1], cal_h=True)
                cur_para_end_h = self.trans_to_page_size(para_bbox[3], cal_h=True)
                if len(paragraph['lines'])==1:
                    line_text = paragraph['lines'][0]['line_text']
                    if len(line_text)>=3 and line_text[:2] in ['一、', '二、', '三、', '四、', '五、','六、','七、','八、','九、','十、']:
                        paragraph['black'] = 1
                    else:
                        paragraph['black'] = 0
                    k = 1.2
                    if self.is_english_doc:
                        k = 1.1
                    if '_' in paragraph['lines'][0]['line_text']:
                        k = 1.3
                    if '`' in paragraph['lines'][0]['line_text'] and '/' in paragraph['lines'][0]['line_text']:
                        k = 1.5
                    line_space = round(k*pt2cm(font_size), 2)
                else:
                    #line_space = self.cal_line_space(paragraph['lines'])
                    line_space = (cur_para_end_h - cur_para_h - pt2cm(font_size))/(len(paragraph['lines'])-1)
                    line_space = max(1.2*pt2cm(font_size), line_space)
                    paragraph['black'] = 0
                line_space = line_space
                up_space = 0.5*(line_space-pt2cm(font_size))
                cur_para_h -= up_space
                nline = len(paragraph['lines'])
                paragraph['line_space_rule'] = 'onehalf'
                paragraph['line_space'] = line_pitch
                paragraph['para_space_before'] = max(0, cur_para_h - last_para_end_h - 0.1)
                if paragraph['para_space_before']<0.1:
                    paragraph['para_space_before'] = 0
                doc_colum_height = doc_colum_height + paragraph['line_space']*len(paragraph['lines']) + paragraph['para_space_before']
                paragraph['page_height_end'] = doc_colum_height
                int_pad_line = 0
                if para_id!=len(paragraphs)-1:
                    next_para_h = self.trans_to_page_size(paragraphs[para_id+1]['bbox'][1], cal_h=True)
                    # hcm = self.trans_to_page_size(paragraphs[para_id+1]['bbox'][1]-paragraphs[para_id]['bbox'][3], colum_bbox[3]-colum_bbox[1], self.valid_page_height)
                    #hcm = self.trans_to_page_size(paragraphs[para_id+1]['bbox'][1]-paragraphs[para_id]['bbox'][3])
                    #pad_line = hcm / pt2cm(font_size)
                    pad_line = (next_para_h - cur_para_end_h) / pt2cm(font_size)
                    if pad_line >=2:
                        int_pad_line = int(pad_line)#-1
                doc_colum_height = doc_colum_height + pt2cm(font_size)*int_pad_line
                last_para_end_h = cur_para_end_h + pt2cm(font_size)*int_pad_line
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
        return mathml, 'omml' + omml_string

    def generate_formula(self, meta_data):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        for colum_id, doc_colum in enumerate(doc_colums):
            paragraphs = doc_colum['paragraphs']
            for para_id, paragraph in enumerate(paragraphs):
                for line in paragraph['lines']:
                    font_size = line['font_size']
                    line_text = line['line_text']
                    print(line_text)
                    latex_dict = ["^{", "_{", "\\frac", "\\sqrt"]
                    islatex = False
                    for latex in latex_dict:
                        if latex in line_text:
                            islatex = True
                            break
                    if not islatex:
                        line['line_text'] = [{"text":line_text}]
                        continue
                    else:
                        mathml_string, omml_string = self.latex_to_omml(line_text)
                        omml_string = add_font_size(omml_string, font_size)
                        line['line_text'] = [{"text":omml_string}]
                        line['latex_string'] = line_text
                        line['mathml_text'] = [mathml_string]

                    # if '`' not in line_text and '¤' not in line_text:
                    #     line['line_text'] = [{"text":line_text}]
                    # else:
                    #     line_text = math_formula_parse(line_text)
                    #     line_text = merge_latex_formula(line_text)
                    #     latex_string = line_text
                    #     omml_line_text = []
                    #     mathml_line_text = []
                    #     for uu in line_text:
                    #         if 'latex' in uu:
                    #             uu = uu.replace('latex', '')
                    #             if 'sqrt' in uu:
                    #                 uu = uu.replace('（','(').replace('）',')')
                    #             try:
                    #                 mathml_string, omml_string = self.latex_to_omml(uu)
                    #             except:
                    #                 omml_line_text.append(uu)
                    #                 mathml_line_text.append(uu)
                    #                 continue
                    #             omml_string = add_font_size(omml_string, font_size)
                    #             mathml_line_text.append(mathml_string)
                    #             omml_line_text.append(omml_string)
                    #         else:
                    #             omml_line_text.append(uu)
                    #             mathml_line_text.append(uu)
                    #     line_text = omml_line_text
                    #     new_line_text = []
                    #     for uu in line_text:
                    #         new_line_text.append({"text":uu})
                    #     line['line_text'] = new_line_text
                    #     line['latex_string'] = latex_string
                    #     line['mathml_text'] = mathml_line_text
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data

    def generate_pictures(self, meta_data):
        doc_pics = meta_data['doc_info']['doc_pics']
        doc_page_info = meta_data['doc_info']['doc_page_info']
        pad_w = doc_page_info['pad_w']
        image_info = meta_data['image_info']
        image = image_info['img']
        index = 1
        new_doc_pics = []
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']

        for doc_pic in doc_pics:
            bbox = doc_pic['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            real_x1 = max(0, int(x1 + image_info['img_left_margin'] - pad_w))
            real_x2 = int(x2 + image_info['img_left_margin'] - pad_w)
            real_y1 = max(0, int(y1 + image_info['img_top_margin']))
            real_y2 = int(y2 + image_info['img_top_margin'])
            if real_y1>=real_y2 or real_x1>=real_x2 or real_x2==0 or real_y2==0:
                continue
            crop_img = image[real_y1:real_y2, real_x1:real_x2]
            # url = os.path.join(self.work_dir, 'tmp', self.prefix+'_pic'+'%03d.jpg'%index)
            url = doc_pic['url']
            try:
                cv2.imwrite(url, crop_img)
            except:
                continue
            page_x1 = self.trans_to_page_size(x1, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y1 = self.trans_to_page_size(y1, cal_h=True) + doc_page_info['page_top_bottom_margin']
            page_x2 = self.trans_to_page_size(x2, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y2 = self.trans_to_page_size(y2, cal_h=True) + doc_page_info['page_top_bottom_margin']
            wcm = page_x2 - page_x1
            hcm = page_y2 - page_y1
            img_wh_ratio = (real_x2-real_x1)/(real_y2-real_y1)
            page_wh_ratio = wcm/hcm
            if len(doc_colums)>1:
                if page_wh_ratio > img_wh_ratio:
                    wcm = hcm * img_wh_ratio
                else:
                    wcm = hcm * img_wh_ratio
            else:
                if page_wh_ratio > img_wh_ratio:
                    wcm = hcm * img_wh_ratio
                else:
                    hcm = wcm / img_wh_ratio
            # warp = 'warpboth'
            warp = 'none'
            if len(doc_colums) == 1:
                hiou = 0.
                for p in doc_colums[0]['paragraphs']:
                    pbbox = p['bbox']
                    if y1 > pbbox[3]:
                        continue
                    if pbbox[1] > y2:
                        break
                    pby1 = pbbox[1]
                    pby2 = pbbox[3]
                    ciou = iou_1d([pby1, pby2], [y1, y2])
                    hiou += ciou
                    if hiou > 0.15 and x1 > pbbox[2]:
                        warp = 'warpleft'
                        break
            doc_pic['url'] = url
            doc_pic['warp'] = warp
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
        new_doc_textboxs = []
        for textbox in doc_textboxs:
            bbox = textbox['bbox']
            x1, y1, x2, y2 = bbox
            page_x1 = self.trans_to_page_size(x1, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y1 = self.trans_to_page_size(y1, cal_h=True) + doc_page_info['page_top_bottom_margin']
            page_x2 = self.trans_to_page_size(x2, cal_w=True) + doc_page_info['page_left_right_margin']
            page_y2 = self.trans_to_page_size(y2, cal_h=True) + doc_page_info['page_top_bottom_margin']
            text = textbox['text']
            if text == '':
                continue
            if len(text)==1 and text[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                continue
            if '_' in text:
                text = self.get_text_width_underline(img, textbox, global_font_size, img_left_margin, img_top_margin)
            textbox['text'] = text.replace('()', '(  )').replace('（）', '（  ）').replace('`','')
            cur_w, cur_h = self.get_text_size(text, 10)
            real_w, real_h = self.trans_to_page_size(x2-x1, cal_w=True), self.trans_to_page_size(y2-y1, cal_h=True)
            font_size = int(min(real_h / cur_h, real_w / cur_w) * 10)
            line_space = 1.5*font_size
            page_y1 = page_y1-pt2cm(line_space-font_size)*0.5
            textbox['page_bbox'] = [page_x1, page_y1, page_x2, page_y2]
            #textbox['font_size'] = font_size
            textbox['font_size'] = 12
            new_doc_textboxs.append(textbox)
        meta_data['doc_info']['doc_textboxs'] = new_doc_textboxs
        return meta_data

    def generate_tables(self, meta_data):
        tables = meta_data['doc_info']['doc_tables']
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        image_info = meta_data['image_info']
        doc_pics = meta_data['pics']
        doc_page_info = meta_data['doc_info']['doc_page_info']
        image = image_info['img']
        index = len(meta_data['doc_info']['doc_pics']) + 1
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
            wcm = page_x2 - page_x1
            hcm = page_y2 - page_y1
            img_wh_ratio = (x2 - x1) / (y2 - y1)
            page_wh_ratio = wcm / hcm
            if len(doc_colums) > 1:
                if page_wh_ratio > img_wh_ratio:
                    wcm = hcm * img_wh_ratio
                else:
                    wcm = hcm * img_wh_ratio
            else:
                if page_wh_ratio > img_wh_ratio:
                    wcm = hcm * img_wh_ratio
                else:
                    hcm = wcm / img_wh_ratio
            page_y2 = hcm + page_y1
            page_x2 = wcm + page_x1
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
                texts = []
                pics = []
                for ct in cell['texts']:
                    if ct['cls']==3:
                        pics.append(ct)
                    else:
                        texts.append(ct)

                lines, _ = self.merge_sentences(texts)
                lines = sorted(lines, key=lambda x:x['bbox'][1])
                paragraphs = []
                last_diff = -1
                target_diff = -1
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
                    line_font_size = self.calculate_line_font_size(new_line, is_table=True)
                    if line_font_size>12 or line_font_size==11:
                        line_font_size = 12
                    new_line['font_size'] = line_font_size
                    diff = max(0., cur_x1-page_x1)
                    if last_diff != -1 and abs(diff - last_diff) <= 0.2:
                        line['left_indent'] = max(0, target_diff)
                        last_diff = diff
                    else:
                        line['left_indent'] = diff
                        target_diff = diff
                        last_diff = diff
                    new_line['left_indent'] = max(0.1, diff-0.2)
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
                cell['pics'] = pics
            table_font_size = 8
            if all_font_list!=[]:
               table_font_size = max(8, get_mid_num(all_font_list))#-1
            for cell_id, cell in enumerate(cells):
                cell['max_cell_text_length'] = 0
                paragraphs = cell['paragraphs']
                if paragraphs==[]:
                    continue
                max_cell_text_length = 0.
                for paragraph in paragraphs:
                    paragraph['font_size'] = table_font_size
                    for line in  paragraph['lines']:
                        line_text = ''
                        for idx, sentence in enumerate(line['sentences']):
                            text = sentence['text']
                            if '_' in text and '`' not in text and not is_chemical_str(text):
                                text = text.replace('_', '____')
                            if idx == 0:
                                line_text += text
                            elif idx >= 1:
                                space = self.trans_to_page_size(sentence['bbox'][0] - line['sentences'][idx-1]['bbox'][2], cal_w=True)
                                num_space = int(space / self.size_map[' '])
                                line_text += num_space * ' '
                                line_text += text
                        #line_text = line_text.replace('\\sqrt', '¤')
                        #计算cell中文本的最长距离
                        text_width, _ = self.get_text_size(line_text, table_font_size)
                        max_cell_text_length = max(max_cell_text_length, text_width+line['left_indent'])
                        if '`' not in line_text and '¤' not in line_text:
                            line_text = line_text.replace('`', '')
                            line['line_text'] = [{"text":line_text}]
                        elif '`' in line_text and '/' in line_text:
                            line_text = line_text.replace('`', '')
                            line['line_text'] = [{"text":line_text}]
                        else:
                            line_text = math_formula_parse(line_text)
                            line_text = merge_latex_formula(line_text)

                            latex_string = line_text
                            omml_line_text = []
                            mathml_line_text = []
                            for uu in line_text:
                                if 'latex' in uu:
                                    uu = uu.replace('latex', '')
                                    try:
                                        mathml_string, omml_string = self.latex_to_omml(uu)
                                    except:
                                        omml_line_text.append(uu)
                                        mathml_line_text.append(uu)
                                        continue
                                    omml_string = add_font_size(omml_string, font_size)
                                    mathml_line_text.append(mathml_string)
                                    omml_line_text.append(omml_string)
                                else:
                                    omml_line_text.append(uu)
                                    mathml_line_text.append(uu)
                            line_text = omml_line_text
                            new_line_text = []
                            for uu in line_text:
                                new_line_text.append({"text": uu})
                            line['line_text'] = new_line_text
                            line['latex_string'] = latex_string
                            line['mathml_text'] = mathml_line_text
                cell['max_cell_text_length'] = max_cell_text_length/(cell['endcol'] - cell['startcol'] + 1)
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
                    paragraph['cell_follow_pics'] = []
                    paragraph['cell_before_pics'] = []
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
            col_text_widths = []
            for i in range(table['col_num']):
                col_w = 0.
                max_cell_text_length = 0.
                for j in range(table['row_num']):
                    cell_idx = cell_index[j][i]
                    cell_w = cells[cell_idx]['page_split_width']
                    col_w = max(col_w, cell_w)
                    max_cell_text_length = max(max_cell_text_length, cells[cell_idx]['max_cell_text_length'])
                if col_w==0:
                    col_w = table['avg_row_height']
                col_widths.append(col_w)
                col_text_widths.append(max_cell_text_length)
            #根据table的尺寸计算cell实际的宽高,
            real_col_widths = []
            real_row_heights = []
            for col_width in col_widths:
                real_col_widths.append(col_width/sum(col_widths)*(table['page_bbox'][2]-table['page_bbox'][0]))
            for row_height in row_heights:
                real_row_heights.append(row_height/sum(row_heights)*(table['page_bbox'][3]-table['page_bbox'][1]))
            #对于存在文本长度大于cell宽度的情况，对table整体宽度进行补偿
            pad_width = 0.
            for col_id, col_width in enumerate(real_col_widths):
                if col_text_widths[col_id]+0.1>col_width:
                    pad_width += (col_text_widths[col_id]+0.1 - col_width)
                    real_col_widths[col_id] += (col_text_widths[col_id]+0.1 - col_width)
            table['page_bbox'][2] = table['page_bbox'][2] + pad_width

            #若补偿长度后超出页面宽度，则选择将字体大小-1
            if table['page_bbox'][2]-table['page_bbox'][0] > self.valid_page_width:
                table['page_bbox'][2] -= pad_width
                for cell in table['cells']:
                    for paragraph in cell['paragraphs']:
                        paragraph['font_size'] -= 1


            for cell in table['cells']:
                cell_bbox = cell['bbox']
                cell_bbox = move_bbox([cell_bbox[0],cell_bbox[1],cell_bbox[4],cell_bbox[5]], image_info['img_left_margin'], image_info['img_top_margin'])
                pics = cell['pics']
                cell['cell_pics'] = []
                if len(pics)==0:
                    continue
                pics = sorted(pics, key=lambda x:x['region'][0])
                for pic_id, pic in enumerate(pics):
                    real_x1, real_y1, real_x2, real_y2 = pic['region'][:4]
                    if real_y1 >= real_y2 or real_x1 >= real_x2 or real_x2 == 0 or real_y2 == 0:
                        continue
                    crop_img = image[int(real_y1):int(real_y2), int(real_x1):int(real_x2)]
                    # url = os.path.join(self.work_dir, 'tmp', self.prefix + '_pic' + '%03d.jpg' % index)
                    url = pic['url']
                    cv2.imwrite(url, crop_img)
                    index += 1
                    pic_bbox = move_bbox(pic['region'], image_info['img_left_margin'], image_info['img_top_margin'])
                    pic_page_width = self.trans_size(pic_bbox[2] - pic_bbox[0], table['bbox'][2] - table['bbox'][0],
                                                     table['page_bbox'][2] - table['page_bbox'][0])
                    pic_page_height = self.trans_size(pic_bbox[3] - pic_bbox[1], pic_bbox[2] - pic_bbox[0],
                                                      pic_page_width)
                    doc_pic = {'bbox': pic_bbox, 'type':'pic', 'text':'', 'url':url, 'warp':'none', \
                               'page_bbox':[0,0,pic_page_width,pic_page_height],'position_h_from_paragraph':0,'position_w_from_colum':0}
                    flag = False
                    for para_id, para in enumerate(cell['paragraphs']):
                        pbbox = para['bbox']
                        pbbox = move_bbox(pbbox, image_info['img_left_margin'], image_info['img_top_margin'])
                        # rule = pbbox[3] <= pic_bbox[1] + 20 and (idx == len(cell['paragraphs']) - 1 or cell['paragraphs'][idx + 1]['bbox'][3] >pic_bbox[1] + 10)
                        #rule =  pic_bbox[3] - 10 >=pbbox[1]  and (idx == len(cell['paragraphs']) - 1 or pic_bbox[3] - 10<cell['paragraphs'][idx + 1]['bbox'][1] )
                        rule =  pic_bbox[1] + 20 >=pbbox[3]  and (para_id == len(cell['paragraphs']) - 1 or pic_bbox[1] + 20 <cell['paragraphs'][para_id + 1]['bbox'][3] )
                        if rule:
                            dis_h = self.trans_size(pic_bbox[1] - pbbox[3], table['bbox'][2] - table['bbox'][0],table['page_bbox'][2] - table['page_bbox'][0])
                            dis_w = max(0, self.trans_size(pic_bbox[0] - cell_bbox[0], table['bbox'][2] - table['bbox'][0],table['page_bbox'][2] - table['page_bbox'][0]))
                            doc_pic['position_h_from_paragraph'] = dis_h
                            doc_pic['position_w_from_colum'] = dis_w
                            para['cell_follow_pics'].append(pic_id)
                            flag = True
                            break
                    cell['cell_pics'].append(doc_pic)
                    if not flag:
                        if len(cell['paragraphs'])!=0:
                            cell['paragraphs'][0]['cell_before_pics'].append(pic_id)
                del cell["pics"]
            table['row_heights'] = real_row_heights
            table['col_widths'] = real_col_widths
            cell_index_dict = []
            for i in range(len(cell_index)):
                row = {'row':cell_index[i]}
                cell_index_dict.append(row)
            table['cell_index'] = cell_index_dict
        meta_data['doc_info']['doc_tables'] = tables
        return meta_data

    def set_index(self, meta_data, overlap_pix_threshold=10):
        doc_info = meta_data['doc_info']
        page_top_bottom_margin = doc_info['doc_page_info']['page_top_bottom_margin']
        page_left_right_margin = doc_info['doc_page_info']['page_left_right_margin']
        doc_colums = doc_info['doc_sections'][0]['doc_colums']
        if len(doc_colums)>1:
            return meta_data
        doc_text_boxs = doc_info['doc_textboxs']
        doc_pics = doc_info['doc_pics']
        doc_tables = doc_info['doc_tables']
        index = 1
        for id, doc_colum in enumerate(doc_colums):
            place_para = {'type':'paragraph', 'lines':[], 'bbox':[doc_colum['bbox'][0]-1, -1, doc_colum['bbox'][2], 0], 'font_size':4, 'align':'both', 'is_place_para':1, \
                          'line_space':0.04, 'line_space_rule':"exact", "para_space_before":0, "page_height_start":0, "page_height_end":0, "right_indent":0., "n_pad_line":0}
            doc_colum['paragraphs'].insert(0, place_para)
            for para in doc_colum['paragraphs']:
                para['index'] = index
                para['float_content_id'] = []
                index += 1

        other_content = doc_text_boxs+doc_pics+doc_tables
        other_content = sorted(other_content, key=lambda x:x['bbox'][1])
        for index, content in enumerate(other_content):
            pic_bbox = content['bbox']
            content['index'] = index
            px1, py1, px2, py2 = pic_bbox

            if len(doc_colums)==0:
                content['in_page'] = 1
                continue
            cid = content['colum_id']
            doc_colum = doc_colums[cid]
            colum_bbox = doc_colum['bbox']
            colum_page_range = doc_colum['page_range']
            colum_width_start = doc_colum['page_range'][0]
            f = False
            # overlap_pix_threshold = 0.2*(py2 - py1)
            overlap_pix_threshold = 0.2*(py2 - py1)
            for idx, para in enumerate(doc_colum['paragraphs']):
                pbbox = para['bbox']
                rule = pbbox[3]<=pic_bbox[1]+overlap_pix_threshold and (idx==len(doc_colum['paragraphs'])-1 or doc_colum['paragraphs'][idx+1]['bbox'][3]>pic_bbox[1]+overlap_pix_threshold)
                if rule:
                    para['float_content_id'].append(index)
                    dis_h = self.trans_to_page_size(pic_bbox[1]-pbbox[3], cal_h=True)
                    dis_w = max(0, self.trans_size(pic_bbox[0]-colum_bbox[0], colum_bbox[2]-colum_bbox[0],colum_page_range[2]-colum_page_range[0] ))
                    if dis_w+content['page_bbox'][2] - content['page_bbox'][0]>colum_page_range[2]:
                        dis_w = colum_page_range[2] - (content['page_bbox'][2] - content['page_bbox'][0]) + 0.2
                    if idx!=len(doc_colum['paragraphs'])-1 and doc_colum['paragraphs'][idx+1]['bbox'][1]<pic_bbox[1]:
                    # if idx!=len(doc_colum['paragraphs'])-1 and doc_colum['paragraphs'][idx+1]['bbox'][3]<pic_bbox[3]:
                        map_h = self.trans_to_page_size(pic_bbox[1]-doc_colum['paragraphs'][idx+1]['bbox'][1], cal_h=True)
                        real_h = (pic_bbox[1] - doc_colum['paragraphs'][idx + 1]['bbox'][1] + overlap_pix_threshold)/(doc_colum['paragraphs'][idx + 1]['bbox'][3]-doc_colum['paragraphs'][idx + 1]['bbox'][1])*\
                                 (doc_colum['paragraphs'][idx + 1]['page_height_end'] - doc_colum['paragraphs'][idx + 1]['page_height_start'] - doc_colum['paragraphs'][idx + 1]['para_space_before'])
                        dis_h += max(0, real_h-map_h)
                        dis_h += 0.1
                        dis_w += 0.5
                        content['warp'] = 'warpboth'
                        if iou_1d([pic_bbox[0], pic_bbox[2]], [doc_colum['paragraphs'][idx + 1]['bbox'][0],doc_colum['paragraphs'][idx + 1]['bbox'][2]])>0.3:
                            min_dis_h = doc_colum['paragraphs'][idx + 1]['page_height_end'] - doc_colum['paragraphs'][idx + 1]['page_height_start'] + 0.1
                            dis_h = max(min_dis_h, dis_h)
                            #doc_colum['paragraphs'][idx+1]['n_pad_line'] += 2
                    if 'warp' in content.keys() and content['warp'] == 'warpleft':
                        dis_w += 1
                    if 'warp' in content.keys() and content['warp'] == 'none':
                        dis_h = max(0, dis_h)
                    content['position_h_from_paragraph'] = dis_h
                    content['position_w_from_colum'] = dis_w
                    content['in_page'] = 0
                    f = True
                    break
            if not f:
                content['in_page'] = 1
        for id, doc_colum in enumerate(doc_colums):
            last_para = doc_colum['paragraphs'][-1]
            if last_para['float_content_id']!=[]:
                y1_list = []
                y2_list = []
                for content_id in last_para['float_content_id']:
                    content = other_content[content_id]
                    y1_list.append(content['page_bbox'][1])
                    y2_list.append(content['page_bbox'][3])
                pad_line = (max(y2_list) - min(y1_list))/ 0.42
                if pad_line >= 1:
                    int_pad_line = int(pad_line)
                    last_para['n_pad_line'] = int_pad_line
        return meta_data

    def process_multi_doc_colums(self, meta_data, overlap_pix_threshold=20):
        doc_colums = meta_data['doc_info']['doc_sections'][0]['doc_colums']
        if len(doc_colums)<=1:
            return meta_data
        for cid, doc_colum in enumerate(doc_colums):
            left_indets = []
            for para in doc_colum['paragraphs']:
                for line in para['lines']:
                    left_indets.append(line['left_indent'])
            for para in doc_colum['paragraphs']:
                for line in para['lines']:
                    line['left_indent'] -= min(left_indets)

        for doc_colum in doc_colums:
            float_contents = doc_colum['float_contents']
            paragraphs = doc_colum['paragraphs']
            contents = float_contents + paragraphs
            contents = sorted(contents, key=lambda x:x['bbox'][1])
            all_blocks = []
            content_id = 0
            while  content_id < len(contents):
                content = contents[content_id]
                bbox = content['bbox']
                type_ = content['type']
                if 'is_place_para' in content.keys() or content_id == len(contents) - 1 or bbox[3] < contents[content_id + 1]['bbox'][1] + overlap_pix_threshold or (type_ == 'paragraph' and contents[content_id + 1]['type'] == 'paragraph'):
                    content['warp'] = 'inline'
                    content['create_para'] = 1
                    content['keep_para'] = 1
                    if type_!='paragraph':
                        content['position_h_from_paragraph'] = 0
                        content['position_w_from_colum'] = 0
                        mid_w = (bbox[0] + bbox[2]) / 2
                        mid_colum_w = (doc_colum['bbox'][2]+doc_colum['bbox'][0])/2
                        # if (bbox[2]-bbox[0])/(doc_colum['bbox'][2]-doc_colum['bbox'][0])>0.7:
                        #     content['position_w_from_colum'] = 5
                        # elif (bbox[0]-doc_colum['bbox'][0]) / (doc_colum['bbox'][2]-doc_colum['bbox'][0])<=0.25:
                        #     content['position_w_from_colum'] = 2.5
                        # elif (bbox[0]-doc_colum['bbox'][0]) / (doc_colum['bbox'][2]-doc_colum['bbox'][0])>=0.75:
                        #     content['position_w_from_colum'] = 7.5
                        # else:
                        #     content['position_w_from_colum'] = 5

                        if abs((mid_w-mid_colum_w)/(doc_colum['bbox'][2]-doc_colum['bbox'][0]))<0.15:
                            content['position_w_from_colum'] = 5
                        elif (mid_w-mid_colum_w)/(doc_colum['bbox'][2]-doc_colum['bbox'][0])>=0.15:
                            content['position_w_from_colum'] = 7.5
                        else:
                            content['position_w_from_colum'] = 2.5
                        content['in_page'] = 0
                        if len(all_blocks)>0 and all_blocks[-1]['type']=='paragraph' and all_blocks[-1]['n_pad_line']!=0:
                            all_blocks[-1]['n_pad_line'] -= 1
                    all_blocks.append(content)
                    content_id += 1
                else:
                    next_content = contents[content_id + 1]
                    div_h1 = min(content['bbox'][1], next_content['bbox'][1])
                    div_h2 = max(content['bbox'][3], next_content['bbox'][3])
                    divs = [content, next_content]
                    content_id = content_id + 2
                    while content_id < len(contents) and contents[content_id]['bbox'][1] + overlap_pix_threshold < div_h2:
                        content = contents[content_id]
                        divs = sorted(divs, key=lambda x: x['bbox'][0])
                        div_h2 = max(div_h2, content['bbox'][3])
                        divs.append(content)
                        content_id += 1
                    paras = []
                    fcs = []
                    for div in divs:
                        if div['type']=='paragraph':
                            div['float_contents'] = []
                            paras.append(div)
                        else:
                            fcs.append(div)
                    paras = sorted(paras, key=lambda x:x['bbox'][1])
                    p_range_x1s = []
                    p_range_x2s = []
                    p_range_y1s = []
                    p_range_y2s = []
                    for uu, para in enumerate(paras):
                        para['create_para'] = 1
                        p_range_x1s.append(para['bbox'][0])
                        p_range_x2s.append(para['bbox'][2])
                        p_range_y1s.append(para['bbox'][1])
                        p_range_y2s.append(para['bbox'][3])
                        para['warp'] = 'warp'
                        if uu==0:
                            para['keep_para'] = 1
                        else:
                            para['keep_para'] = 1
                    fcs = sorted(fcs, key=lambda x:x['bbox'][0])
                    inline_pic = False
                    for uu, fc in enumerate(fcs):
                        fc['keep_para'] = 0
                        fc['create_para'] = 0
                        if len(paras)!=0:
                            fc['page_bbox'][2] = fc['page_bbox'][0] + (fc['page_bbox'][2] - fc['page_bbox'][0])  # *0.8
                            fc['page_bbox'][3] = fc['page_bbox'][1] + (fc['page_bbox'][3] - fc['page_bbox'][1])  # *0.8
                            p_range_x1 = min(p_range_x1s)
                            p_range_x2 = max(p_range_x2s)
                            right_region = [fc['bbox'][2], fc['bbox'][1], p_range_x2, fc['bbox'][3]]
                            left_region = [p_range_x1, fc['bbox'][1], fc['bbox'][0], fc['bbox'][3]]
                            left_overlap = False
                            right_overlap = False
                            for para in paras:
                                for line in para['lines']:
                                    if ((right_region[2]-right_region[0])*(right_region[3]-right_region[1]))>0 and bbox_iou(line['bbox'], right_region) != 0:
                                        right_overlap = True
                                    if ((left_region[2]-left_region[0])*(left_region[3]-left_region[1]))>0 and bbox_iou(line['bbox'], left_region) != 0:
                                        left_overlap = True
                            if left_overlap and not right_overlap:
                                fc['warp'] = 'warpleft'
                            elif not left_overlap and right_overlap:
                                fc['warp'] = 'warpright'
                            else:
                                fc['warp'] = 'warpboth'
                        else:
                            fc['warp'] = 'inline'
                            inline_pic  = True
                            if uu==0:
                                fc['keep_para'] = 1
                                fc['create_para'] = 1
                        content['pad_text'] = 0
                        if uu!=len(fcs)-1:
                            cx2 = fc['bbox'][2]
                            nx1 = fcs[uu+1]['bbox'][1]
                            margin = self.trans_to_page_size(nx1-cx2, cal_w=True)
                            pad_text = max(0, int(margin/pt2cm(2*12)))
                            content['pad_text'] = pad_text
                    if inline_pic:
                         divs = paras + fcs
                         if len(all_blocks) > 0 and all_blocks[-1]['type'] == 'paragraph' and all_blocks[-1]['n_pad_line'] != 0:
                             all_blocks[-1]['n_pad_line'] -= 1
                         for fc in fcs:
                            fc['position_h_from_paragraph'] = 0
                            fc['position_w_from_colum'] = 0
                            fc['in_page'] = 0
                    else:
                        colum_bbox = doc_colum['bbox']
                        colum_page_range = doc_colum['page_range']
                        # p_range_y1 = min(p_range_y1s)
                        # for fc in fcs:
                        #     pic_bbox = fc['bbox']
                        #     dis_h = self.trans_to_page_size(pic_bbox[1] -p_range_y1, cal_h=True) + 0.1 + paras[0]['para_space_before']
                        #     dis_w = max(0,self.trans_size(pic_bbox[0] - colum_bbox[0], colum_bbox[2] - colum_bbox[0],
                        #                                 colum_page_range[2] - colum_page_range[0]))
                        #     if fc['warp']=='warpleft':
                        #         dis_w += 0.5
                        #     if dis_w + fc['page_bbox'][2] - fc['page_bbox'][0] > colum_page_range[2]:
                        #         dis_w = colum_page_range[2] - ( fc['page_bbox'][2] - fc['page_bbox'][0]) + 0.2
                        #     fc['position_h_from_paragraph'] = dis_h
                        #     fc['position_w_from_colum'] = dis_w
                        #     fc['in_page'] = 0
                        divs = sorted(divs, key=lambda x: x['bbox'][1])
                        for div_idx, div in enumerate(divs):
                            if div['type'] == 'paragraph':
                                continue
                            # 为当前图片找到跟随的段落
                            pid = div_idx - 1
                            while pid > 0 and divs[pid]['type'] != 'paragraph':
                                pid -= 1
                            if pid >= 0 and divs[pid]['type'] == 'paragraph':
                                ppbox = divs[pid]['bbox']
                                follow_para = divs[pid]
                            else:
                                pid = div_idx + 1
                                while pid < len(divs) - 1 and divs[pid]['type'] != 'paragraph':
                                    pid += 1
                                ppbox = divs[pid]['bbox']
                                follow_para = divs[pid]
                            # 计算图片和段落之间的相对位置
                            follow_para['float_contents'].append(div)
                            pic_bbox = div['bbox']
                            if div_idx < pid:
                                dis_h = self.trans_to_page_size(pic_bbox[1] - ppbox[1], cal_h=True) + 0.15 + follow_para['para_space_before']
                            else:
                                dis_h = self.trans_to_page_size(pic_bbox[1] - ppbox[1], cal_h=True) + 0.15 + follow_para['para_space_before'] #+ 1.25 * (len(follow_para['lines']) * 0.39)
                            div['pid'] = pid
                            dis_w = max(0, self.trans_size(pic_bbox[0] - colum_bbox[0], colum_bbox[2] - colum_bbox[0],
                                                           colum_page_range[2] - colum_page_range[0]))

                            if div['warp'] == 'warpleft' and len(follow_para['lines'])>=1:
                                #para_line_length = self.get_text_size(follow_para['lines'][0]['line_text'], 12)[0]
                                dis_w = max(0, self.trans_size(pic_bbox[2] - colum_bbox[0], colum_bbox[2] - colum_bbox[0],colum_page_range[2] - colum_page_range[0]) - (div['page_bbox'][2] - div['page_bbox'][0]) )
                                dis_w += 2
                            if dis_w + div['page_bbox'][2] - div['page_bbox'][0] > colum_page_range[2]:
                                dis_w = colum_page_range[2] - (div['page_bbox'][2] - div['page_bbox'][0]) + 0.2
                            div['position_h_from_paragraph'] = dis_h
                            div['position_w_from_colum'] = dis_w
                            div['in_page'] = 0
                        new_divs = []
                        for pid, div in enumerate(divs):
                            if div['type'] == 'paragraph':
                                new_divs.append(div)
                                cur_fcs = sorted(div['float_contents'], key=lambda x: x['bbox'][0])
                                for fc in cur_fcs:
                                    if fc['type'] == 'table' and fc['bbox'][3] < div['bbox'][1]:
                                        new_divs.insert(-1, fc)
                                    else:
                                        new_divs.append(fc)
                        divs = new_divs
                    for div in divs:
                        all_blocks.append(div)
            blocks = []
            for cid, content in enumerate(all_blocks):
                block = {'type': 'block', 'paragraphs': [], 'pics': [], 'tables': []}
                block['warp'] = content['warp']
                block['keep_para'] = 0
                block['create_para'] = 0
                if 'keep_para' in content.keys():
                    block['keep_para'] = content['keep_para']
                if 'create_para' in content.keys():
                    block['create_para'] = content['create_para']
                block['bbox'] = content['bbox']
                flag = False
                if content['type']=='paragraph':
                    content['right_indent'] = 0
                    if cid!=len(all_blocks)-1 and all_blocks[cid+1]['type']!='paragraph' and all_blocks[cid+1]['warp']=='inline' :
                        diff = self.trans_to_page_size(all_blocks[cid + 1]['bbox'][1] - content['bbox'][3], cal_h=True)
                        if diff<content['n_pad_line']*pt2cm(12)*0.8:
                            pid = cid + 0
                            py1s = []
                            py2s = []
                            while  pid!=len(all_blocks)-1 and all_blocks[pid+1]['type']!='paragraph' and all_blocks[pid+1]['warp']=='inline' and 'page_bbox' in all_blocks[pid+1].keys():
                                py1s.append(all_blocks[pid+1]['page_bbox'][1])
                                py2s.append(all_blocks[pid+1]['page_bbox'][3])
                                pid += 1
                            if len(py1s)>0:
                                inline_img_h = max(py2s) - min(py1s)
                                pad_line = int(inline_img_h / pt2cm(12))
                            else:
                                pad_line = 0
                            content['n_pad_line'] = max(0, content['n_pad_line'] - pad_line)
                            all_blocks[pid]['n_pad_line'] = content['n_pad_line']
                            content['n_pad_line'] = 0
                            flag = True
                    if cid!=0 and all_blocks[cid-1]['type']!='paragraph' and all_blocks[cid-1]['warp']=='inline':
                        content['para_space_before'] = 0.352
                if content['type']=='table' and cid!=0 and all_blocks[cid-1]['type']=='table':
                    place_para = {'type': 'paragraph', 'lines': [],
                                  'bbox': [doc_colum['bbox'][0] - 1, -1, doc_colum['bbox'][2], 0], 'font_size': 12,
                                  'align': 'both', \
                                  'line_space': 0.352, 'line_space_rule': "exact", "para_space_before": 0,
                                  "page_height_start": 0, "page_height_end": 0, "right_indent": 0., "n_pad_line": 0}
                    block['paragraphs'].append(place_para)

                if content['type']!='paragraph' and  'n_pad_line' in content.keys():# or not flag:
                    block['n_pad_line'] = content['n_pad_line']
                elif content['type']=='paragraph' and 'n_pad_line' in content.keys() and not flag:
                    block['n_pad_line'] = content['n_pad_line']
                else:
                    block['n_pad_line'] = 0
                block[content['type']+'s'].append(content)
                blocks.append(block)
            doc_colum['blocks'] = blocks
        meta_data['doc_info']['doc_sections'][0]['doc_colums'] = doc_colums
        return meta_data



if __name__ == "__main__":
    dc = doc_caculator()
    img_file = '20220831-153635.jpg'
    json_file = 'test1.json'
    dc.process(img_file=img_file, json_file=json_file, work_dir='/home/ateam/xychen/projects/image_2_word/word')