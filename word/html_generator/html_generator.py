from bs4 import BeautifulSoup
import json, pdb, os,cv2


import shutil
import numpy as np

def iou_1d(l1, l2, real_iou=False):
    range1 = l1[1] - l1[0]
    range2 = l2[1] - l2[0]
    xmin = max(l1[0], l2[0])
    xmax = min(l1[1], l2[1])
    if xmax<=xmin:
        return 0
    if real_iou:
        return (xmax-xmin) / (range1 + range2 - (xmax-xmin))
    else:
        return (xmax-xmin)/range2

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
    ratio = inter_area * 1. / cur_area
    return ratio

class html_generator:
    def __init__(self, template_html="template.html"):
        file = open(template_html, 'rb')
        self.template_html = file.read()

    def get_style(self, name, atts):
        atts_string = ''
        for key, value in atts.items():
            atts_string += "%s:%s;\n"%(key, value)
        style_string = '%s{\n%s}'%(name, atts_string)
        return style_string+'\n'
        
    def set_body_attributes(self, body_meta_data):
        page_width = body_meta_data['page_width']
        margin_top = body_meta_data['page_top_bottom_margin']
        margin_left_right = body_meta_data['page_left_right_margin']
        body_att = {
            "max-width":  "21cm",
            "margin-top":  "2.03cm",
            "margin-left": "1.3cm",
            "margin-right": "1.3cm",
            "height": "100%",
            "width": "100%",
            "text-align":"left",
        }
        body_style = self.get_style('html, body', body_att)
        self.bs.head.style.string += body_style
    
    def add_para_tag(self, name_id, para_meta_data):
        space_before = para_meta_data['space_before']
        line_space = para_meta_data['line_space']#*1.2
        left_indent = para_meta_data['left_indent']
        right_indent = para_meta_data['right_indent']
        first_line_indent = max(0, para_meta_data['first_line_indent'])
        align = para_meta_data['align']
        font_size = para_meta_data['font_size']
        if align == 'both':
            align = "justify"
        para_attrs = {"class":"pt-%06d"%name_id}
        para_style_attrs = {	
            "margin-top":str(space_before)+"cm",
            "line-height":str(line_space)+"cm",
            "margin-left":str(left_indent)+"cm",
            "text-indent":str(first_line_indent)+"cm",
            "text-align":align,
            "font-family":'宋体',
            "font-size":str(font_size)+"pt",
            # "font-size":"1em",
            "vertical-align":"baseline",
            "margin-right":str(right_indent)+"pt",
            "margin-bottom":".001pt",
            "-webkit-text-size-adjust": "none",
            "text-size-adjust": "none",
        }
        if "margin_left" in para_meta_data.keys():
            para_style_attrs['position'] = 'relative'
            para_style_attrs['left'] = str(int(para_meta_data['margin_left']))+'%'
        para_style = self.get_style("p."+"pt-%06d"%name_id, para_style_attrs)
        tag = self.bs.new_tag("p", attrs=para_attrs)
        self.bs.head.style.string += para_style
        name_id += 1
        return tag,name_id 

    def add_div_tag(self, name_id, block_range, color=None, float='left', align='center', absolute=False):
        left, top, width, height = block_range
        color = 'white' if color is None else color
        block_attrs = {"class":"pt-%06d"%name_id, "align":align}
        block_style_attrs = {
                "position":"relative",
                "left":str(left)+"%",
                "top":str(top)+"%",
                "-webkit-text-size-adjust":"none",
                "text-size-adjust":"none",

                "width":str(width)+"%",
                "float":str(float),
                "background-color":color
                }
        if height!=-1:
            block_style_attrs['height'] = str(height)+'%'
        if absolute:
            block_style_attrs['height'] = str(height) + 'cm'
        block_style = self.get_style("div."+"pt-%06d"%name_id, block_style_attrs)
        tag = self.bs.new_tag("div", attrs=block_attrs)
        self.bs.head.style.string += block_style
        name_id += 1
        return tag,name_id 

    def add_blank_para_tag(self, name_id):
        blank_div, name_id = self.add_div_tag(name_id, [0,0,100,5])
        return blank_div, name_id

    def add_span_tag(self, name_id, line_text, span_meta_data):
        if '_' in line_text:
            line_text = line_text.replace('__', '_')
        line_text = line_text.replace('  ', ' ')
        line_text = line_text.replace('（', '(').replace('）',')')
        font_size = span_meta_data['font_size']
        span_attrs = {"class":"pt-%06d"%name_id, "xml:space":"preserve"}
        span_style_attrs = {	
            "font-family": "宋体",
            "font-size":str(font_size)+"pt",
            "letter-spacing": "0",
            "font-style": "normal",
            "font-weight": "normal",
            "margin": "0",
            "padding": "0",
            "-webkit-text-size-adjust": "none",
            "text-size-adjust": "none",
        }
        # if "w_space" in span_meta_data.keys():
            # span_style_attrs['letter-spacing'] = str(span_meta_data['w_space'])+"pt"
        texts = []
        if '^' not in line_text:
            texts.append(line_text)
        else:
            sub_text = ''
            idx = 0
            while idx<len(line_text):
                char = line_text[idx]
                if char!='^':
                    sub_text += char
                else:
                    texts.append(sub_text)
                    sub_text = ''
                    if idx!=len(line_text)-1:
                        texts.append(char+line_text[idx+1])
                        idx += 1
                idx += 1
            if sub_text!='':
                texts.append(sub_text)
        tags = []
        for sub_text in texts:
            if '^' not in sub_text:
                span_style = self.get_style("span."+"pt-%06d"%name_id, span_style_attrs)
                tag = self.bs.new_tag("span", attrs=span_attrs)
                self.bs.head.style.string += span_style
                tag.string = sub_text
            else:
                tag = self.bs.new_tag("sup")
                tag.string = sub_text.replace('^', '')

            name_id += 1
            tags.append(tag)
        return tags,name_id

    def add_pic_tag(self, name_id, img_meta_data):
        url = img_meta_data['url']
        basename = url.split('/')[-1]
        shutil.copyfile(url, os.path.join(self.html_files_dir, basename))
        url =  os.path.join(self.url_prefix, basename)
        img_attrs = {"src":url, "class":"pt-%06d"%name_id}
        height = img_meta_data['page_bbox'][3] - img_meta_data['page_bbox'][1]
        width = img_meta_data['page_bbox'][2] - img_meta_data['page_bbox'][0]

        img_style_attrs = {'height':str(height)+'cm', 'width':str(width)+'cm'}#, 'vertical-align':'middle'}
        if "margin_left" in img_meta_data.keys():
            img_style_attrs['position'] = 'relative'
            img_style_attrs['left'] = str(img_meta_data["margin_left"])+"%"
        img_style = self.get_style("img." + "pt-%06d" % name_id, img_style_attrs)
        tag = self.bs.new_tag("img", attrs=img_attrs)
        self.bs.head.style.string += img_style
        name_id += 1
        return tag, name_id

    def add_table_row_tag(self, name_id, height):
        tr_attrs = {"class": "pt-%06d" % name_id}
        tr_style_attrs = {'height': str(height) + 'cm'}
        tr_style = self.get_style("tr." + "pt-%06d" % name_id, tr_style_attrs)
        tag = self.bs.new_tag("tr", attrs=tr_attrs)
        self.bs.head.style.string += tr_style
        name_id += 1
        return tag, name_id

    def add_cell_tag(self, name_id, width, rowspan=0, colspan=0):
        td_attrs = {"class": "pt-%06d" % name_id}
        td_style_attrs = {'width': str(width) + '%', "border-top": "solid windowtext 1.0pt","border-bottom": "solid windowtext 1.0pt",
                          "border-right": "solid windowtext 1.0pt","border-left": "solid windowtext 1.0pt"
                          }
        if rowspan!=0:
            td_attrs['rowspan'] = rowspan
        if colspan!=0:
            td_attrs['colspan'] = colspan
        td_style = self.get_style("td." + "pt-%06d" % name_id, td_style_attrs)
        tag = self.bs.new_tag("td", attrs=td_attrs)
        self.bs.head.style.string += td_style
        name_id += 1
        return tag, name_id

    def add_table_tag(self, name_id, width):
        table_attrs = {"class": "pt-%06d" % name_id}
        table_style_attrs = {'height': '90%', 'width':str(100*width)+'%',
                             "border-collapse": "collapse","border": "none"
        }
        table_style = self.get_style("table." + "pt-%06d" % name_id, table_style_attrs)
        tag = self.bs.new_tag("table", attrs=table_attrs)
        self.bs.head.style.string += table_style
        name_id += 1
        return tag, name_id

    def add_hr_tag(self):
        hr_attrs = {'style':'height:2px;background-color:black'}
        tag = self.bs.new_tag("hr", attrs=hr_attrs)
        return tag

    def add_para_in_cell_with_pics(self, td, index, paragraphs, cell_pics):
        for paragraph in paragraphs:
            for before_pic_index in paragraph['cell_before_pics']:
                before_pic = cell_pics[before_pic_index]
                if os.path.exists(before_pic['url']):
                    img_tag, index = self.add_pic_tag(index, before_pic)
                    td.append(img_tag)
            line_space = paragraph['line_space']
            space_before = paragraph['para_space_before']
            align = paragraph['align']
            font_size = paragraph['font_size']
            right_indent = paragraph['right_indent']
            lines = paragraph['lines']
            left_indent = 0.
            first_line_indent = 0
            if len(lines) >= 1 and align != 'center':
                first_line_indent = lines[0]['left_indent']
            if len(lines) > 1 and align != 'center':
                left_indent = lines[1]['left_indent']
                first_line_indent = lines[0]['left_indent'] - left_indent
            para_meta_data = {"line_space": line_space, "space_before": space_before, "align": align,
                              "font_size": font_size, "right_indent": right_indent, \
                              "first_line_indent": first_line_indent, "left_indent": left_indent}
            p_tag, index = self.process_paragraph(index, para_meta_data, lines=lines)
            td.append(p_tag)
            for after_pic_index in paragraph['cell_follow_pics']:
                after_pic = cell_pics[after_pic_index]
                if os.path.exists(after_pic['url']):
                    img_tag, index = self.add_pic_tag(index, after_pic)
                    td.append(img_tag)
        return td, index

    def add_para_in_cell(self, td, index, paragraphs):
        for paragraph in paragraphs:
            line_space = paragraph['line_space']
            space_before = paragraph['para_space_before']
            align = paragraph['align']
            font_size = paragraph['font_size']
            right_indent = paragraph['right_indent']
            lines = paragraph['lines']
            left_indent = 0.
            first_line_indent = 0
            if len(lines) >= 1 and align != 'center':
                first_line_indent = lines[0]['left_indent']
            if len(lines) > 1 and align != 'center':
                left_indent = lines[1]['left_indent']
                first_line_indent = lines[0]['left_indent'] - left_indent
            para_meta_data = {"line_space": line_space, "space_before": space_before, "align": align,
                              "font_size": font_size, "right_indent": right_indent, \
                              "first_line_indent": first_line_indent, "left_indent": left_indent}
            p_tag, index = self.process_paragraph(index, para_meta_data, lines=lines)
            td.append(p_tag)
        return td, index
        
    def process_table(self, name_id, table_meta_data):
        row_num = table_meta_data['row_num']
        col_num = table_meta_data['col_num']
        height = table_meta_data['page_bbox'][3] - table_meta_data['page_bbox'][1]
        width  = table_meta_data['page_bbox'][2] - table_meta_data['page_bbox'][0]
        row_heights = table_meta_data['row_heights']
        col_widths = table_meta_data['col_widths']
        width = sum(col_widths)
        cell_index =  table_meta_data['cell_index']
        cells = table_meta_data['cells']
        width_ratio = table_meta_data['width_ratio']
        table, name_id =  self.add_table_tag(name_id, width_ratio)
        for i in range(row_num):
            row_index = cell_index[i]
            rheight = row_heights[i]/height * 100
            # tr, name_id = self.add_table_row_tag(name_id, rheight)
            tr, name_id = self.add_table_row_tag(name_id, row_heights[i])
            j = 0
            while j<col_num:
                cell_idx = row_index['row'][j]
                cur_width = col_widths[j]
                if (j>0 and cell_idx!=-1 and cell_idx==row_index['row'][j-1]) or (i>0 and cell_idx!=-1 and cell_idx==cell_index[i-1]['row'][j]):
                    j += 1
                    continue
                elif cell_idx==-1:
                    rwidth = cur_width / width * 100
                    td, name_id = self.add_cell_tag(name_id, rwidth)
                    j += 1
                    tr.append(td)
                    continue
                colspan = 0
                rowspan = 0
                if j<col_num-1 and cell_idx==row_index['row'][j+1]:
                    colspan = 2
                    j = j+1
                    cur_width +=  col_widths[j]
                    while j<col_num-1 and cell_idx==row_index['row'][j+1]:
                        j += 1
                        colspan += 1
                        cur_width += col_widths[j]
                    # rwidth = cur_width/width * 100# * width_ratio
                    # td, name_id = self.add_cell_tag(name_id, rwidth, colspan=colspan)
                    # td, name_id = self.add_para_in_cell(td, name_id, cells[cell_idx]['paragraphs'])
                    # j += 1
                    # tr.append(td)
                if i<row_num-1 and cell_idx==cell_index[i+1]['row'][j]:
                    rowspan = 2
                    row_id = i + 1
                    while row_id<row_num-1 and cell_idx==cell_index[row_id+1]['row'][j]:
                        row_id += 1
                        rowspan += 1
                    # rwidth = cur_width / width * 100# * width_ratio
                    # td, name_id = self.add_cell_tag(name_id, rwidth, rowspan=rowspan)
                    # td, name_id = self.add_para_in_cell(td, name_id, cells[cell_idx]['paragraphs'])
                    # j += 1
                    # tr.append(td)
                rwidth = cur_width / width * 100 #* width_ratio
                td, name_id = self.add_cell_tag(name_id, rwidth, colspan=colspan, rowspan=rowspan)
                if 'cell_pics' in cells[cell_idx].keys():
                    td, name_id = self.add_para_in_cell_with_pics(td, name_id, cells[cell_idx]['paragraphs'],cells[cell_idx]['cell_pics'])
                else:
                    td, name_id = self.add_para_in_cell(td, name_id, cells[cell_idx]['paragraphs'])
                j += 1
                tr.append(td)
            table.append(tr)
        return table, name_id

    def process_paragraph(self, name_id, para_meta_data, lines=[], text='', pad_p=False):
        p_tag, name_id = self.add_para_tag(name_id, para_meta_data)
        if text!='':
            p_tag.string = text
        if pad_p:
            span_meta_data = {'font_size': 12}
            span_tags, name_id = self.add_span_tag(name_id, '  ', span_meta_data)
            for tag in span_tags:
                p_tag.append(tag)
            return p_tag, name_id
        for line in lines:
            span_meta_data = {'font_size':para_meta_data['font_size']}
            if 'w_space' in line.keys():
                span_meta_data['w_space'] = line['w_space']

            line_text = ''
            if 'mathml_text' not in line.keys():
                for text in line['line_text']:
                    line_text+=text['text']
                span_tags, name_id = self.add_span_tag(name_id, line_text, span_meta_data)
                for tag in span_tags:
                    p_tag.append(tag)
            else:
                for text in line['mathml_text']:
                    if 'math' in text:
                        tmp_soup = BeautifulSoup(text, features='lxml')
                        math_tag =tmp_soup.find('math')
                        if math_tag is None:
                            span_tags, name_id = self.add_span_tag(name_id, text, span_meta_data)
                            for tag in span_tags:
                                p_tag.append(tag)
                        else:
                            p_tag.append(math_tag)
                    else:
                        span_tags, name_id = self.add_span_tag(name_id, text, span_meta_data)
                        for tag in span_tags:
                            p_tag.append(tag)
        return p_tag, name_id

    def merge_all_content(self, meta_data):
        rets = []
        for image_info in meta_data['images_info']:
            ret = {}
            #将浮动内容和段落 归并至栏内,按高度排序
            img_size_info = image_info['img_info']
            doc_sections = image_info['doc_sections']
            doc_colums = doc_sections[0]['doc_colums']
            doc_pics = image_info['doc_pics']
            doc_tables = image_info['doc_tables']
            doc_textboxs = image_info['doc_textboxs']
            float_contents = doc_textboxs+doc_pics+doc_tables
            outside_colum_contents = []
            if doc_colums==[]:
                doc_colums = [{}]
                doc_colums[0]['contents'] = float_contents
                doc_colums[0]['page_range'] = [0,0,image_info['doc_page_info']['page_width'],image_info['doc_page_info']['page_height']]
                doc_colums[0]['colum_space'] = 0
                doc_colums[0]['width_ratio'] = 1
            else:
                for doc_colum in doc_colums:
                    colum_contents = doc_colum['paragraphs']
                    doc_colum['contents'] = doc_colum['float_contents']
                    for content in colum_contents:
                        if 'is_place_para' in content.keys() and content['font_size']==4:
                            continue
                        bbox = content['bbox']
                        para_in_pic = False
                        for float_content in float_contents:
                            fbbox = float_content['bbox']
                            iou = bbox_iou(fbbox, bbox)
                            if iou>0.9:
                                para_in_pic = True
                                break
                        if not para_in_pic:
                            doc_colum['contents'].append(content)
                # for idx, float_content in enumerate(float_contents):
                #     px1, py1, px2, py2 = float_content['bbox']
                #     cid = 0
                #     flag = False
                #     for ii, doc_colum in enumerate(doc_colums):
                #         colum_bbox = doc_colum['bbox']
                #         cx1, cy1, cx2, cy2 = colum_bbox
                #         if bbox_iou([cx1, cy1, cx2, cy2], [px1, py1, px2, py2]):
                #             cid = ii
                #             flag = True
                #             break
                #     if flag:
                #         doc_colums[cid]['contents'].append(float_content)
                #         float_content['width_ratio'] = (px2-px1)/(cx2-cx1)
                #     else:
                #         outside_colum_contents.append(float_content)

            for doc_colum in doc_colums:
                doc_colum['contents'] = sorted(doc_colum['contents'], key=lambda x:x['bbox'][1])
                x1_list = []
                y1_list = []
                x2_list = []
                y2_list = []
                for content in doc_colum['contents']:
                    x1_list.append(content['bbox'][0])
                    y1_list.append(content['bbox'][1])
                    x2_list.append(content['bbox'][2])
                    y2_list.append(content['bbox'][3])
                if 'bbox' not in doc_colum.keys() and x1_list!=[]:
                    doc_colum['bbox'] = [min(x1_list), min(y1_list), max(x2_list), max(y2_list)]
                elif 'bbox' not in doc_colum.keys() and x1_list==[]:
                    doc_colum['bbox'] = [0,0,img_size_info['valid_img_w'], img_size_info['valid_img_h']]
                elif 'bbox' in doc_colum.keys() and x1_list!=[]:
                    doc_colum['bbox'][0] = min(doc_colum['bbox'][0], min(x1_list))
                    doc_colum['bbox'][1] = min(doc_colum['bbox'][1], min(y1_list))
                    doc_colum['bbox'][2] = max(doc_colum['bbox'][2], max(x2_list))
                    doc_colum['bbox'][3] = max(doc_colum['bbox'][3], max(y2_list))
            for doc_colum in doc_colums:
                for cid, content in enumerate(doc_colum['contents']):
                    content['width_ratio'] =  min(1, 1*(content['bbox'][2] - content['bbox'][0]) / ( doc_colum['bbox'][2] - doc_colum['bbox'][0]))
                    if cid!=len(doc_colum['contents'])-1 and content['type']=='paragraph' and \
                            doc_colum['contents'][cid+1]['type']!='paragraph' and content['n_pad_line']>0:
                        next_height = (doc_colum['contents'][cid+1]['page_bbox'][3] - doc_colum['contents'][cid+1]['page_bbox'][1])//(12*0.0352)
                        content['n_pad_line'] = max(0, content['n_pad_line']-next_height)
            sections = self.split_section(doc_colums, outside_colum_contents)
            ret['sections'] = sections
            ret['img_size_info'] = img_size_info
            rets.append(ret)
        return rets

    def split_div2(self, meta_data, overlap_pix_threshold=5):
        all_divs = []
        for data in meta_data:
            sections = data['sections']
            img_size_info = data['img_size_info']
            parent_bbox = [0,0,img_size_info['valid_img_w'], img_size_info['valid_img_h']]
            sections = sorted(sections, key=lambda x:x['bbox'][1])
            out_sections = []
            sec_id = 0
            while sec_id<len(sections):
                section = sections[sec_id]
                bbox = section['bbox']
                type_ = section['type']
                contents = section['contents']
                if sec_id == len(sections) - 1 or bbox[3] <sections[sec_id+1]['bbox'][1] + overlap_pix_threshold:
                    if type_=='outside_colum_content':
                        out_sections.extend(contents)
                    elif type_=='section':
                        for cid, colum in enumerate(contents):
                            ret = self.split_colum_div(colum, bbox)
                            # if cid != len(contents) - 1:
                            out_sections.append(ret)
                            if cid!=len(contents)-1:
                                blank_div = {'type':'div', 'bbox':colum['bbox'], 'contents':[], 'float':'left', 'align':'left', 'colum_height_ratio':ret['colum_height_ratio'], 'colum_width_ratio':2}
                                out_sections.append(blank_div)
                    sec_id += 1
                else:
                    next_section = sections[sec_id+1]
                    div_h1 = min(section['bbox'][1], next_section['bbox'][1])
                    div_h2 = max(section['bbox'][3], next_section['bbox'][3])
                    div1 = {'type':'div',
                            'bbox': [section['bbox'][0], div_h1, section['bbox'][2], div_h2],
                            'contents': contents}
                    div2 = {'type': 'div',
                            'bbox': [next_section['bbox'][0], div_h1, next_section['bbox'][2], div_h2],
                            'contents': next_section['contents']}
                    sec_id = sec_id + 2
                    divs = [div1, div2]
                    while sec_id < len(sections) and sections[sec_id]['bbox'][1] + overlap_pix_threshold < div_h2:
                        divs = sorted(divs, key=lambda x: x['bbox'][0])
                        section = sections[sec_id]
                        div_h2 = max(div_h2, section['bbox'][1])
                        w_ious = []
                        for div in divs:
                            div['bbox'][3] = div_h2
                            div_w1 = div['bbox'][0]
                            div_w2 = div['bbox'][2]
                            w_iou = iou_1d([div_w1, div_w2], [section['bbox'][0], section['bbox'][2]])
                            w_ious.append(w_iou)
                        if max(w_ious) > 0.2:
                            div_id = w_ious.index(max(w_ious))
                            divs[div_id]['bbox'][0] = min(divs[div_id]['bbox'][0], section['bbox'][0])
                            divs[div_id]['bbox'][2] = max(divs[div_id]['bbox'][2], section['bbox'][2])
                            divs[div_id]['contents'].extend(section['contents'])
                        else:
                            div = {'type':'div',
                                   'bbox': [section['bbox'][0], div_h1, section['bbox'][2], div_h2],
                                   'contents': section['contents']}
                            divs.append(div)
                        sec_id += 1
                    divs = sorted(divs, key=lambda x: x['bbox'][0])
                    ratio_sum = 0
                    colum_height_ratio = 100 * (div_h2 - div_h1) / (parent_bbox[3] - parent_bbox[1])
                    for id, div in enumerate(divs):
                        div['float'] = 'left'
                        div['align'] = 'left'
                        div['colum_height_ratio'] = colum_height_ratio
                        if id == len(divs) - 1:
                            div['bbox'][2] = parent_bbox[2]
                        if id == 0:
                            div['bbox'][0] = parent_bbox[0]
                        if id != 0:
                            div['bbox'][0] = divs[id - 1]['bbox'][2]
                        colum_width_ratio = int(100 * (div['bbox'][2] - div['bbox'][0]) / (parent_bbox[2] - parent_bbox[0]))
                        if id != len(divs) - 1:
                            div['colum_width_ratio'] = colum_width_ratio
                            ratio_sum += colum_width_ratio
                        else:
                            div['colum_width_ratio'] = 99 - ratio_sum
                        new_contents = []
                        for content in div['contents']:
                            if 'colum' == content['type']:
                                new_contents.append(self.split_colum_div(colum, bbox))
                            else:
                                new_contents.append(content)
                        div['contents'] = new_contents
                        div['colum_width_ratio'] = 100
                        out_sections.append(div)
            all_divs.append({'divs':out_sections, 'bbox':parent_bbox})
        return all_divs

    def split_colum_div(self, colum, parent_bbox, overlap_pix_threshold=5):
        # colum_height_ratio = 100*(colum['bbox'][3]-colum['bbox'][1])/(parent_bbox[3]-parent_bbox[1])
        colum_height_ratio = -1
        # colum_width_ratio = int(100*(colum['bbox'][2] -colum['bbox'][0])/(parent_bbox[2]-parent_bbox[0]))
        # colum_width_ratio = colum['width_ratio']
        colum_width_ratio = 100
        colum_div = {'is_colum':'1', 'type':'div', 'bbox':colum['bbox'], 'contents':[], 'float':'left', 'align':'left', 'colum_height_ratio':colum_height_ratio, 'colum_width_ratio':colum_width_ratio}
        parent_bbox = colum_div['bbox']
        contents = colum['contents']
        content_id = 0
        while content_id < len(contents):
            content = contents[content_id]
            bbox = content['bbox']
            type_ = content['type']
            if content_id == len(contents) - 1 or bbox[3] < contents[content_id + 1]['bbox'][1] + overlap_pix_threshold or (type_ == 'paragraph' and contents[content_id + 1]['type'] == 'paragraph'):
                if type_ in ['pic', 'table', 'textbox']:
                    div = { 'type':'div', 'bbox': [parent_bbox[0],bbox[1],parent_bbox[2],bbox[3]],'contents': [content]}
                    div['colum_width_ratio'] = 100
                    div['colum_height_ratio'] = -1
                    div['float'] = 'left'
                    div['align'] = 'left'
                    if type_=='table':
                        div['align'] = 'center'
                    colum_div['contents'].append(div)
                else:
                    colum_div['contents'].append(content)
                content_id += 1
            else:
                next_content = contents[content_id + 1]
                if iou_1d([content['bbox'][0], content['bbox'][2]], [next_content['bbox'][0], next_content['bbox'][2]]) > 0.2 and content['type']=='paragraph':
                    colum_div['contents'].append(content)
                    content_id += 1
                    continue
                div_h1 = min(content['bbox'][1], next_content['bbox'][1])
                div_h2 = max(content['bbox'][3], next_content['bbox'][3])
                div1 = {'type': 'div',
                        'bbox': [content['bbox'][0], div_h1, content['bbox'][2], div_h2],
                        'contents': [content]}
                div2 = {'type': 'div',
                        'bbox': [next_content['bbox'][0], div_h1, next_content['bbox'][2], div_h2],
                        'contents': [next_content]}
                content_id = content_id + 2
                divs = [div1, div2]

                while content_id < len(contents) and contents[content_id]['bbox'][1] + overlap_pix_threshold*4 < div_h2:
                    content = contents[content_id]
                    divs = sorted(divs, key=lambda x: x['bbox'][0])
                    div_h2 = max(div_h2, content['bbox'][3])
                    w_ious = []
                    for div in divs:
                        div['bbox'][3] = div_h2
                        div_w1 = div['bbox'][0]
                        div_w2 = div['bbox'][2]
                        w_iou = iou_1d([div_w1, div_w2], [content['bbox'][0], content['bbox'][2]])
                        w_ious.append(w_iou)
                    overlap_cnt = 0
                    for w_iou in w_ious:
                        if w_iou>0.2:
                            overlap_cnt+=1
                    # if overlap_cnt>=2:
                    #     content_id += 1
                    #     colum_div['contents'].append(content)
                    #     continue
                    if max(w_ious) > 0.2:
                        div_id = w_ious.index(max(w_ious))
                        for idx, w_iou in enumerate(w_ious):
                            if w_iou>0.2:
                                div_id = idx
                                break
                        divs[div_id]['bbox'][0] = min(divs[div_id]['bbox'][0], content['bbox'][0])
                        divs[div_id]['bbox'][2] = max(divs[div_id]['bbox'][2], content['bbox'][2])
                        divs[div_id]['contents'].append(content)
                    else:
                        div = {
                            'type':'div',
                            'bbox': [content['bbox'][0], div_h1, content['bbox'][2], div_h2],
                            'contents': [content]}
                        divs.append(div)
                    content_id += 1
                divs = sorted(divs, key=lambda x: x['bbox'][2]-x['bbox'][0])
                divs = divs[::-1]
                div0 = divs[0]
                has_overlap_div = False
                for id, div in enumerate(divs[1:]):
                    if iou_1d([div0['bbox'][0], div0['bbox'][2]], [div['bbox'][0], div['bbox'][2]], real_iou=False)>0.5:
                        has_overlap_div = True
                if has_overlap_div:
                    div0['float'] = 'left'
                    div0['align'] = 'left'
                    div0['colum_height_ratio'] = -1
                    div0['colum_width_ratio'] = 100
                    colum_div['contents'].append(div0)
                    divs = divs[1:]

                divs = sorted(divs, key=lambda x: x['bbox'][0])
                ratio_sum = 0
                colum_height_ratio = 100 * (div_h2 - div_h1) / (parent_bbox[3] - parent_bbox[1])
                place_div = {'type': 'div',
                        'bbox': [next_content['bbox'][0], div_h1, next_content['bbox'][2], div_h2],
                        'contents': [], 'colum_height_ratio':-1, 'colum_width_ratio':100,'align':'center', 'float':'left'}

                for id, div in enumerate(divs):
                    div['float'] = 'left'
                    div['align'] = 'left'
                    div['colum_height_ratio'] = -1
                    if id == 0:
                        div['bbox'][0] = parent_bbox[0]
                    if id != 0:
                        div['bbox'][0] = divs[id - 1]['bbox'][2]
                    colum_width_ratio = min(100-ratio_sum, int(100 * (div['bbox'][2] - div['bbox'][0]) / (parent_bbox[2] - parent_bbox[0])))
                    new_contents = []
                    p_div = False
                    paras = []
                    others = []
                    for ii, content in enumerate(div['contents']):
                        if id!=0 and content['type']=='paragraph':
                            for line_id in range(len(content['lines'])):
                                content['lines'][line_id]['left_indent'] = 0
                        if content['type'] ==  'paragraph':
                            p_div = True
                        elif content['type'] ==  'table' and colum_width_ratio>5:
                            content['width_ratio'] = content['width_ratio'] * 100. / colum_width_ratio
                        if content['type'] ==  'paragraph':
                            paras.append(content)
                        else:
                            others.append(content)
                    if len(paras)!=0 and len(others)!=0:
                        others[0]['n_pad_line'] = paras[0]['n_pad_line']
                        paras[0]['n_pad_line'] = 0
                    new_contents = paras+others

                    if colum_width_ratio<15 and p_div:
                        colum_width_ratio += 3
                    div['colum_width_ratio'] = colum_width_ratio
                    ratio_sum += colum_width_ratio
                    div['contents'] = new_contents
                    # colum_div['contents'].append(div)
                    place_div['contents'].append(div)
                    if id == len(divs) - 1 and ratio_sum<99:
                       pad_div = {'type': 'div',
                        'bbox': [next_content['bbox'][0], div_h1, next_content['bbox'][2], div_h2],
                        'contents': [], 'colum_width_ratio':99 - ratio_sum, \
                                  'colum_height_ratio':-1, 'align':'left', 'float':'left'}
                       # colum_div['contents'].append(pad_div)
                       place_div['contents'].append(pad_div)
                colum_div['contents'].append(place_div)
        #colum_div['contents'] = sorted(colum_div['contents'], key=lambda x:x['bbox'][1])
        return colum_div

    def split_section(self, colums, outside_colum_contents):
        section = {'type':'section', 'contents':colums}
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for colum in colums:
            x1_list.append(colum['bbox'][0])
            y1_list.append(colum['bbox'][1])
            x2_list.append(colum['bbox'][2])
            y2_list.append(colum['bbox'][3])
        section['bbox'] = [min(x1_list), min(y1_list), max(x2_list), max(y2_list)]
        sections = [section]
        for content in outside_colum_contents:
            sections.append({'type':'outside_colum_content', 'contents':[content], 'bbox':content['bbox']})
        return sections

    def process_div(self, index, block, content, parent_bbox):
        if content['type'] == 'paragraph':
            paragraph = content
            # if index==34:
            #     print(content)
            line_space = paragraph['line_space']
            space_before = paragraph['para_space_before']
            align = paragraph['align']
            font_size = paragraph['font_size']
            right_indent = paragraph['right_indent']
            lines = paragraph['lines']
            left_indent = 0.
            first_line_indent = 0
            if len(lines) >= 1 and align != 'center':
                left_indent = lines[0]['left_indent']
            if len(lines) > 1 and align != 'center':
                left_indent = lines[1]['left_indent']
                first_line_indent = lines[0]['left_indent'] - left_indent
            para_meta_data = {"line_space": line_space, "space_before": space_before, "align": align,
                              "font_size": font_size, "right_indent": right_indent, \
                              "first_line_indent": first_line_indent, "left_indent": left_indent}
            p_tag, index = self.process_paragraph(index, para_meta_data, lines=lines)
            block.append(p_tag)
            n_pad_line = paragraph['n_pad_line']
            if n_pad_line >0:
                pad_height = max(0, (n_pad_line)*12*0.0352)
                if pad_height>0:
                    blank_div, index = self.add_div_tag(index, [0, 0, 100, pad_height], absolute=True)
                    block.append(blank_div)
        elif content['type'] == 'pic':
            margin_left = int(100 * (content['bbox'][0] - parent_bbox[0]) / (parent_bbox[2] - parent_bbox[0]))
            content["margin_left"] = margin_left
            if 'url' in content.keys() and os.path.exists(content['url']):
                img_tag, index = self.add_pic_tag(index, content)
                block.append(img_tag)
                if 'n_pad_line' in content.keys():
                    n_pad_line = content['n_pad_line']
                    pad_height = max(0, (n_pad_line)*12*0.0352)
                    if pad_height > 0:
                        blank_div, index = self.add_div_tag(index, [0, 0, 100, pad_height], absolute=True)
                        block.append(blank_div)
        elif content['type'] == 'table':
            table_tag, index = self.process_table(index, content)
            block.append(table_tag)
        elif content['type'] in ['textbox', 'text']:
            txbx = content
            font_size = txbx['font_size']
            line_space = font_size*1.5*0.0352
            space_before = 0
            align = 'left'
            right_indent = 0
            left_indent = 0.
            first_line_indent = 0
            para_meta_data = {"line_space": line_space, "space_before": space_before, "align": align,
                              "font_size": font_size, "right_indent": right_indent, \
                              "first_line_indent": first_line_indent, "left_indent": left_indent}
            margin_left = int(100*(txbx['bbox'][0] - parent_bbox[0])/(parent_bbox[2] - parent_bbox[0]))
            para_meta_data["margin_left"] = margin_left
            p_tag, index = self.process_paragraph(index, para_meta_data, text=txbx['text'])
            block.append(p_tag)
        elif content['type'] == 'div':
            colum_width_ratio = content['colum_width_ratio']
            colum_height_ratio = content['colum_height_ratio']
            align = content['align']
            subblock, index = self.add_div_tag(index, [0, 0, colum_width_ratio, colum_height_ratio], float=content['float'], align=align)
            parent_bbox = content['bbox']
            for sub_id, subcontent in enumerate(content['contents']):
                if sub_id==0 and subcontent['type']=='paragraph':
                    subcontent['para_space_before'] = 0.
                subblock, index = self.process_div(index, subblock, subcontent, parent_bbox)
            block.append(subblock)
            if 'is_colum' in content.keys():
                p_tag, index = self.add_blank_para_tag(index)
                block.append(p_tag)
                line_tag = self.add_hr_tag()
                br = self.bs.new_tag("br")
                block.append(line_tag)
                block.append(br)
                br = self.bs.new_tag("br")
                block.append(br)
        return block, index

    def process(self, meta_data=None, json_file=None, html_file=None):
        if json_file is not None:
            with open(json_file) as f:
                meta_data = json.load(f)
        self.work_dir = ''
        self.prefix = html_file.replace('.html','')

        self.html_files_dir = self.prefix+".files"
        if os.path.exists(self.html_files_dir) is not True:
            os.makedirs(self.html_files_dir)
        self.url_prefix = os.path.basename(self.html_files_dir)
        self.bs = BeautifulSoup(self.template_html, "html.parser")
        image_info = meta_data['images_info'][0]
        doc_page_info = image_info['doc_page_info']
        self.set_body_attributes(doc_page_info)
        self.page_width = doc_page_info['page_width']
        meta_data = self.merge_all_content(meta_data)
        multi_page_infos = self.split_div2(meta_data)
        index = 0
        for single_page_info in multi_page_infos:
            bbox = single_page_info['bbox']
            divs = single_page_info['divs']
            for div in divs:
                _, index = self.process_div(index, self.bs.body, div, bbox)
        with open(html_file, 'w') as f:
            f.write(str(self.bs))
                
if __name__ == "__main__":
    hg = html_generator("template.html")
    hg.process(json_file='2.json', html_file='test.html')

    
