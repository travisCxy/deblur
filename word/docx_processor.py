import zipfile
import codecs
import os, re
import datetime
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import shutil

# prepare global variables
template_path = os.path.join(os.path.dirname(__file__), 'template')
def read_template_resource(path, open_mode="rb", encoding=None):
    global template_path
    with codecs.open(template_path + path, open_mode, encoding=encoding) as f:
        return f.read()

def read_template_resources():
    static_template_resource_pathes = ['/word/media/image1.png',
                                       '/word/media/image2.png',
                                       '/word/theme/theme1.xml',
                                       '/word/endnotes.xml',
                                       '/word/fontTable.xml',
                                       '/word/footnotes.xml',
                                       '/word/settings.xml',
                                       '/word/styles.xml',
                                       '/word/webSettings.xml'
                                       ]
    result = {}
    for k in static_template_resource_pathes:
        result[k[1:]] = read_template_resource(k)
    return result

def read_document_resource(xml):
    template_doc_str = read_template_resource(xml, 'r', 'utf-8')
    idx = template_doc_str.find("<w:sectPr>")
    return template_doc_str[:idx], template_doc_str[idx:]

def read_content_type_resource():
    c_type = read_template_resource('/[Content_Types].xml', 'r', 'utf-8')
    idx = c_type.find('</Types>')
    c_type_image = c_type[:idx]+'<Default ContentType="image/jpeg" Extension="jpg"/>' + c_type[idx:]
    return c_type, c_type_image


static_template_resource_map = read_template_resources()
template_doc_str1, template_doc_str2 = read_document_resource('/word/document.xml')
template_doc_str1_v2, template_doc_str2_v2 = read_document_resource('/word/document2.xml')


#在多栏情况下用于保持头单栏的节属性模板
template_header_section_str = "<w:p w:rsidR=\"00501420\" w:rsidRDefault=\"00501420\"><w:pPr><w:sectPr w:rsidR=\"00501420\"><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1200\" w:right=\"1200\" w:bottom=\"1200\" w:left=\"1200\" w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/><w:cols w:space=\"720\"/><w:docGrid w:type=\"lines\" w:linePitch=\"312\"/></w:sectPr></w:pPr></w:p>"
content_types, content_types_with_image = read_content_type_resource()
header_xml_str = read_template_resource('/word/header1.xml', 'r', 'utf-8')

def get_image_list(ref_str):
    image_list = []
    segs = ref_str.split('<Relationship ')
    for seg in segs:
        if 'Id="' in seg and 'Target="' in seg:
            s = seg[seg.find('Id="')+4:]
            id = s[:s.find('"')]
            s = seg[seg.find('Target="')+8:]
            target= s[:s.find('"')]
            image_list.append((id, target))
    return image_list

def merge_images_to_zip(merge_info, zip, out_zip):
    if "word/_rels/document.xml.rels" in zip.namelist():
        ref_str = zip.read("word/_rels/document.xml.rels").decode()
        image_list = get_image_list(ref_str)
        for id, target in image_list:
            if "media" not in target:
                continue
            if target[0] == '/':
                target = target[1:]
            else:
                target = 'word/' + target
            data = zip.read(target)
            fname = "media/image"+str(merge_info[0])+".jpg"
            out_zip.writestr(fname, data)
            merge_info[0] += 1
            merge_info[1] += '<Relationship Id="%s" Target="/%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>' % (id, fname)

def copy_refs(has_image, out_zip):
    global content_types, content_types_with_image
    out_zip.writestr('_rels/.rels', '<?xml version="1.0" encoding="utf-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="/word/document.xml" Id="Rsjb" /></Relationships>')
    if has_image:
        out_zip.writestr('[Content_Types].xml', content_types_with_image)
    else:
        out_zip.writestr('[Content_Types].xml', content_types)

def get_document_word(zip1, idx):
    doc_str = zip1.read("word/document.xml").decode()
    insert_str = doc_str[doc_str.find("<w:body>")+8:doc_str.rfind("</w:body>")]
    #insert_str = insert_str[:insert_str.find("<w:sectPr>")] + insert_str[insert_str.rfind("</w:sectPr>")+11:]
    insert_str = re.sub(r'<w:sectPr>.*?</w:sectPr>', 'sjb_doc_split_mark', insert_str)
    insert_str_splits = insert_str.split('sjb_doc_split_mark')
    result_str = ''+ insert_str_splits[0]
    for i in range(1, len(insert_str_splits)):
        if insert_str_splits[i]=='':
            continue
        result_str += "<w:p/>" * 5
        idx = idx + 1
        result_str = add_topic_head_str(result_str, idx)
        result_str += insert_str_splits[i]
    idx += 1
    return result_str, idx

def insert_document_word(output_zip, insert_str):
    global template_doc_str1, template_doc_str2
    doc_str = template_doc_str1 + insert_str + template_doc_str2
    output_zip.writestr("word/document.xml", doc_str)

def insert_document_word_v2(output_zip, insert_str, subject_name, num_columns=1):
    #print(len(insert_str))
    if '<w:br w:type=\"page\" />' in insert_str:
        insert_str = insert_str.replace('<w:br w:type=\"page\" />', '')
    #print(len(insert_str))
    global template_doc_str1_v2, template_doc_str2_v2
    global template_header_section_str
    if num_columns == 2:
        tmp_str = template_doc_str2_v2.replace("<w:pgSz", "<w:type w:val=\"continuous\"/><w:pgSz")
        tmp_str = tmp_str.replace("<w:cols w:space=\"720\"/>", "<w:cols w:num=\"2\" w:space=\"720\" w:sep=\"1\"/>")
    elif num_columns == 3:
        tmp_str = template_doc_str2_v2.replace("<w:pgSz", "<w:type w:val=\"continuous\"/><w:pgSz")
        tmp_str = tmp_str.replace("<w:cols w:space=\"720\"/>", "<w:cols w:num=\"3\" w:space=\"720\" w:sep=\"1\"/>")
    #print(template_doc_str2_v2)
    now = datetime.datetime.now()
    t_str = now.strftime('%Y-%m-%d %H:%M:%S')
    doc_str1_v2 = template_doc_str1_v2.replace("DATE_PLACEHOLDER", t_str)
    doc_str1_v2 = doc_str1_v2.replace("SUBJECT_PLACEHOLDER", subject_name)
    if num_columns==1:
        doc_str = doc_str1_v2 + insert_str + template_doc_str2_v2
    else:
        doc_str = doc_str1_v2 + template_header_section_str + insert_str + tmp_str
    output_zip.writestr("word/document.xml", doc_str)



def insert_word_refs(output_zip, subject_name, qrcode_image, relation_ship_str):
    global header_xml_str

    relation_ships_str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId21" Target="header1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"/>
<Relationship Id="rId22" Target="media/qrcode.png" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
<Relationship Id="rId23" Target="media/image2.png" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
<Relationship Id="rId24" Target="webSettings.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings"/>
<Relationship Id="rId25" Target="settings.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings"/>
<Relationship Id="rId26" Target="styles.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"/>
<Relationship Id="rId27" Target="endnotes.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"/>
<Relationship Id="rId28" Target="theme/theme1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"/>
<Relationship Id="rId29" Target="footnotes.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"/>
<Relationship Id="rId30" Target="fontTable.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable"/>
%s
</Relationships>
    ''' % relation_ship_str
    output_zip.writestr("word/_rels/document.xml.rels", relation_ships_str)
    if qrcode_image is not None:
        now = datetime.datetime.now()
        t_str = now.strftime('%Y-%m-%d')
        header1_str = header_xml_str.replace("DATE_PLACEHOLDER", t_str)
        header1_str = header1_str.replace("SUBJECT_PLACEHOLDER", subject_name)

        header_ref_str = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/></Relationships>'
        output_zip.writestr("word/_rels/header1.xml.rels", header_ref_str)

        output_zip.writestr("word/media/qrcode.png", qrcode_image)
        output_zip.writestr("word/header1.xml", header1_str)

def insert_static_resources(output_zip):
    global static_template_resource_map
    for k, v in static_template_resource_map.items():
        output_zip.writestr(k, v)

def add_topic_head_str(insert_str, topic_id):
    topic_str = '题目%s'%str(topic_id)
    insert_str += '<w:p><w:pPr><w:spacing w:after="0"/><w:rPr><w:rFonts w:hint="default" w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体" w:cs="宋体"/><w:b w:val="1"/><w:bCs w:val="1"/><w:sz w:val="24"/><w:szCs w:val="24"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:pPr><w:r><w:rPr><w:rFonts w:hint="eastAsia" w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体" w:cs="宋体"/><w:b w:val="1"/><w:bCs w:val="1"/><w:sz w:val="24"/><w:szCs w:val="24"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr><w:t>' + topic_str + '</w:t></w:r><w:bookmarkStart w:id="0" w:name="_GoBack"/><w:bookmarkEnd w:id="0"/></w:p>'
    return insert_str

def merge_docxs(docx_name, subject_name, qrcode_image, docxs, num_columns=1):
    insert_str = ""
    first = True
    topic_idx = 1
    relation_ship_str = ""
    merge_info = [topic_idx, relation_ship_str]
    if subject_name is None:
        subject_name = ""
    qrcode_image = None
    with zipfile.ZipFile(docx_name, mode="w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for idx, docx in enumerate(docxs):
            with zipfile.ZipFile(docx) as zip:
                if first:
                    first = False
                else:
                    insert_str += "<w:p/>" * 5
                if qrcode_image is  None:
                    insert_str = add_topic_head_str(insert_str, topic_idx)
                cur_insert_str, topic_idx = get_document_word(zip, topic_idx)
                insert_str += cur_insert_str
                merge_images_to_zip(merge_info, zip, output_zip)
        if qrcode_image is None:
            insert_document_word_v2(output_zip, insert_str, subject_name, num_columns=num_columns)
        else:
            insert_document_word(output_zip, insert_str)
        image_relation_ship_str = merge_info[1]
        has_image = len(image_relation_ship_str) > 0
        insert_word_refs(output_zip, subject_name, qrcode_image, image_relation_ship_str)
        copy_refs(has_image, output_zip)
        insert_static_resources(output_zip)


def merge_htmls(htmls, subject, output_html_path, output_html_files_path):
    bs_htmls = []
    if subject is None:
        subject = ''
    for html in htmls:
        #bs = BeautifulSoup(html.stream.read(), "html.parser")
        bs = BeautifulSoup(open(html).read(), "html.parser")
        bs_htmls.append(bs)
    for idx, bs in enumerate(bs_htmls):
        bs_prefix = 'bs' + str(idx+1)
        bs = rename_style(bs, bs_prefix, output_html_files_path)
    output_bs = BeautifulSoup(open('word/template/template.html').read(), "html.parser")
    output_bs = add_time_subject_str(output_bs, subject, output_html_files_path)
    topic_idx = 1
    for idx, bs in enumerate(bs_htmls):
        if idx==0:
            add_topic_tag(output_bs, topic_idx)
            topic_idx += 1
        for cid in range(len(bs.body.contents)):
            if bs.body.contents[cid].name == 'hr':
                if not (idx==len(bs_htmls)-1 and cid == len(bs.body.contents)-3):
                    add_topic_tag(output_bs, topic_idx)
                    topic_idx += 1
                continue
            if bs.body.contents[cid].name == 'br':
                continue
            output_bs.body.append(bs.body.contents[cid].__copy__())
        add_div_tag(output_bs, 'blank'+str(idx+1))
    for idx, bs in enumerate(bs_htmls):
        for content in bs.style.contents:
            output_bs.style.append(content)
    with open(output_html_path, 'w') as f:
        f.write(str(output_bs))

def rename_style(bs, bs_prefix, output_html_files_path):
    for tag_prefix in ['span', 'p', 'div', 'table', 'td', 'tr', 'img']:
            tags = bs.find_all(tag_prefix)
            for tag_index, tag in enumerate(tags):
                tag_attrs = tag.attrs
                if tag_attrs is not None and 'class' in tag_attrs.keys():
                    tags[tag_index]['class'][0] = tags[tag_index]['class'][0].replace('pt-', bs_prefix+'_pt-')
    for style_i in range(len(bs.style.contents)):
        bs.style.contents[style_i] = bs.style.contents[style_i].replace('pt-', bs_prefix+'_pt-')
    imgs = bs.find_all('img')
    for img in imgs:
        imgpath = os.path.join('tmp', img.attrs['src'])
        imgname = os.path.basename(imgpath)
        img.attrs['src'] = os.path.join(os.path.basename(output_html_files_path), imgname)
        if os.path.exists(imgpath):
            shutil.copyfile(imgpath, os.path.join(output_html_files_path, imgname))
    return bs


def get_style(name, atts):
    atts_string = ''
    for key, value in atts.items():
        atts_string += "%s:%s;\n"%(key, value)
    style_string = '%s{\n%s}'%(name, atts_string)
    return style_string+'\n'
        
        
def add_topic_tag(bs, name_id):
    para_attrs = {"class":"topic-%06d"%name_id}
    para_style_attrs = {	
        "font-family":'宋体',
        "font-size":"12pt",
        "vertical-align":"baseline",
        "margin-bottom":"4pt",
        "-webkit-text-size-adjust": "none",
        "text-size-adjust": "none",
    }
    para_style = get_style("p."+"topic-%06d"%name_id, para_style_attrs)
    tag = bs.new_tag("p", attrs=para_attrs)
    tag.string = '题目'+str(name_id)
    bold_tag = bs.new_tag("b")
    bold_tag.append(tag)
    bs.head.style.string += para_style
    bs.body.append(bold_tag)

    return bs 
  
def add_div_tag(bs, name, height=None):
    block_attrs = {"class":name, "align":'center'}
    block_style_attrs = {
            "position":"relative",
            "left":"0%",
            "top":"0%",
            "-webkit-text-size-adjust":"none",
            "text-size-adjust":"none",
            "width":"100%",
            "float":'left',
            "background-color":'white'
            }
    if height==None:
        block_style_attrs['height'] = '3cm'
    else:
        block_style_attrs['height'] = str(height)+'cm'
    block_style = get_style("div."+name, block_style_attrs)
    tag = bs.new_tag("div", attrs=block_attrs)
    bs.head.style.string += block_style
    bs.body.append(tag)
    return bs
    
def add_time_subject_str(bs, subject, output_html_files_path):
    now = datetime.datetime.now()
    t_str = now.strftime('%Y-%m-%d %I:%M:%S')
    spans = bs.find_all('span')
    spans[2].string = t_str
    spans[1].string = spans[1].string.replace('科目', subject)
    img = bs.find("img")
    img.attrs['src'] = os.path.join(os.path.basename(output_html_files_path), os.path.basename(img.attrs['src']))
    return bs 



