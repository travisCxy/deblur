import json
import logging
import os
import uuid
import cv2
import io
import subprocess
import time
import sys
sys.path.append('word/converter')
import numpy as np
from PIL import Image
from triton_client import TritonClient
from word.word_ocr import WordOcr
from word.converter.doc_constructor import doc_builder
from word.converter.doc_utils import get_ratoted_box
colors = {1:(0,255,0), 3:(255,0,0), 4:(0,0,255), 10:(238, 238, 0), 5:(211,85,186), 6:(36,127,255)}

class WordEngine:
    def __init__(self, work_dir, triton_client):
        self._work_dir = work_dir
        self._wordocr = WordOcr(triton_client)
        self.doc_builder = doc_builder()
        

    def convert(self, images, todoc="word", subject="None", typesetting=False, generate_html=False , for_wrong_question=False, num_columns=1):
        paths_to_clear, cost = [], {}
        exception = 0
        if todoc == "txt" or not typesetting:
            images_info, prepare_cost = self.prepare_image_info(images, paths_to_clear)
            cost.update(prepare_cost)
            doc_path, convert_cost = self.do_convert(images_info, todoc, subject, paths_to_clear)
            if generate_html:
                cmd = "pandoc  -s %s -o %s" % (doc_path, doc_path.replace('.docx', '.html'))
                status, _ = subprocess.getstatusoutput(cmd)
                paths_to_clear.append(doc_path.replace('.docx', '.html'))
            cost.update(convert_cost)
        else:
            filename = str(uuid.uuid4())
            doc_path = os.path.abspath(self._work_dir + "/%s.docx" % filename)
            try:
                status, doc_path, paths_to_clear, cost, exception = self.convert_word(images, doc_path, generate_html=generate_html, for_wrong_question=for_wrong_question, num_columns=num_columns)
            except Exception as ex:
                logging.exception('images_to_word_convert: {}'.format(ex))
                status = 1
            if status != 0:
                self.clear_files(paths_to_clear)
                doc_path, paths_to_clear, cost, exception = self.convert(images, todoc, subject, generate_html=generate_html)
                exception += 1
                return doc_path, paths_to_clear, cost, exception
        return doc_path, paths_to_clear, cost, exception


    def convert_word(self, images, doc_file, work_dir = './', subject="paper", generate_html=False, for_wrong_question=False, num_columns=1):

        paths_to_clear, cost = [], {}
        start_time = time.time()
        images_info, exception = self.prepare_image_info0(images, paths_to_clear, subject, for_wrong_question=for_wrong_question)
        # for idx in range(len(images)):
        #     image = images[idx]
        #     image_info = images_info[idx]
        #     apath = os.path.join('/home/ateam/xychen/projects/image_2_word/user-pictures/test_wrong',
        #                                                    os.path.basename(doc_file.replace('.docx', '_show.jpg')))
        #     self.visualize(image, image_info, apath)
        # return doc_file, 0, 0, 0,0

        cost['detect'] = time.time()-start_time
        assert len(images)==len(images_info)
        start_time = time.time()
        status, ret = self.doc_builder.oxml_convert(imgs=images, meta_datas=images_info, doc_file=doc_file,
                                                    work_dir=work_dir,paths_to_clear=paths_to_clear, generate_html=generate_html , for_wrong_question=for_wrong_question, num_columns=num_columns)

        if status != 0:
            logging.error('oxml_convert: ' + ret)
        cost['convert'] = time.time()-start_time
        return status, doc_file, paths_to_clear, cost, exception

    def visualize(self, image, meta_data, save_path):
        image = image.copy()
        for region in meta_data['regions']:
            bbox = region['region']
            rotation = region['rotation']
            cls = region['cls']
            if cls==2:
                continue
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2),int(y2)), colors[cls],thickness=2)

        for region in meta_data['pics']:
            bbox = region['region']
            rotation = region['rotation']
            x1, y1, x2, y2 = get_ratoted_box(bbox, rotation)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[3], thickness=2)

        for table in meta_data['tables']:
            bbox = table['rect']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[5], thickness=5)
            for cell in table['cells']:
                cell_bbox = [int(cell['bbox'][0]), int(cell['bbox'][1]), int(cell['bbox'][2]), int(cell['bbox'][5])]
                x1, y1, x2, y2 = cell_bbox
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[6], thickness=2)
                for text in cell['texts']:
                    text_bbox = [int(text['bbox'][0]), int(text['bbox'][1]), int(text['bbox'][2]), int(text['bbox'][3])]
                    x1, y1, x2, y2 = text_bbox
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[1], thickness=2)

        cv2.imwrite(save_path, image)

    def do_convert(self, images_info, todoc, subject, paths_to_clear):
        filename = str(uuid.uuid4())
        json_path = os.path.abspath(self._work_dir + "/%s.json" % filename)
        doc_path = os.path.abspath(self._work_dir + "/%s.docx" % filename)
        if todoc == "txt":
            doc_path = os.path.abspath(self._work_dir + "/%s.txt" % filename)

        paths_to_clear.append(json_path)
        paths_to_clear.append(doc_path)

        start = time.time()
        with io.open(json_path, "w") as fp:
            json.dump(images_info, fp, ensure_ascii=False)

        converter_path = os.path.abspath("./bin/OpenXmlCore.dll")
        cmd = "dotnet %s %s %s %s %s" % (converter_path, todoc, json_path, doc_path, subject)
        status, ret = subprocess.getstatusoutput(cmd)
        end = time.time()

        return doc_path, {"convert": end-start}

    def detect_table(self, image_path, json_path, vis_path):
        image = np.array(Image.open(image_path).convert('RGB'))[:,:,::-1]
        ret = self._wordocr.identify_table(image, json_path, vis_path)
        return ret 

    def prepare_image_info(self, images,  paths_to_clear):
        infos, cost = [], {}
        for image in images:
            info, img_cost = self._wordocr.identify(image)
            self.fill_image(image, info, paths_to_clear)
            infos.append(info)

            for k, v in img_cost.items():
                ev = cost.setdefault(k, 0.0)
                cost[k] = ev + v
        cost["count"] = len(images)
        return {"images": infos}, cost

    def prepare_image_info0(self, images,  paths_to_clear, subject="paper",for_wrong_question=False):
        infos = []
        exception = 0
        for image in images:
            if subject=="paper":
                info, e = self._wordocr.identify_paper(image, for_wrong_question=for_wrong_question)
                infos.append(info)
                exception += e
            elif subject=="document":
                info = self._wordocr.identify_document(image)
                infos.append(info)
            else:
                raise ValueError("subject should be paper or document")
        return infos, exception

    def fill_image(self, image, data, paths_to_clear):
        topics = data["yyt"]["topics"]
        for topic in topics:
            if "pictures" not in topic:
                continue
            for picture in topic["pictures"]:
                x1, y1, x2, y2 = picture["region"]
                crop = image[y1:y2, x1:x2]
                crop_path = os.path.abspath(self._work_dir + "/%s.jpg" % str(uuid.uuid4()))
                cv2.imwrite(crop_path, crop)
                paths_to_clear.append(crop_path)
                picture["url"] = crop_path

    def clear_files(self, paths_to_clear):
        try:
            for clear_path in paths_to_clear:
                if os.path.exists(clear_path) == False:
                    continue
                if os.path.isdir(clear_path):
                    os.rmdir(clear_path)
                else:
                    os.remove(clear_path)
        except:
            pass

    def identify(self, image):
        info, img_cost = self._wordocr.identify(image)
        return info, img_cost
    
    def indentify_paper(self, image_file):
        image = np.array(Image.open(image_file).convert('RGB'))[:,:,::-1]
        info, img_cost = self._wordocr.indentify_paper(image)
        return info, img_cost

    def identify_document(self, image_file):
        image = cv2.imread(image_file)
        result = self._wordocr.identify_document(image)
        return result

    def identify_cell(self, image_file):
        image = cv2.imread(image_file)
        result = self._wordocr.identify_cell(image)
        return result

    def detect(self, image, mode, region_type, subject=None):
        info, img_cost = self._wordocr.detect(image, mode, region_type, subject)
        return info, img_cost

    def paging(self, image):
        info = self._wordocr.paging(image)
        return info


if __name__ == "__main__":
    triton_client_test = TritonClient()
    wordengine = WordEngine('tmp', triton_client_test)
    img = np.array(Image.open('5f3681c6030e24113b58d4a9.jpg'))
    a,b,c = wordengine.convert([img])
    print(a,b,c)
