# -*-coding:utf-8-*-
import cv2, json, os
import subprocess
import sys
from doc_caculator import doc_caculator
from doc_caculator_v3 import doc_caculator_v3
from doc_caculator_errorquestion import doc_caculator_errorquestion

from ..html_generator.html_generator import html_generator
import time
import threading
import psutil
import traceback
class doc_builder:
    def __init__(self):
        self.doc_cal = doc_caculator()
        self.doc_cal_v3 = doc_caculator_v3()
        self.doc_cal_errorquestion = doc_caculator_errorquestion()
        self.html_gen = html_generator('word/html_generator/template.html')

        self.process = subprocess.Popen(['dotnet', "bin/imageToWordCuoTi/imageToWord.dll"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.process_lock = threading.Lock()

        self.process_v3 = subprocess.Popen(['dotnet', "bin/imageToWordV3/imageToWord.dll"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.process_lock_v3 = threading.Lock()


    def local_convert(self, img_file=None, img=None, json_file=None, meta_data=None, doc_file=None, work_dir='', create_doc=True):
        assert (img_file is not None) or (img is not None), "need image resource"
        if img_file is not None:
            img = cv2.imread(img_file)
            # 读取json数据
        assert json_file or meta_data, "need meta data resource"
        if json_file != None:
            with open(json_file) as f:
                meta_data = json.load(f)
        prefix = os.path.basename(doc_file).replace('docx','')
        meta_data = self.doc_cal.process(img, meta_data, prefix, work_dir=work_dir)
        self.doc_constructor.process(meta_data, doc_file)

    def oxml_convert_old(self, img_file=None, img=None, json_file=None, meta_data=None, doc_file=None, work_dir='', create_doc=True, paths_to_clear=[]):
        assert (img_file is not None) or (img is not None), "need image resource"
        if img_file is not None:
            img = cv2.imread(img_file)
            # 读取json数据
        assert json_file or meta_data, "need meta data resource"
        if json_file != None:
            with open(json_file) as f:
                meta_data = json.load(f)
        prefix = os.path.basename(doc_file).replace('docx', '')
        s = time.time()
        meta_data = self.doc_cal.process(img, meta_data, prefix, work_dir=work_dir)
        e = time.time()
        if create_doc:
            meta_data['doc_info']['doc_page_info']['create_doc'] = 1
        else:
            meta_data['doc_info']['doc_page_info']['create_doc'] = 0
        doc_pics = meta_data['doc_info']['doc_pics']
        for doc_pic in doc_pics:
            paths_to_clear.append(doc_pic['url'])
        if(len(meta_data['doc_info']['doc_tables'])!=0):
            print(doc_file)
        doc_info = json.dumps(meta_data['doc_info'], ensure_ascii=False)

        json_file = doc_file.replace('docx', 'json')
        with open(json_file, 'w') as f:
            f.write(doc_info)
        s = time.time()
        converter_path = os.path.abspath(os.path.join(work_dir, "bin/imageToWord/imageToWord.dll"))
        paths_to_clear.append(json_file)
        paths_to_clear.append(doc_file)
        cmd = "dotnet %s %s %s " % (converter_path, json_file, doc_file)
        status, ret = subprocess.getstatusoutput(cmd)
        e = time.time()
        return status, ret

    def oxml_convert(self, img_files=None, imgs=None, json_files=None, meta_datas=None, doc_file=None, work_dir='', paths_to_clear=[], generate_html=False, for_wrong_question=False, num_columns=1):
        assert (img_files is not None) or (imgs is not None), "need image resource"
        if img_files is not None:
            imgs = []
            for img_file in img_files:
                imgs.append(cv2.imread(img_file))
            # 读取json数据
        assert json_files or meta_datas, "need meta data resource"
        if json_files != None:
            meta_datas = []
            with open(json_files) as f:
                meta_datas.append(json.load(f))
        prefix = os.path.basename(doc_file).replace('docx', '')
        assert len(meta_datas)==len(imgs)
        doc_images_info = {'images_info':[]}
        json_file = doc_file.replace('docx', 'json')
        paths_to_clear.append(doc_file)
        paths_to_clear.append(json_file)
        if generate_html:
            html_file = doc_file.replace('.docx', '.html')
            paths_to_clear.append(html_file)
        index = 1
        for i in range(len(meta_datas)):

            for pic in meta_datas[i]['pics']:
                url = os.path.join(work_dir, 'tmp', prefix + '_pic' + '%03d.jpg' % index)
                pic['url'] = url
                paths_to_clear.append(url)
                index += 1
            for table in meta_datas[i]['tables']:
                for cell in table['cells']:
                    for content in cell['texts']:
                        if content['cls']!=3:
                            continue
                        url = os.path.join(work_dir, 'tmp', prefix + '_table_pic' + '%03d.jpg' % index)
                        content['url'] = url
                        paths_to_clear.append(url)
                        index += 1
        try:
            for i in range(len(imgs)):
                if for_wrong_question:
                    meta_data = self.doc_cal_errorquestion.process(imgs[i], meta_datas[i], prefix + 'img' + str(i),work_dir=work_dir, num_columns=num_columns)
                elif generate_html and not for_wrong_question:
                    meta_data = self.doc_cal_v3.process(imgs[i], meta_datas[i], prefix + 'img' + str(i), work_dir=work_dir)
                else:
                    meta_data = self.doc_cal.process(imgs[i], meta_datas[i], prefix + 'img' + str(i), work_dir=work_dir)
                del meta_data['image_info']['img']
                meta_data['doc_info']['img_info'] = meta_data['image_info']
                doc_images_info['images_info'].append(meta_data['doc_info'])
        except Exception as e:
            traceback.print_exc()
            return 1, "doc calculate error"
        doc_images_info = json.dumps(doc_images_info, ensure_ascii=False)
        with open(json_file, 'w') as f:
            f.write(doc_images_info)

        if generate_html:
            html_files_dir = html_file.replace('.html', '.files')
            try:
                self.html_gen.process(json_file=json_file, html_file=html_file)
            except Exception as e:
                traceback.print_exc()
                html_files_dir = html_file.replace('.html', '.files')
                for file in os.listdir(html_files_dir):
                    paths_to_clear.append(os.path.join(html_files_dir, file))
                paths_to_clear.append(html_files_dir)
                return 1, "generate html error"
            if not os.path.exists(html_file):
                return 1, "html file dont exists"
            for file in os.listdir(html_files_dir):
                paths_to_clear.append(os.path.join(html_files_dir, file))
            paths_to_clear.append(html_files_dir)

        if for_wrong_question:
            message = json_file + ';' + doc_file + ';' + str(id) + '\n'
            pid = self.process.pid
            process = psutil.Process(pid)
            status = process.status()
            if status not in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
                print('start new process:', status)
                os.system('kill -9 ' + str(pid))
                self.process = subprocess.Popen(['dotnet', "bin/imageToWordCuoTi/imageToWord.dll"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            current_process = self.process
            self.process_lock.acquire()
            current_process.stdin.write(message.encode('utf-8'))
            current_process.stdin.flush()
            result = current_process.stdout.readline().decode('utf-8').strip()
            self.process_lock.release()
            if result=="" or result=="failed":
                return 1, "c# code run failed"
            return 0, 'success'
        else:
            # converter_path = os.path.abspath(os.path.join(work_dir, "bin/imageToWordV3/imageToWord.dll"))
            # cmd = "dotnet %s %s %s " % (converter_path, json_file, doc_file)
            # status, ret = subprocess.getstatusoutput(cmd)
            # print('time:', time.time() - s)
            # return status, ret
            message = json_file + ';' + doc_file + ';' + str(id) + '\n'
            pid = self.process_v3.pid
            process = psutil.Process(pid)
            status = process.status()
            if status not in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
                print('images to word v3 start new process:', status)
                os.system('kill -9 ' + str(pid))
                self.process_v3 = subprocess.Popen(['dotnet', "bin/imageToWordV3/imageToWord.dll"],
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            current_process = self.process_v3
            self.process_lock_v3.acquire()
            current_process.stdin.write(message.encode('utf-8'))
            current_process.stdin.flush()
            result = current_process.stdout.readline().decode('utf-8').strip()

            self.process_lock_v3.release()
            if result == "" or result == "failed":
                return 1, "c# code run failed"
            return 0, 'success'


if __name__ == "__main__":
    db = doc_builder()
    assert len(sys.argv) >= 4, len(sys.argv)
    image_file = sys.argv[1]
    json_file = sys.argv[2]
    doc_file = sys.argv[3]
    create_doc = True
    db.oxml_convert(img_file=None, img=None, json_file=None, meta_data=None, doc_file=None, work_dir='/home/ateam/xychen/projects/image_2_word/word',
                 create_doc=True, paths_to_clear=[])
