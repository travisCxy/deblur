import os
import time
import subprocess
from sys import argv, stdout
from shutil import copyfile
import sys


def _create_cmd_line(project, model_path, model_index, data_path, gpu):
    cmd = "export CUDA_VISIBLE_DEVICES=" + gpu + "\n"
    cmd = cmd + "python3 ./eval.py"
    cmd = cmd + " --checkpoint_file=" + model_path + "/ckpt-" + str(model_index)
    cmd = cmd + " --val_data_path=" + data_path
    cmd = cmd + " --project=" + project
    return cmd


def _list_model_indexes(model_dir):
    indexes = []
    if not os.path.exists(model_dir):
        return indexes
    
    names = os.listdir(model_dir)
    for name in names:
        if "764001" in name:
            continue
        if "ckpt-" in name and ".index" in name:
            index = name[5: -6]
            indexes.append(int(index))
    indexes.sort()
    return indexes[::-1]


def _parse_record(record):
    if 'All:' not in record or 'Step:' in record:
        return 0.0
    else:
        index = record.rfind(" ")
        score = record[index:]
        return float(score)


def _get_max_record(record_path):
    records = []
    records.append(0.0)
    with open(record_path, "r") as record_file:
        for line in record_file:
            records.append(_parse_record(line.rstrip()))
    return max(records)


def _save_models(index, src_dir, dest_dir, record_path):
    files = os.listdir(dest_dir)
    for name in files:
        os.remove(os.path.join(dest_dir, name))
    files=[]
    files.append("ckpt-%d.data-00000-of-00001"%index)
    #files.append("ckpt-%d.data-00001-of-00002"%index)
    files.append("ckpt-%d.index"%index)
    for name in files:
        copyfile(os.path.join(src_dir, name), os.path.join(dest_dir, name))
    cmd = 'less ' + record_path + ' | grep ' + str(index) + ' > ' + os.path.join(dest_dir, 'record.txt')
    os.system(cmd)


def _load_last_record_index(record_path):
    last_line = None
    with open(record_path, "r") as record_file:
        while True:
            line = record_file.readline()
            if line == "":
                break
            last_line = line.strip()
    if last_line == None or last_line == "":
        return -1
    else:
        return int(str(last_line).split(":")[0])


def _save_record(record_path, model_index, result):
    with open(record_path, "a+") as record_file:
        line = str(model_index) + ":" + result + "\n"
        record_file.write(line)


def _empty_record_file(record_path):
    with open(record_path, "a") as record_file:
        pass


def getopts(argv):
    opts = {}  
    while argv:  
        if argv[0][0] == '-': 
            opts[argv[0]] = argv[1]  
        argv = argv[1:]  
    return opts


def get_result_lines(result):
    all_lines = str(result).split("\n")
    flag_line = all_lines[-1]
    flags = flag_line.split(" ")
    if len(flags) != 2:
        print("flags format error:  get", flag_line)
        sys.stdout.flush()
    show_lines = int(flags[0])
    current_score = float(flags[1])
    
    return current_score, all_lines[-show_lines: -1]


if __name__ == "__main__":
    #myargs = getopts(argv)
    #model_path = myargs['-m']
    #record_path = myargs['-r']
    #save_path = myargs['-s']
    #data_path = myargs['-d']
    #gpu = myargs['-g']
    #project = myargs['-p']

    save_path = "./save_model_latex"
    model_path = "/mnt/server_data2/data/seq_chemical/models_equ_latex1"
    record_path = "./record_latex.txt"
    project = 'cht'
    gpu = "4"

    data_path = "/mnt/server_data2/data/seq_chemical/tfrecords_20240712_latex/val*"



    if not os.path.exists(save_path):
        os.makedirs(save_path)    
            
    print('model_path', model_path)
    print('record_path', record_path)
    print('save_path', save_path)
    print('data_path', data_path)
    print('gpu', gpu)

    stdout.flush()
    
    _empty_record_file(record_path)
    max_score = -1.0
    while True:   
        indexes = _list_model_indexes(model_path)
        print(indexes)
        recorded_index = _load_last_record_index(record_path)
        next_index = next((x for x in indexes if x>recorded_index), -1)
        if next_index != -1:
            cmd = _create_cmd_line(project, model_path, next_index, data_path, gpu)
            print("run cmd: ", cmd)
            stdout.flush()
            status, result = subprocess.getstatusoutput(cmd)
            print(result)
            if status == 0:
                current_score, result_lines = get_result_lines(result)

                for result_line in result_lines:
                    _save_record(record_path, next_index, result_line)
                    print("result:", result_line)

                if max_score >= current_score:
                    print(next_index, " score ", current_score, "max_score", max_score)
                else:
                    # try:
                    max_score = current_score
                    _save_models(next_index, model_path, save_path, record_path)
                    print("save new model index", next_index, "with score ", current_score)
                    # except:
                    #         pass
            else:
                run_result = "0 0 0.00"
                _save_record(record_path, next_index, run_result)
                print("result:", run_result)
        else:
            print("no new models")
        stdout.flush()
        time.sleep(60)
