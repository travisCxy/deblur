import os
import time
import subprocess
from sys import argv, stdout
from shutil import copyfile
import sys


def _create_cmd_line(project, model_path, model_index, data_path, gpu, test_cht=False):
    cmd = "export CUDA_VISIBLE_DEVICES=" + gpu + "\n"
    cmd = cmd + "python3 ./eval.py"
    cmd = cmd + " --checkpoint_file=" + model_path + "/ckpt-" + str(model_index)
    cmd = cmd + " --val_data_path=" + data_path
    cmd = cmd + " --project=" + project
    if test_cht:
        cmd = cmd + " --test_cht=1"
    return cmd


def _list_model_indexes(model_dir):
    indexes = []
    if not os.path.exists(model_dir):
        return indexes

    names = os.listdir(model_dir)
    for name in names:
        if "ckpt-" in name and ".index" in name:
            index = name[5: -6]
            if index == "764001":
                continue
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
    files = []
    files.append("ckpt-%d.data-00000-of-00001" % index)
    # files.append("ckpt-%d.data-00001-of-00002"%index)
    files.append("ckpt-%d.index" % index)
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


def get_cht_result_lines(result):
    all_lines = str(result).split("\n")
    flag_line = all_lines[-1]
    flags = flag_line.split(" ")
    if len(flags) != 2:
        print("flags format error:  get", flag_line)
        sys.stdout.flush()
    show_lines = int(flags[0])

    score_line = all_lines[-2].split(" ")
    current_score = float(score_line[1])

    return current_score, all_lines[-show_lines: -1]


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
    # save_path = "./save_model"
    # model_path = "/mnt/server_data/data/sequni/models_equ_new"
    # record_path = "./record.txt"
    # gpu = "3"

    save_path = "./save_model_chemical"
    model_path = "/mnt/server_data/data/seq_chemical/models_equ"
    record_path = "./record_chemical.txt"
    gpu = "0"
    project = 'cht'

    cht_data_path = "/mnt/server_data/data/seq_chemical/tfrecords_mini/val_cht*"
    data_path = "/mnt/server_data/data/seq_chemical/tfrecords_mini/val*"
    chemical_data_path = "/mnt/server_data/data/seq_chemical/tfrecords_mini/val_chemical*"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('model_path', model_path)
    print('record_path', record_path)
    print('save_path', save_path)
    print('data_path', data_path)
    print('cht_data_path', cht_data_path)
    print('gpu', gpu)
    # print('project', project)
    stdout.flush()
    import pdb
    _empty_record_file(record_path)
    max_score = 80

    while True:
        indexes = _list_model_indexes(model_path)
        recorded_index = _load_last_record_index(record_path)
        next_index = next((x for x in indexes if x > recorded_index), -1)
        # for next_index in indexes[::-1]:
        #     if next_index<120000:
        #         continue
        if next_index != -1:
            cmd = _create_cmd_line(project, model_path, next_index, data_path, gpu, test_cht=False)
            print("run cmd: ", cmd)
            stdout.flush()
            status, result = subprocess.getstatusoutput(cmd)
            print(result)

            cmd = _create_cmd_line(project, model_path, next_index, cht_data_path, gpu, test_cht=True)
            print("run cmd: ", cmd)
            status, cht_result = subprocess.getstatusoutput(cmd)
            print(cht_result)

            # cmd = _create_cmd_line(project, model_path, next_index, chemical_data_path, gpu, test_cht=False)
            # print("run cmd: ", cmd)
            # status, chemical_result = subprocess.getstatusoutput(cmd)
            # print(chemical_result)
            if status == 0:
                ch_current_score, result_lines = get_result_lines(result)
                cht_current_score, cht_result_lines = get_cht_result_lines(cht_result)
                #chemical_current_score, chemical_result_lines = get_result_lines(chemical_result)
                result_lines[-1] += cht_result_lines[-1]
                #result_lines[-1] += chemical_result_lines[-1]
                for result_line in result_lines:
                    _save_record(record_path, next_index, result_line)
                    print("result:", result_line)
                current_score = (ch_current_score * 0.4 + cht_current_score * 0.25)# + chemical_current_score*0.35)
                if max_score >= current_score or ch_current_score < 86 or cht_current_score < 85:
                    print(next_index, " score ", current_score, "max_score", max_score)
                else:
                    # try:
                    max_score = current_score
                    _save_models(next_index, model_path, save_path, record_path)
                    print("save new model index:", next_index, "with score:", current_score, 'ch score:',
                          ch_current_score, 'cht score:', cht_current_score)#, 'chemical score:', chemical_current_score)
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
