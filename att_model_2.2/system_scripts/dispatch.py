import sys
import os
import subprocess
import base64
import json

def check_opt_in_opts(opts, name, desc):
    value = opts.get(name, '')
    if value == '':
        print(desc)
        return False
    return True

def check_opts(opts):
    params = [['-l', 'no input dirs'],
              ['-o', 'no output dirs'],
              ['-train_gpu', 'no train gpu'],
              ['-val_gpu', 'no val gpu']]
    for param in params:
        if not check_opt_in_opts(opts, param[0], param[1]):
            return False
        
    return True

def params_to_opts(args, opts):
    try:
        while args:  
            if args[0][0] == '-':
                opts[args[0]] = args[1]
            args = args[1:]
    except:
        print('parse paramters error')
    

def parse_opts(args):
    opts = {}
    params_to_opts(args, opts)
    ext_params = opts.get('-p', '')
    if ext_params != '':
        ext_params = base64.b64decode(ext_params)
        ext_params = json.loads(ext_params)
        for key in ext_params:
            opts['-' + key] = ext_params[key]
    return opts     

def parse_dirs(dirs_str):
    raw_dirs = dirs_str.split(';')
    ret = ""
    for d in raw_dirs:
        ret = ret + d.split(':')[1] + ';'
    return ret[0: -1]

def execute(command):
    print('execute command: ')
    
    print('===============================================================')
    print(command)
    print('===============================================================')
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        nextline = process.stdout.readline()
        nextline = nextline.decode('utf-8')
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0]
    exitCode = process.returncode

    if (exitCode == 0):
        return output
    else:
        return None
    
def parse_gpu_setting(opts):
    env = os.environ
    gpus = env.get('CUDA_VISIBLE_DEVICES', None)
    if gpus is None:
        return
    ind = str(gpus).index(',')
    val_gpu = gpus[:ind]
    tran_gpu = gpus[ind + 1:]
    opts['-train_gpu'] = tran_gpu
    opts['-val_gpu'] = val_gpu

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    
    action = "train"
    project = None
    if len(args) > 0:
        if args[0][0] != '-':
            action = args[0]
            args = args[1:]
            
            if args[0][0] != '-':
                project = args[0]
                args = args[1:]

    if project not in ["hw", "ch", "uni", "equ", "ml", "sl", "pill"]:
        msg = "project: %s not supported"%project
        print(msg)
        sys.stdout.flush()
        raise ValueError(msg)

    opts = parse_opts(args)
    parse_gpu_setting(opts)
    print(opts)
    
    if check_opts(opts):
        train_gpu = opts.get('-train_gpu')
        train_gpu_num = len(train_gpu.split(','))
        val_gpu = opts.get('-val_gpu')
        input_dirs = opts.get('-l')
        output_dir = opts.get('-o')

        input_dirs = parse_dirs(input_dirs)
        output_dir = parse_dirs(output_dir)
        pretrained_dir=input_dirs.split(";")[1]

        params = " '%s' '%s' '%s' '%s' '%s' '%s' '%s'"%(project, train_gpu, train_gpu_num, val_gpu, input_dirs, output_dir, pretrained_dir)
        if action == 'prepare':
            execute("sh system_scripts/prepare_system.sh" + params)
        elif action == 'train':
            execute("sh system_scripts/train_system.sh" + params)
        elif action == 'prepareAndTrain':
            execute("sh system_scripts/prepare_and_train_system.sh" + params)
        elif action == 'eval':
            execute("sh system_scripts/eval_and_record_system.sh" + params)
        elif action == 'export':
            execute("sh system_scripts/export_system.sh" + params)
        else:
            print('unknown command')
    print('Scripts finished')
