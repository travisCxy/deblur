train_gpu=7
val_gpu=0

input_dir="/mnt/data2/hwhelper_train/data/seq_pill"
output_dir="/mnt/data2/hwhelper_train/data/seq_pill/"
pretrained_dir="/mnt/data1/pretrain_models/v1d_cascade/"


#format to system style params
input_dirs="1:/mnt/data2/1111/1;1:$pretrained_dir;1:$input_dir"
output_dir="1:$output_dir"

export CUDA_VISIBLE_DEVICES=$val_gpu,$train_gpu
sh run_system.sh $1 $2 -train_gpu $train_gpu -val_gpu $val_gpu -l $input_dirs -o $output_dir
