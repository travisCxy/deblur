project=$1
train_gpu=$2
train_gpu_num=$3
val_gpu=$4
input_dirs=$5
output_dir=$6
pretrained_model=$7

echo "train_system.sh project: "$project
echo "train_system.sh train_gpu: "$train_gpu
echo "train_system.sh train_gpu_num: "$train_gpu_num
echo "train_system.sh val_gpu: "$val_gpu
echo "train_system.sh input_dirs: "$input_dirs
echo "train_system.sh output_dir: "$output_dir

train_dir="$output_dir"models_equ
train_data_path="$output_dir"tfrecords/*train*
val_data_path="$output_dir"tfrecords/val*
record_path="$output_dir"record.txt
save_path="$output_dir"save_model
pretrained_model="$pretrained_model"model.ckpt-360000


echo "Start to training with train output dir $train_dir"
export CUDA_VISIBLE_DEVICES=$train_gpu

horovodrun -np $train_gpu_num -H localhost:$train_gpu_num python3.7 train.py --project=$project \
    --model_dir=$train_dir --data_dir=$train_data_path

