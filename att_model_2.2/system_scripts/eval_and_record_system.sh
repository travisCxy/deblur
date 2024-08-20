project=$1
train_gpu=$2
train_gpu_num=$3
val_gpu=$4
input_dirs=$5
output_dir=$6

echo "eval_and_record_system.sh project: "$project
echo "eval_and_record_system.sh train_gpu: "$train_gpu
echo "eval_and_record_system.sh train_gpu_num: "$train_gpu_num
echo "eval_and_record_system.sh val_gpu: " $val_gpu
echo "eval_and_record_system.sh input_dirs: "$input_dirs
echo "eval_and_record_system.sh output_dir: "$output_dir


train_dir="$output_dir"models_equ
record_path="$output_dir"record.txt
save_path="$output_dir"save_model
val_data_path="$output_dir""tfrecords/val\*"

python3.7 system_scripts/eval_and_record.py -p $project -m $train_dir -r $record_path -s $save_path -d $val_data_path -g $val_gpu
