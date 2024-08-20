project=$1
train_gpu=$2
train_gpu_num=$3
val_gpu=$4
input_dirs=$5
output_dir=$6

echo "prepare_system.sh project: "$project
echo "prepare_system.sh train_gpu: "$train_gpu
echo "prepare_system.sh train_gpu_num: "$train_gpu_num
echo "prepare_system.sh val_gpu: "$val_gpu
echo "prepare_system.sh input_dirs: "$input_dirs
echo "prepare_system.sh output_dir: "$output_dir

export PYTHONPATH=./:../
python3.7 preprocess/data_prepare_"$project"_system.py --project=$project --input_dirs=$input_dirs --output_dir=$output_dir
