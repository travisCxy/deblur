project=$1
train_gpu=$2
train_gpu_num=$3
val_gpu=$4
input_dirs=$5
output_dir=$6

echo "export_system.sh project: "$project
echo "export_system.sh train_gpu: "$train_gpu
echo "export_system.sh train_gpu_num: "$train_gpu_num
echo "export_system.sh val_gpu: "$val_gpu
echo "export_system.sh input_dirs: "$input_dirs
echo "export_system.sh output_dir: "$output_dir

model_dir="$output_dir"save_model/
model_name=$(ls $model_dir | grep "index" | cut -d '.' -f 1,2)
echo $model_name
export_record="$model_dir"record.txt
cat $export_record
record_line="$output_dir"record.txt

checkpoint_file="$model_dir"$model_name
export_dir="$output_dir""export/"

echo "checkpoint: $checkpoint_file"
echo "export to: $export_dir"

export CUDA_VISIBLE_DEVICES=-1

python3.7 export.py --project=$project --checkpoint_file=$checkpoint_file #--export_dir=$export_dir

cp $export_record "$export_dir"/export_record.txt
cp $record_line "$export_dir"/record_line.txt
