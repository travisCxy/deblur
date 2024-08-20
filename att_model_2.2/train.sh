#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
#horovodrun -np 8 -H localhost:8 python train.py --project=uni --model_dir=/mnt/server_data/data/sequni/models_equ_cht  --data_dir=/mnt/server_data/data/sequni/tfrecords_full/train*
#horovodrun -np 4 -H localhost:4 python train.py --project=cht --model_dir=/mnt/server_data/data/sequni/models_equ_cht_new  --data_dir=/mnt/server_data/data/sequni/tfrecords_full/train*
#horovodrun -np 8 -H localhost:8 python train.py --project=cht --model_dir=/mnt/server_data/data/sequni/models_equ_cht_new  --data_dir=/mnt/server_data/data/sequni/tfrecords_full/train*
#horovodrun -np 4 -H localhost:4 python train.py --project=cht --model_dir=/mnt/server_data/data/sequni/models_equ_cht_new  --data_dir=/mnt/server_data/data/sequni/tfrecords_full/train*
#horovodrun -np 4 -H localhost:4 python train.py --project=cht --model_dir=/mnt/server_data/data/sequni/models_equ_cht_new  --data_dir=/mnt/server_data/data/sequni/tfrecords_full/train*
#horovodrun -np 4 -H localhost:4 python train.py --project=uni --model_dir=/mnt/server_data/data/sequni/models_equ_new  --data_dir=/mnt/server_data/data/sequni/tfrecords_full/train*
#horovodrun -np 4 -H localhost:4 python train.py --project=uni --model_dir=/mnt/server_data/data/sequni/models_equ_cht --data_dir=/home/ateam/xychen/data/val*


#horovodrun -np 7 -H localhost:8 python train.py --project=cht --model_dir=/mnt/server_data2/data/seq_chemical/models_equ_latex0802  --data_dir=/mnt/server_data2/data/seq_chemical/tfrecords_20240712_latex/train*
horovodrun -np 5 -H localhost:8 python train.py --project=cht --model_dir=/mnt/server_data2/data/seq_latex/models_equ_latex_0808  --data_dir=/mnt/server_data2/data/seq_latex/tfrecords_0809/train*
