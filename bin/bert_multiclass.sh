#! /bin/bash

cd bin/transformers/examples
source your_path/bin/activate your_path/envs/transformers

event=$1
train_file=$2
dev_file=$3
test_file=$4
results_file=$5


num_epoch=3.0
export HOME_DIR="$PWD/"
export TASK_NAME=multiclass
model=bert-base-uncased
outputDir=$HOME_DIR/output_multi_class_${model}_$event
cache_dir=$HOME_DIR/exp_cache
export data_dir=$HOME_DIR"/data_bert_model_$event/"
mkdir -p $data_dir
mkdir -p $outputDir


python run_glue_multiclass.py --model_type bert --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_train --do_eval --do_lower_case \
    --data_dir $data_dir --max_seq_length 128 --per_gpu_eval_batch_size=8  --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs $num_epoch \
    --train_file $HOME_DIR/$train_file --dev_file $HOME_DIR/$dev_file \
    --output_dir $outputDir --overwrite_output_dir 

rm $data_dir/cached_test_bert-base-uncased_128_multiclass

python run_glue_multiclass.py --model_type bert --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_test --do_lower_case --data_dir $data_dir \
--max_seq_length 128 --per_gpu_eval_batch_size=8  --per_gpu_train_batch_size=8   --learning_rate 2e-5 --num_train_epochs $num_epoch \
--test_file $HOME_DIR/$dev_file  --output_dir $outputDir --results_file $results_file"_dev.txt"

rm $data_dir/cached_test_bert-base-uncased_128_multiclass

python run_glue_multiclass.py --model_type bert --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_test --do_lower_case --data_dir $data_dir \
--max_seq_length 128 --per_gpu_eval_batch_size=8  --per_gpu_train_batch_size=8   --learning_rate 2e-5 --num_train_epochs $num_epoch \
--test_file $HOME_DIR/$test_file  --output_dir $outputDir --results_file $results_file"_test.txt"

