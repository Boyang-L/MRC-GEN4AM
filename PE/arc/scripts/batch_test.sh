# SELECT TASK
export TASK_NAME=SciArg
export MODELTYPE=bart

# PATH TO TRAINING DATA
export DATA_DIR=../../data/

# MAXIMUM SEQUENCE LENGTH
export MAXSEQLENGTH=512
export AUG_TYPE=none
export GRAPH_REP_TYPE=path

for i in 42 44 46
do
export OUTPUTDIR=gen_bart_20epoch_cased_3e-5_path_0.7/seed_$i/$TASK_NAME+$MAXSEQLENGTH/
export MODEL=gen_bart_20epoch_cased_3e-5_path_0.7/seed_$i/$TASK_NAME+$MAXSEQLENGTH/

#CUDA_LAUNCH_BLOCKING=1 
python3 ../run.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --seed $i \
  --warm_start \
  --with_graph \
  --graph_rep_type $GRAPH_REP_TYPE \
  --save_steps 1000
done
