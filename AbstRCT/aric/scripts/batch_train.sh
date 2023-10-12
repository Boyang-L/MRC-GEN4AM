# SELECT TASK
export TASK_NAME=SciArg
export MODELTYPE=bart

# PATH TO TRAINING DATA
export DATA_DIR=../../data/neo/

# MAXIMUM SEQUENCE LENGTH
export MAXSEQLENGTH=768

# SELECT MODEL FOR FINE-TUNING
export MODEL=GanjinZero/biobart-v2-base

export AUG_TYPE=td
export GRAPH_REP_TYPE=path
export WEIGHTS=0.7

for i in 42 44 46
do
export OUTPUTDIR=gen_bart_15epoch_cased_8e-5/seed_$i/$TASK_NAME+$MAXSEQLENGTH/

CUDA_LAUNCH_BLOCKING=1 python3 ../run.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --do_train \
  --evaluate_during_training \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 8e-5 \
  --seed $i \
  --num_train_epochs 15 \
  --warm_start \
  --with_graph \
  --graph_rep_type $GRAPH_REP_TYPE \
  --data_aug $AUG_TYPE \
  --main_weights $WEIGHTS \
  --save_steps 100000
done
