# SELECT TASK
export TASK_NAME=SciArg
export MODELTYPE=bart

# PATH TO TRAINING DATA
export DATA_DIR=../../data/

# MAXIMUM SEQUENCE LENGTH
export MAXSEQLENGTH=512


for i in 42 44 46
do
export OUTPUTDIR=gen_bart_15epoch_cased_3e-5/seed_$i/$TASK_NAME+$MAXSEQLENGTH/
export MODEL=gen_bart_15epoch_cased_3e-5/seed_$i/$TASK_NAME+$MAXSEQLENGTH/

# CUDA_LAUNCH_BLOCKING=1
python3 ../run.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --with_graph \
  --seed $i \
  --save_steps 1000
done
