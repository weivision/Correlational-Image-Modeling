#!/usr/bin/bash
echo '[USAGE] sh scripts/dist_finetune.sh <GPU_NUM> <PORT> <MODEL_TYPE> <CKPT> <RESUME> <JOB_NAME>'
set -x


GPU_NUM=$1
PORT=$2
DATA=$3
MODEL_TYPE=$4
CKPT=$5
RESUME=$6
JOB_NAME=$7

PRETRAIN_CKPT=$CKPT

EXPS='exps'
if ! [ -d "$EXPS/$JOB_NAME" ]; then
   mkdir -p $EXPS/$JOB_NAME
fi


case $MODEL_TYPE in
   "tiny")
      MODEL="vit_tiny_patch16"
      EPOCHS=200
      ;;
   "small")
      MODEL="vit_small_patch16"
      EPOCHS=200
      ;;
   "base")
      MODEL="vit_base_patch16"
      EPOCHS=100
      ;;
   "large")
      MODEL="vit_large_patch16"
      EPOCHS=50
      ;;
   "huge")
      MODEL="vit_huge_patch14"
      EPOCHS=50
      ;;
   *)
esac



EXT_FLAGS="$EXT_FLAGS --batch_size 256 --accum_iter 1"



EXT_FLAGS="$EXT_FLAGS --num_workers 12"



if [ $CKPT = "none" ]; then
   EXT_FLAGS="$EXT_FLAGS"
else
   EXT_FLAGS="$EXT_FLAGS --finetune $PRETRAIN_CKPT"
fi


if [ "$RESUME" = "none" ]
then
   EXT_FLAGS="$EXT_FLAGS"
else
   EXT_FLAGS="$EXT_FLAGS --resume $RESUME"
fi


IMAGENET_DIR=$DATA


export PYTHONPATH=./:$PYTHONPATH

OMP_NUM_THREADS=1 torchrun --master_addr 127.0.0.1 --master_port $PORT --nproc_per_node $GPU_NUM \
    tools/main_finetune.py \
        --output_dir $EXPS/$JOB_NAME \
        --log_dir $EXPS/$JOB_NAME \
        --port $PORT \
        --model $MODEL \
        --epochs $EPOCHS \
        --save_per_epochs 1 \
        --blr 1.2e-3 --layer_decay 0.8 \
        --weight_decay 0.05 --drop_path 0.1 \
        --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
        --dist_eval --data_path ${IMAGENET_DIR} \
        $EXT_FLAGS \
        2>&1 | tee $EXPS/$JOB_NAME/$JOB_NAME.log > /dev/null & 

echo "tail -f $EXPS/$JOB_NAME/$JOB_NAME.log"
