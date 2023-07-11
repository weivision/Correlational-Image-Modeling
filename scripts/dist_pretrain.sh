#!/usr/bin/bash
echo "[USAGE] sh scripts/dist_pretrain.sh <GPU_NUM> <PORT> <DATA> <GEAR> <MODEL> <RESUME> <JOB_NAME>"
set -x

GPU_NUM=$1
PORT=$2
DATA=$3
GEAR=$4
MODEL_NAME=$5
RESUME=$6
JOB_NAME=$7

EPOCHS=300
WEPOCHS=40


case $MODEL_NAME in
   "tiny")
      MODEL="vit_tiny_patch16"
      ;;
   "small")
      MODEL="vit_small_patch16"
      ;;
   "base")
      MODEL="vit_base_patch16"
      ;;
   "large")
      MODEL="vit_large_patch16"
      ;;
   "huge")
      MODEL="vit_huge_patch14"
      ;;
   *)
esac


case $GEAR in
   "mae")
      EXT_FLAGS="--epochs $EPOCHS --warmup_epochs $WEPOCHS --blr 1.5e-4 --weight_decay 0.05 --norm_pix_loss"
      ;;
   "cim")
      EXT_FLAGS="--epochs $EPOCHS --warmup_epochs $WEPOCHS"
      EXT_FLAGS="$EXT_FLAGS --context_min_scale 0.5 --rotaton_max_degree 45"
      EXT_FLAGS="$EXT_FLAGS --input_size 224 --context_size 160 --template_size 64 --template_num 6"
      EXT_FLAGS="$EXT_FLAGS --common_aug --template_aug"
      EXT_FLAGS="$EXT_FLAGS --template_min_scale 0.05 --template_max_scale 0.48"
      EXT_FLAGS="$EXT_FLAGS --blr 1.5e-4 --weight_decay 0.05 --clip_grad 1.0"
      EXT_FLAGS="$EXT_FLAGS --sigma_cont 0.0 --sigma_corr 1.0"
      ;;
   *)
esac



EXT_FLAGS="$EXT_FLAGS --num_workers 12"


if [ "$RESUME" = "none" ]
then
   EXT_FLAGS="$EXT_FLAGS"
else
   EXT_FLAGS="$EXT_FLAGS --resume $RESUME"
fi


EXPS='exps'
if ! [ -d "$EXPS/$JOB_NAME" ]; then
   mkdir -p $EXPS/$JOB_NAME
fi


EXT_FLAGS="$EXT_FLAGS --batch_size 256 --accum_iter 2"



IMAGENET_DIR=$DATA


export PYTHONPATH=./:$PYTHONPATH
OMP_NUM_THREADS=1 torchrun --master_addr 127.0.0.1 --master_port $PORT --nproc_per_node $GPU_NUM \
    tools/main_pretrain.py \
            --gear $GEAR \
            --output_dir $EXPS/$JOB_NAME \
            --log_dir $EXPS/$JOB_NAME \
            --port $PORT \
            --model $MODEL \
            --epochs $EPOCHS \
            --warmup_epochs $WEPOCHS \
            --data_path ${IMAGENET_DIR} \
            $EXT_FLAGS \
        2>&1 | tee $EXPS/$JOB_NAME/$JOB_NAME.log > /dev/null & 

echo "tail -f $EXPS/$JOB_NAME/$JOB_NAME.log"
