#!/bin/bash

RECORD=mix_b
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/mix/test/test_mix_b.yaml

WEIGHTS=/home/niyunfei/workspace/wuRenji/wuRenji/ckpt/train_mixformer_bone-36-19314.pt

BATCH_SIZE=32

python3 main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
