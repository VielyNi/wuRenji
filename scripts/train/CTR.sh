#!/bin/bash
RECORD=ctr_jb
CONFIG=./config/ctr/train/train_ctr_jb.yaml


WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD
START_EPOCH=0
EPOCH_NUM=60
BATCH_SIZE=64
WARM_UP=5
SEED=777

python3 main_gcl.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED
