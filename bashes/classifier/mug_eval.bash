#!/usr/bin/env bash

cd classifier/

EXP_NAME='mug_classifier'

DATAROOT=/scratch/dataset/MUG_facial_expression/

CKPT_PATH=/research/sunxm/classifier/mug/best_model.pth.tar


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset MUG \
    --dataroot ${DATAROOT} \
    --textroot ../videolist/MUG/ \
    --ckpt_path ${CKPT_PATH} \
    --every_nth 1 \
