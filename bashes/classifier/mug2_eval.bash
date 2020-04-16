#!/usr/bin/env bash

cd classifier/

EXP_NAME='mug2_classifier'

DATAROOT=/research/sunxm/dataset/MUG_facial_expression/

CKPT_PATH=/research/sunxm/classifier/checkpoints/mug2_classifier/best_model.pth.tar


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset MUG2 \
    --dataroot ${DATAROOT} \
    --textroot ../videolist/MUG/ \
    --ckpt_path ${CKPT_PATH} \
    --every_nth 1 \
