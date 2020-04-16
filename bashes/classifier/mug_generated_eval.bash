#!/usr/bin/env bash

cd classifier/

EXP_NAME='mug_classifier'

DATAROOT=//research/sunxm/outer_prod_motion_mask/results/mug_2/evaluation/500000
#DATAROOT=/research/sunxm/mocogan/mug_16/results


CKPT_PATH=/research/sunxm/classifier/mug/best_model.pth.tar


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset Generated_video \
    --dataroot  ${DATAROOT} \
    --ckpt_path ${CKPT_PATH} \
    --test_dataset MUG \
    --test_dataroot /scratch/dataset/MUG_facial_expression/ \
    --textroot ../videolist/MUG/ \
