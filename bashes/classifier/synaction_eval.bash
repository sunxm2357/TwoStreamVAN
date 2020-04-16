#!/usr/bin/env bash

cd classifier/

EXP_NAME='synAction_classifier_3'

DATAROOT=/scratch/dataset/synaction_npy2

CKPT_PATH=/research/sunxm/classifier/synaction/best_model.pth.tar


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset SynAction \
    --dataroot ${DATAROOT} \
    --textroot ../videolist/SynAction/ \
    --ckpt_path ${CKPT_PATH} \
    --every_nth 1 \
