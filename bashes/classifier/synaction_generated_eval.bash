#!/usr/bin/env bash

cd classifier/

EXP_NAME='synaction_classifier'

DATAROOT=/research/sunxm/mocogan/synaction/results/

CKPT_PATH=/research/sunxm/classifier/synaction/best_model.pth.tar


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset Generated_video \
    --dataroot  ${DATAROOT} \
    --ckpt_path ${CKPT_PATH} \
    --test_dataset SynAction \
    --test_dataroot /scratch/dataset/synaction_npy2/ \
    --textroot ../videolist/SynAction/ \
