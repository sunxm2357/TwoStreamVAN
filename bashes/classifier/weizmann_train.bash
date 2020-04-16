#!/usr/bin/env bash

cd classifier/

EXP_NAME='weizmann_classifier'

if [ "${1}" == "--is_actor" ]
then
    EXP_NAME+='_actor'
    MODEL_IS_ACTOR='--model_is_actor'
    DATA_IS_ACTOR='--data_is_actor'
    LR=0.0001
    DECAY=0.999
else
    MODEL_IS_ACTOR=' '
    DATA_IS_ACTOR=' '
    LR=0.0005
    DECAY=0.995
fi

if [ "${2}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

python train.py \
    --exp_name ${EXP_NAME} \
    --dataset Weizmann \
    --dataroot  /scratch4/datasets/Weizmann_crop_npy_new2 \
    --textroot ../videolist/Weizmann/ \
    --lr ${LR} \
    --decay ${DECAY} \
    --crop \
    ${MODEL_IS_ACTOR} \
    ${DATA_IS_ACTOR} \
    ${RESUME} \


