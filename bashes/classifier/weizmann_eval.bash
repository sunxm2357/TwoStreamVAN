#!/usr/bin/env bash

cd classifier/

EXP_NAME='weizmann_classifier'

if [ "${1}" == "--is_actor" ]
then
    EXP_NAME+='_actor'
    MODEL_IS_ACTOR='--model_is_actor'
    DATA_IS_ACTOR='--data_is_actor'
    CKPT_PATH='/research/sunxm/classifier/weizmann/checkpoints/actorAction_best_model.pth.tar'
else
    MODEL_IS_ACTOR=' '
    DATA_IS_ACTOR=' '
    CKPT_PATH='/research/sunxm/classifier/weizmann/checkpoints/action_best_model.pth.tar'
fi


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset Weizmann \
    --dataroot  /scratch/dataset/Weizmann_crop_npy_new2/  \
    --textroot ../videolist/Weizmann/ \
    --ckpt_path ${CKPT_PATH} \
    ${MODEL_IS_ACTOR} \
    ${DATA_IS_ACTOR}
