#!/usr/bin/env bash

cd classifier/

EXP_NAME='weizmann_classifier'

if [ "${1}" == "--model_is_actor" ]
then
    EXP_NAME+='_actor'
    MODEL_IS_ACTOR='--model_is_actor'
    CKPT_PATH='/research/sunxm/classifier/weizmann/checkpoints/actorAction_best_model.pth.tar'
else
    MODEL_IS_ACTOR=' '
    CKPT_PATH='/research/sunxm/classifier/weizmann/checkpoints/action_best_model.pth.tar'
fi

if [ "${2}" == "--data_is_actor" ]
then
    DATA_IS_ACTOR='--data_is_actor'
else
    DATA_IS_ACTOR=' '
fi

#DATAROOT='/research/sunxm/outer_prod_motion_mask/results/weizmann_convlstm_nobn_correct_excmotion_3/evaluation/500000'
#DATAROOT='/research/sunxm/mocogan/weizmann/results'
DATAROOT='/research/sunxm/ae_gan_nobn/results/weizmann_videoVAEGAN_2/evaluation/400000'
DATAROOT='/research/sunxm/outer_prod_motion_mask/results/weizmann_convlstm_unjoint_nobn/evaluation/500000'
DATAROOT='/research/sunxm/outer_prod_motion_mask/results/weizmann_convlstm_nobn_nomask/evaluation/500000'


python eval.py \
    --exp_name ${EXP_NAME} \
    --dataset Generated_video \
    --dataroot  ${DATAROOT} \
    --ckpt_path ${CKPT_PATH} \
    --test_dataset=Weizmann \
    --test_dataroot /scratch/dataset/Weizmann_crop_npy_new2/ \
    --textroot ../videolist/Weizmann/ \
    ${MODEL_IS_ACTOR} \
    ${DATA_IS_ACTOR}

