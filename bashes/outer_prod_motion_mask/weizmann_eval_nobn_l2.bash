#!/usr/bin/env bash

if [ "${1}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

cd outer_prod_motion_mask

python eval.py \
    --exp_name=weizmann_convlstm_nobn_correct_excmotion_3 \
    --dataset=Weizmann \
    --textroot ../videolist/Weizmann/ \
    --dataroot /scratch/dataset/Weizmann_crop_npy_new2/ \
    --n_channels 3 \
    --video_length=10 \
    --every_nth=2 \
    --crop \
    --which_iter 500000 \
    --no_bn \
    --joint \
    ${RESUME} \
