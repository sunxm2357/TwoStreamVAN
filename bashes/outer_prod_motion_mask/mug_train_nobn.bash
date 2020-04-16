#!/usr/bin/env bash

if [ "${1}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

cd outer_prod_motion_mask

python3 train.py \
    --exp_name=mug \
    --dataset=MUG \
    --textroot ../videolist/MUG/ \
    --dataroot /scratch/dataset/MUG_facial_expression/ \
    --n_channels 3 \
    --print_freq=50 \
    --save_freq=10000 \
    --video_length=10 \
    --every_nth=1 \
    --crop \
    --joint \
    --total_iters 500000 \
    --c_kl=5 \
    --img_m_kl=5 \
    --vid_m_kl_start=5 \
    --vid_m_kl_end=25 \
    --c_img_dis=10 \
    --pred_scale_feat=100 \
    --video_scale_feat=100 \
    --no_bn \
    --batch_size=16 \
    ${RESUME} \
