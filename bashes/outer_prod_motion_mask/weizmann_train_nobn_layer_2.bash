#!/usr/bin/env bash

if [ "${1}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

cd outer_prod_motion_mask

python3 train.py \
    --exp_name=weizmann_convlstm_nobn_kernel_layer_2_try2 \
    --dataset=Weizmann \
    --textroot ../videolist/Weizmann/ \
    --dataroot /scratch/dataset/Weizmann_crop_npy_new2/ \
    --n_channels 3 \
    --print_freq=50 \
    --save_freq=10000 \
    --video_length=10 \
    --every_nth=2 \
    --crop \
    --joint \
    --total_iters 500000 \
    --c_kl=7 \
    --img_m_kl=7 \
    --vid_m_kl_start=2 \
    --vid_m_kl_end=20 \
    --c_img_dis=10 \
    --pred_scale_feat=100 \
    --video_scale_feat=100 \
    --no_bn \
    --batch_size=16 \
    --kernel_layer=2 \
    ${RESUME} \
