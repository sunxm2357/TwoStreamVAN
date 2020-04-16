#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --exp_name=weizmann_convlstm_nobn_kernel_layer_2_try2  \
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
    --kernel_layer=2 \
