#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --exp_name=mug_unjoint \
    --dataset=MUG \
    --textroot ../videolist/MUG/ \
    --dataroot /scratch/dataset/MUG_facial_expression/ \
    --n_channels 3 \
    --video_length=10 \
    --every_nth=2 \
    --crop \
    --no_bn \
