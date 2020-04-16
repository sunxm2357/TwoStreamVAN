#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --exp_name=synaction_w_pretrain_2 \
    --dataset=SynAction \
    --textroot ../videolist/SynAction/ \
    --dataroot /scratch/dataset/synaction_npy2/ \
    --n_channels 3 \
    --video_length=10 \
    --every_nth=2 \
    --which_iter 640000 \
    --no_bn \
    --joint \
