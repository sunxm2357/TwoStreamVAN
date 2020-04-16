#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --exp_name=synaction_w_pretrain_nomask \
    --dataset=SynAction \
    --textroot ../videolist/SynAction/ \
    --dataroot /scratch/dataset/synaction_npy2/ \
    --n_channels 3 \
    --video_length=10 \
    --every_nth=2 \
    --no_mask \
    --which_iter 760000 \
    --no_bn \
    --joint \
