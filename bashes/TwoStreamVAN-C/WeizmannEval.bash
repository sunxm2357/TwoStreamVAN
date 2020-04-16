#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --dataset=Weizmann \
    --textroot ../videolist/Weizmann/ \
    --dataroot /scratch2/research/TwoStreamVAN/dataset/weizmann \
    --every_nth=2 \
    --crop \
    --model TwoStreamVAN \
    --exp_name=twostreamvan-c_weizmann \
    --log_dir /scratch2/research/TwoStreamVAN/exps/logs \
    --checkpoint_dir /scratch2/research/TwoStreamVAN/exps/checkpoints \
    --output_dir /scratch2/research/TwoStreamVAN/exps/results \
    --motion_dim 64 \
    --cont_dim 512 \
    --gf_dim 16 \
    --which_iter 500000 \
    --val_num 100