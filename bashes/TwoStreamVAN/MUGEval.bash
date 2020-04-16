#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --dataset=MUG \
    --textroot ../videolist/MUG/ \
    --dataroot /scratch2/research/TwoStreamVAN/dataset/MUG_facial_expression \
    --every_nth=1 \
    --crop \
    --model TwoStreamVAN \
    --exp_name=twostreamvan_mug \
    --log_dir /scratch2/research/TwoStreamVAN/exps/logs \
    --checkpoint_dir /scratch2/research/TwoStreamVAN/exps/checkpoints \
    --output_dir /scratch2/research/TwoStreamVAN/exps/results \
    --motion_dim 64 \
    --cont_dim 512 \
    --gf_dim 16 \
    --joint \
    --which_iter 500000 \
    --val_num 100
