#!/usr/bin/env bash

cd outer_prod_motion_mask

python eval.py \
    --dataset=SynAction \
    --textroot ../videolist/SynAction/ \
    --dataroot /scratch2/research/TwoStreamVAN/dataset/synaction_npy2 \
    --every_nth=2 \
    --crop \
    --model SGVAN \
    --exp_name=sgvan_synaction \
    --log_dir /scratch2/research/TwoStreamVAN/exps/logs \
    --checkpoint_dir /scratch2/research/TwoStreamVAN/exps/checkpoints \
    --output_dir /scratch2/research/TwoStreamVAN/exps/results \
    --motion_dim 64 \
    --cont_dim 1024 \
    --gf_dim 32 \
    --which_iter 770000 \
    --val_num 100