#!/usr/bin/env bash

if [ "${1}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

cd outer_prod_motion_mask

python3 train.py \
    --dataset=MUG \
    --textroot ../videolist/MUG/ \
    --dataroot /scratch2/research/TwoStreamVAN/dataset/MUG_facial_expression \
    --every_nth=1 \
    --crop \
    --model SGVAN \
    --exp_name=sgvan_mug \
    --log_dir /scratch2/research/TwoStreamVAN/exps/logs \
    --checkpoint_dir /scratch2/research/TwoStreamVAN/exps/checkpoints \
    --output_dir /scratch2/research/TwoStreamVAN/exps/results \
    --motion_dim 64 \
    --cont_dim 512 \
    --gf_dim 16 \
    --batch_size=16 \
    --print_freq=50 \
    --save_freq=10000 \
    --total_iters 500000 \
    --c_kl_start=5 \
    --c_kl_end=5 \
    --img_m_kl=5 \
    --xp_vs_xtilde=1 \
    --vid_m_kl_start=5 \
    --vid_m_kl_end=25 \
    --c_img_dis=10 \
    --cont_ratio_start 0.3 \
    --cont_ratio_end 0.6 \
    --cont_ratio_iter_start 180000 \
    --cont_ratio_iter_end 200000 \
    --motion_ratio_start 0.1 \
    --motion_ratio_end 0.9 \
    --motion_ratio_iter_start 0 \
    --motion_ratio_iter_end 200000 \
    ${RESUME} \

