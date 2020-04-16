#!/usr/bin/env bash

if [ "${1}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

cd outer_prod_motion_mask

python3 train.py \
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
    --batch_size=16 \
    --print_freq=50 \
    --save_freq=10000 \
    --total_iters 500000 \
    --c_kl_start=7 \
    --c_kl_end=7 \
    --img_m_kl=7 \
    --xp_vs_xtilde=1 \
    --vid_m_kl_start=2 \
    --vid_m_kl_end=20 \
    --c_img_dis=10 \
    --pred_scale_feat=100 \
    --video_scale_feat=100 \
    --cont_ratio_start 0.4 \
    --cont_ratio_end 0.4 \
    --cont_ratio_iter_start 0 \
    --cont_ratio_iter_end 200000 \
    --motion_ratio_start 0.1 \
    --motion_ratio_end 0.9 \
    --motion_ratio_iter_start 0 \
    --motion_ratio_iter_end 150000 \
    ${RESUME} \
