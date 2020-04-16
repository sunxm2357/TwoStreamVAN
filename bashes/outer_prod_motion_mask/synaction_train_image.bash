#!/usr/bin/env bash

if [ "${1}" == "--resume" ]
then
    RESUME='--resume'
else
    RESUME=' '
fi

cd outer_prod_motion_mask

#export MKL_NUM_THREADS="2"
#export MKL_DYNAMIC="FALSE"
#export OMP_NUM_THREADS="2"
#export OMP_DYNAMIC="FALSE"

python3 train.py \
    --exp_name=debug_synaction_image_32_aegan_smallsample  \
    --dataset=SynAction \
    --textroot ../videolist/SynAction/ \
    --dataroot /scratch/dataset/synaction_npy2/ \
    --n_channels 3 \
    --print_freq=50 \
    --save_freq=10000 \
    --video_length=10 \
    --every_nth=2 \
    --crop \
    --joint \
    --total_iters 500000 \
    --c_kl_start=5 \
    --c_kl_end=5 \
    --img_m_kl=7 \
    --vid_m_kl_start=2 \
    --vid_m_kl_end=20 \
    --c_img_dis=1 \
    --pred_scale_feat=100 \
    --video_scale_feat=100 \
    --no_bn \
    --batch_size=16 \
    ${RESUME} \
