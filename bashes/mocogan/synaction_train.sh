#!/usr/bin/env bash

cd mocogan

python train.py  \
    --image_batch 32 \
    --video_batch 32 \
    --use_infogan \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator CategoricalVideoDiscriminator \
    --print_every 100 \
    --every_nth 2 \
    --dim_z_content 50 \
    --dim_z_motion 10 \
    --dim_z_category 20 \
    --video_length 10 \
     /scratch/dataset/synaction_npy2/ /research/sunxm/mocogan/synaction/