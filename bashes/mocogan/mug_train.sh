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
    --every_nth 1 \
    --dim_z_content 50 \
    --dim_z_motion 10 \
    --dim_z_category 6 \
    --video_length 16 \
     /scratch/dataset/MUG_facial_expression/ /research/sunxm/mocogan/mug_16/