#!/usr/bin/env bash

cd mocogan

python generate_videos.py \
    -n 900 \
    -o gif \
    -f 10 \
    /research/sunxm/mocogan/synaction/generator_100000.pytorch \
    /research/sunxm/mocogan/synaction/results/ \
