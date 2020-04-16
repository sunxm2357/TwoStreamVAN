#!/bin/bash

GPU_ARCH=$1
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

nvcc -c -o separable_convolution/cfile/SeparableConvolution_kernel.o separable_convolution/cfile/SeparableConvolution_kernel.cu --gpu-architecture=$GPU_ARCH --gpu-code=$GPU_ARCH --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python separable_convolution/install.py