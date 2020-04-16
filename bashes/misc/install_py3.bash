#!/bin/bash

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
export CUDA_HOME=/usr/local/cuda-9.0/


GPU_ARCH=compute_52
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

nvcc -c -o separable_convolution/cfile/SeparableConvolution_kernel.o separable_convolution/cfile/SeparableConvolution_kernel.cu --gpu-architecture=$GPU_ARCH --gpu-code=$GPU_ARCH --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python3 separable_convolution/install.py