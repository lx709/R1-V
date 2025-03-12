#!/bin/bash
# Set CUDA version explicitly
export CUDA_HOME=/sw/rl9g/cuda/12.1/rl9_binary/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create clean conda env
conda create -n r1-v2 python=3.11 -y
conda activate r1-v2

# Install torch (must match CUDA version)
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install CUDA-aware packages
pip install bitsandbytes==0.43.0
pip install flash-attn --no-build-isolation

# Install your editable repo
cd src/r1-v
pip install -e ".[dev]"

# Install additional packages
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils
pip install vllm==0.7.2
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

DS_BUILD_CPU_ADAM=1 DS_BUILD_OPS=1 pip install deepspeed==0.15.4
pip install -U git+https://github.com/huggingface/transformers.git
