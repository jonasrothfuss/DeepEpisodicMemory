#!/bin/sh
pip install --user --upgrade virtualenv
mkdir ~/virtual_envs
virtualenv -p python --system-site-packages ~/virtual_envs/tf-gpu
source ~/virtual_envs/tf-gpu/bin/activate
pip install -r ~/DeepEpisodicMemory/requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl
pip install --user --upgrade $TF_BINARY_URL
