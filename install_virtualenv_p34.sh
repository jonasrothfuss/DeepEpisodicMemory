#!/bin/sh
virtualenv ~/virtual_envs/tf-gpu
source ~/virtual_envs/tf-gpu/bin/activate
pip3 install -r ~/DeepEpisodicMemory/requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
