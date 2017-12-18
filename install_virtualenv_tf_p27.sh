#!/bin/sh
pip install --user --upgrade virtualenv
virtualenv -p python --system-site-packages ~/virtual_envs/tf
source ~/virtual_envs/tf/bin/activate
pip install -r ./requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl
pip install --user --upgrade $TF_BINARY_URL
