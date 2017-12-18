#!/bin/sh
pip install --user --upgrade virtualenv
virtualenv -p python3 --system-site-packages ~/virtual_envs/tf
source ~/virtual_envs/tf/bin/activate
pip3 install -r ./requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
pip3 install --user --upgrade $TF_BINARY_URL
