## Tensorflow
### Installation

##### Installing Tensorflow 0.12.1 (Python 2.7, Ubuntu 64, with GPU)
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl \
pip install --upgrade $TF_BINARY_URL

##### Installing Tensorflow 0.12.1 (Python 3.4, Ubuntu 64, with GPU)
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl \
pip3 install --upgrade $TF_BINARY_URL

### Tensorboad
##### Running tensorboard without docker:
python3 /usr/local/lib/python3.4/dist-packages/tensorflow/tensorboard/tensorboard.py --logdir=/data/rothfuss/training/

##### Accessing Tensorboard on other pc via ssh
ssh -L 6007:127.0.0.1:6006 rothfuss@xxxxx032

## virtualenv
##### Run python 3.4 virtual environment:
source ~/p3.4/bin/activate
source /localhome/rothfuss/p3.4_local

## Using Docker
Container content: TensorFlow r0.12.0 rc1 CUDA8.0 cuDNN 5 Python 3.4.3

##### Run command:
nvidia-docker run --rm -ti ferreirafabio/deepepisodicmemory

##### Run command with ~/Downloads directory of host OS attached
nvidia-docker run -v ~Downloads:/Downloads --rm -ti ferreirafabio/deepepisodicmemory

##### Run command with local dir attached
nvidia-docker run -v /localhome/rothfuss:/local --rm -ti ferreirafabio/deepepisodicmemory

##### Then run training:
python3 train_model.py --path /local/data/ArtificialFlyingBlobs/tfrecords --output_dir /local/training --num_epochs 80000

##### Load trained model:
python3 train_model.py --path /local/data/ArtificialFlyingBlobs/tfrecords --output_dir /local/training --num_epochs 80000 --pretrained_model /local/training/

##### Running tensorboard:
tensorboard --logdir=/local/training/log/ --port 6006
python3 /usr/local/lib/python3.4/dist-packages/tensorflow/tensorboard/tensorboard.py  --logdir=/data/rothfuss/training

##### Explicit command to origin:
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:0.12.0-rc1-devel-gpu-py3

##### newer version:
nvidia-docker run -it tensorflow/tensorflow:1.0.0-devel-gpu-py3

##### save a docker image and commit it to docker hub:
1. run image, make changes to the image
2. in another terminal, run "nvidia-docker ps -a"
3. get the container_id
4. run "nvidia-docker commit <container_id> ferreirafabio/deepepisodicmemory 

