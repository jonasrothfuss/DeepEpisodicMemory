## General
##### Running tensorboard without docker:
python3 /usr/local/lib/python3.4/dist-packages/tensorflow/tensorboard/tensorboard.py --logdir=/data/rothfuss/training/

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

##### Accessing Tensorboard on other pc via ssh
ssh -L 6007:127.0.0.1:6006 rothfuss@i61pc032

##### Explicit command to origin:
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:0.12.0-rc1-devel-gpu-py3

##### newer version:
nvidia-docker run -it tensorflow/tensorflow:1.0.0-devel-gpu-py3

##### save a docker image and commit it to docker hub:
1. run image, make changes to the image
2. in another terminal, run "nvidia-docker ps -a"
3. get the container_id
4. run "nvidia-docker commit <container_id> ferreirafabio/deepepisodicmemory 

