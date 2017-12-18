#  Deep Episodic Memory: Encoding, Recalling, and Predicting Episodic Experiences for Robot Action Execution
## Abstract
We present a novel deep neural network architecture for representing robot experiences in an episodic-like memory which facilitates encoding, recalling, and predicting action experiences. Our proposed unsupervised deep episodic memory model 1) encodes observed actions in a latent vector space and, based on this latent encoding, 2) infers action categories, 3) reconstructs original frames, and 4) predicts future frames. We evaluate the proposed model on two different large-scale action datasets. Results show that conceptually similar actions are mapped into the same region of the latent vector space. Based on this contribution, we introduce an action matching and retrieval mechanism and evaluate its performance and generalization capability on a real humanoid robot in an action execution scenario.

## Brief code introduction
If you're interested in running the code, it is recommended to start with _main.py_ since this is our entry pointe for the three modes 1)training, 2)validation and 3)feeding. The first two modes are used with tfrecords data while the third mode allows to query/train the net by using raw images from the OS file system. Please ensure you're using at least _tensorflow 0.12.1_. We also suggest to have at least 12GB GPU RAM for training due to our deep architecture models. The following listing should give you an overview about the files/directories that likely require a closer look for your intention:
+ **main.py**
Use it to run training, validation and feeding cycles. Set hyperparameters and constants at the top of settings.py first
+ **core/development_op.py**
Train and validation code to train and test the memory
+ **core/production_op.py**
Feeding code, used for querying the memory (meant for e.g. demonstrations), allows adapatations for accessing the memory over a network (e.g. with an ICE service)
+ **models**
  directory containing files for loss functions (mse, gradient difference loss, decoder/encoder loss, PSNR) and basic lstm cell
+ **models/model_zoo**
  directory that contains our composite model in different configurations (mostly affecting depth and filter sizes)
+ **data_prep/**
  directory with a collection of code for preprocessing data, e.g. converting video files to numpy or generating tf records from raw .avi
+ **data_postp/**
  directory with all the code that we used to compute latent space similarities, classification accuracies and also for executing the retrieving and matching mechanism
+ **utils/**
  directory containing mostly i/o scripts for reoccuring tasks (e.g. frames to .gif)
+ **data_prep/convertToRecords.py**
  file used for generating .tfrecords from raw video files (e.g. .avi). Also includes code for selecting frames equally distributed over the entire playtime. Hyperparameters at the top allow adjustments



## Website
http://h2t-projects.webarchiv.kit.edu/projects/episodicmemory
