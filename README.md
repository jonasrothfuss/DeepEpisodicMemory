#  Deep Episodic Memory: Encoding, Recalling, and Predicting Episodic Experiences for Robot Action Execution
## Abstract
We present a novel deep neural network architecture for representing robot experiences in an episodic-like memory which facilitates encoding, recalling, and predicting action experiences. Our proposed unsupervised deep episodic memory model 1) encodes observed actions in a latent vector space and, based on this latent encoding, 2) infers action categories, 3) reconstructs original frames, and 4) predicts future frames. We evaluate the proposed model on two different large-scale action datasets. Results show that conceptually similar actions are mapped into the same region of the latent vector space. Based on this contribution, we introduce an action matching and retrieval mechanism and evaluate its performance and generalization capability on a real humanoid robot in an action execution scenario.

## Brief code introduction
If interested in running the code, it is recommended to start with _train_model.py_ since this is our entry point/main file for training and validation and having at least _tensorflow 0.12.1_ installed. We also suggest to have at least 12GB GPU RAM for training due to our deep architecture. The following listing should give you an overview about the files/directories that likely require a closer look for your intention:
+ **train_model.py**
  main file. Use it to run training and validation cycles. Set hyperparameters and constants at the top of the file first
+ **convertToRecords.py**
  file used for generating .tfrecords from raw video files (e.g. .avi). Also includes code for selecting frames equally distributed over the entire playtime. Hyperparameters at the top allow adjustments
+ **models**
  directory containing files for loss functions (mse, gradient difference loss, decoder/encoder loss, PSNR) and basic lstm cell
+ **models/model_zoo**
  directory that contains our composite model in different configurations (mostly affecting depth and filter sizes)
+ **data_prep**
  directory with a collection of code for preprocessing data, e.g. converting video files to numpy or generating tf records from raw .avi
+ **data_postp**
  directory with all the code that we used to compute latent space similarities, classification accuracies and also for executing the retrieving and matching mechanism
+ **utils**
  directory containing mostly i/o scripts for reoccuring tasks (e.g. frames to .gif)


## Website
http://h2t-projects.webarchiv.kit.edu/projects/episodicmemory
