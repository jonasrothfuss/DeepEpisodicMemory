from tensorflow.python.platform import flags
import os
import utils.helpers as helpers
import warnings

FLAGS = flags.FLAGS

""" Set Model From Model Zoo"""
from models.model_zoo import model_conv5_fc_lstm2_1000_deep_64 as model
""""""

# --- SPECIFY MANDATORY VARIABLES--- #
OUT_DIR = '/common/homes/students/rothfuss/Documents/training_tests'
DUMP_DIR = '/common/homes/students/rothfuss/Documents/Episodic_Memory/Armar_Experiences'
TF_RECORDS_DIR = '/data/rothfuss/data/ArmarExperiences/tf_records/tf_records_memory'
MODE = 'valid_mode'
VALID_MODE = 'memory_prep' #'data_frame gif'

NUM_IMAGES = 15
NUM_DEPTH = 4
WIDTH = 128
HEIGHT = 128
NUM_THREADS_QUEUERUNNER = 32 # specifies the number of pre-processing threads

# PRETRAINING / FINETUNING
PRETRAINED_MODEL = "/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow"
EXCLUDE_FROM_RESTORING = None
FINE_TUNING_WEIGHTS_LIST = None
# FINE_TUNING_WEIGHTS_LIST = [ 'train_model/encoder/conv4', 'train_model/encoder/convlstm4', 'train_model/encoder/conv5', 'train_model/encoder/convlstm5',
#                       'train_model/encoder/fc_conv', 'train_model/encoder/convlstm6', 'train_model/decoder_pred/upconv4',
#                       'train_model/decoder_pred/conv4', 'train_model/decoder_pred/convlstm5', 'train_model/decoder_pred/upconv5',
#                       'train_model/decoder_reconst/conv4', 'train_model/decoder_reconst/convlstm5', 'train_model/decoder_reconst/upconv5',
#                       'train_model/decoder_reconst/upconv4']



# --- INFORMAL LOCAL VARIABLES --- #
LOSS_FUNCTIONS = ['mse', 'gdl', 'mse_gdl']
MODES = ["train_mode", "valid_mode", "feeding_mode"]
VALID_MODES = ['count_trainable_weights', 'vector', 'gif', 'similarity', 'data_frame', 'psnr', 'memory_prep']



# --- MODEL INPUT PARAMETERS --- #
flags.DEFINE_integer('num_images', NUM_IMAGES, 'specify the number of images in the tfrecords')
flags.DEFINE_integer('num_depth', NUM_DEPTH, 'specifies the number of depth channels in the images')
flags.DEFINE_integer('width', WIDTH, 'specifies the width of an image')
flags.DEFINE_integer('height', HEIGHT, 'specifies the height of an image')


# --- I/O SPECIFICATIONS --- #
flags.DEFINE_string('tf_records_dir', TF_RECORDS_DIR, 'specify the path to where tfrecords are stored, defaults to "../data/"')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('dump_dir', DUMP_DIR, 'directory for validation dumps such as gif and data_frames')

# --- TFRECORDS --- #
flags.DEFINE_string('train_files', 'train*.tfrecords', 'Regex for filtering train tfrecords files.')
flags.DEFINE_string('valid_files', 'valid*.tfrecords', 'Regex for filtering valid tfrecords files.')
flags.DEFINE_string('test_files', 'test*.tfrecords', 'Regex for filtering test tfrecords files.')
flags.DEFINE_integer('num_threads', NUM_THREADS_QUEUERUNNER, 'specifies the number of threads for the queue runner')


# --- MODEL HYPERPARAMETERS --- #
flags.DEFINE_integer('num_iterations', 100000, 'specify number of training iterations, defaults to 100000')
flags.DEFINE_string('loss_function', 'mse_gdl', 'specify loss function to minimize, defaults to gdl')
flags.DEFINE_integer('batch_size', 30, 'specify the batch size, defaults to 50')
flags.DEFINE_integer('valid_batch_size', 80, 'specify the validation batch size, defaults to 50')
flags.DEFINE_bool('uniform_init', False,
                  'specifies if the weights should be drawn from gaussian(false) or uniform(true) distribution')
flags.DEFINE_integer('num_gpus', len(helpers.get_available_gpus()), 'specifies the number of available GPUs of the machine')

flags.DEFINE_integer('image_range_start', 0,
                     'parameter that controls the index of the starting image for the train/valid batch')
flags.DEFINE_integer('overall_images_count', 15,
                     'specifies the number of images that are available to create the train/valid batches')
flags.DEFINE_integer('encoder_length', 5, 'specifies how many images the encoder receives, defaults to 5')
flags.DEFINE_integer('decoder_future_length', 5,
                    'specifies how many images the future prediction decoder receives, defaults to 5')
flags.DEFINE_integer('decoder_reconst_length', 5,
                    'specifies how many images the reconstruction decoder receives, defaults to 5')
flags.DEFINE_integer('num_channels', 4, 'number of channels in the input frames')
flags.DEFINE_bool('fc_layer', True,
                  'indicates whether fully connected layer shall be added between encoder and decoder')
flags.DEFINE_float('learning_rate_decay', 0.000008, 'learning rate decay factor')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate for Adam optimizer')
flags.DEFINE_float('noise_std', 0.1,
                   'defines standard deviation of gaussian noise to be added to the hidden representation during training')
flags.DEFINE_float('keep_prob_dopout', 0.85,
                   'keep probability for dropout during training, for valid automatically 1')


# --- INTERVALS --- #
flags.DEFINE_integer('valid_interval', 100, 'number of training steps between each validation')
flags.DEFINE_integer('summary_interval', 100, 'number of training steps between summary is stored')
flags.DEFINE_integer('save_interval', 500, 'number of training steps between session/model dumps')


# --- MODE OF OPERATION --- #
flags.DEFINE_string('mode', MODE, 'Allowed modes: ' + str(
  MODES) + '. "feeding_mode": model is fed from numpy data directly instead of tfrecords'
           '"valid_mode": '
           '"train_mode": ')

flags.DEFINE_string('valid_mode', VALID_MODE, 'Allowed modes: ' + str(
  VALID_MODES) + '. "vector": encoder latent vector for each validation is exported to '
                 '"gif": gifs are generated from the videos'
                 '"similarity": compute (cos) similarity matrix'
                 '"data_frame": the model output is retrieved as a df'
                 '"count_trainable_weights": number of tr. weights is emitted to the'
                 'console')

flags.DEFINE_string('pretrained_model', PRETRAINED_MODEL, 'filepath of a pretrained model to initialize from.')
flags.DEFINE_string('exclude_from_restoring', EXCLUDE_FROM_RESTORING,
                    'variable names to exclude from saving and restoring')
flags.DEFINE_string('fine_tuning_weights_list', FINE_TUNING_WEIGHTS_LIST,
                    'variable names (layer scopes) that should be trained during fine-tuning')


assert os.path.isdir(FLAGS.tf_records_dir), "tf_records_dir must be a directory"
assert os.path.isdir(FLAGS.output_dir), "output_dir must be a directory"
assert not FLAGS.pretrained_model or os.path.isdir(FLAGS.pretrained_model), "pretrained_model must be a directory"
assert not FLAGS.dump_dir or os.path.isdir(FLAGS.dump_dir), "dump_dir must be a directory"

assert any([mode in FLAGS.valid_mode for mode in VALID_MODES]), "valid_mode must contain at least one of the following" + str(VALID_MODES)
assert FLAGS.mode in MODES, "mode must be one of " + str(MODES)

assert WIDTH == HEIGHT, "width must be equal to height"
assert FLAGS.num_images > 0, 'num_images must be positive integer'
assert FLAGS.num_depth > 0, 'num_depth must be positive integer'
assert FLAGS.num_threads > 0, 'num_threads must be a positive integer'
assert FLAGS.num_gpus >= 0

assert FLAGS.loss_function in LOSS_FUNCTIONS, "loss functions must be one of " + str(LOSS_FUNCTIONS)
assert FLAGS.num_depth >= FLAGS.num_channels, "provided number of depth channels in input data must be >= number of channels in model"

assert FLAGS.learning_rate_decay > 0.0 and FLAGS.learning_rate_decay < 1.0, 'learning rate decay should be in [0,1]'
assert FLAGS.learning_rate > 0.0 and FLAGS.learning_rate < 1.0, 'learning rate should be in [0,1]'
assert FLAGS.noise_std > 0.0 and FLAGS.noise_std < 1.0, 'noise_std should be in [0,1]'
assert FLAGS.keep_prob_dopout > 0.0 and FLAGS.keep_prob_dopout <= 1.0, 'keep_prob_dopout must be in [0,1]'


if FLAGS.exclude_from_restoring:
  warnings.warn("exclude_from_restoring is not empty -> layers may be omitted from restoring")

if FLAGS.fine_tuning_weights_list:
  warnings.warn("fine_tuning_weights_list is not empty -> layers may be omitted from training")
