from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

OUT_DIR = '/common/homes/students/rothfuss/Documents/selected_trainings/5_actNet_20bn_gdl'
DATA_PATH = '/PDFData/rothfuss/data/ArmarExperiences/videos/tf_records'

# other constants
LOSS_FUNCTIONS = ['mse', 'gdl', 'mse_gdl']

# for pretraining-mode only, e.g. ".../selected_trainings/5_actNet_20bn_gdl"
PRETRAINED_MODEL = ""

# available modes of operation
MODES = ["train_mode", "valid_mode", "demo_mode"]
VALID_MODES = ['count_trainable_weights', 'vector', 'gif', 'similarity', 'data_frame', 'psnr']

EXCLUDE_FROM_RESTORING = None
FINE_TUNING_WEIGHTS_LIST = None
# FINE_TUNING_WEIGHTS_LIST = [ 'train_model/encoder/conv4', 'train_model/encoder/convlstm4', 'train_model/encoder/conv5', 'train_model/encoder/convlstm5',
#                       'train_model/encoder/fc_conv', 'train_model/encoder/convlstm6', 'train_model/decoder_pred/upconv4',
#                       'train_model/decoder_pred/conv4', 'train_model/decoder_pred/convlstm5', 'train_model/decoder_pred/upconv5',
#                       'train_model/decoder_reconst/conv4', 'train_model/decoder_reconst/convlstm5', 'train_model/decoder_reconst/upconv5',
#                       'train_model/decoder_reconst/upconv4']


# model hyperparameters
flags.DEFINE_integer('num_iterations', 100000, 'specify number of training iterations, defaults to 100000')
flags.DEFINE_string('loss_function', 'mse_gdl', 'specify loss function to minimize, defaults to gdl')
flags.DEFINE_string('batch_size', 50, 'specify the batch size, defaults to 50')
flags.DEFINE_integer('valid_batch_size', 150, 'specify the validation batch size, defaults to 50')
flags.DEFINE_bool('uniform_init', False,
                  'specifies if the weights should be drawn from gaussian(false) or uniform(true) distribution')
flags.DEFINE_integer('num_gpus', 1, 'specifies the number of available GPUs of the machine')

flags.DEFINE_integer('image_range_start', 0,
                     'parameter that controls the index of the starting image for the train/valid batch')
flags.DEFINE_integer('overall_images_count', 15,
                     'specifies the number of images that are available to create the train/valid batches')
flags.DEFINE_string('encoder_length', 5, 'specifies how many images the encoder receives, defaults to 5')
flags.DEFINE_string('decoder_future_length', 5,
                    'specifies how many images the future prediction decoder receives, defaults to 5')
flags.DEFINE_string('decoder_reconst_length', 5,
                    'specifies how many images the reconstruction decoder receives, defaults to 5')
flags.DEFINE_integer('num_channels', 4, 'number of channels in the input frames')
flags.DEFINE_bool('fc_layer', True,
                  'indicates whether fully connected layer shall be added between encoder and decoder')
flags.DEFINE_float('learning_rate_decay', 0.000008, 'learning rate decay factor')
flags.DEFINE_integer('learning_rate', 0.00001, 'initial learning rate for Adam optimizer')
flags.DEFINE_float('noise_std', 0.1,
                   'defines standard deviation of gaussian noise to be added to the hidden representation during training')
flags.DEFINE_float('keep_prob_dopout', 0.85,
                   'keep probability for dropout during training, for valid automatically 1')

# intervals
flags.DEFINE_integer('valid_interval', 100, 'number of training steps between each validation')
flags.DEFINE_integer('summary_interval', 100, 'number of training steps between summary is stored')
flags.DEFINE_integer('save_interval', 500, 'number of training steps between session/model dumps')

# set mode of operation
flags.DEFINE_string('mode', "demo_mode", 'Allowed modes: ' + str(
  MODES) + '. "demo_mode": model is fed from numpy data directly instead of tfrecords'
           '"valid_mode": '
           '"train_mode": ')

# valid mode specifications
flags.DEFINE_string('valid_mode', "data_frame", 'Allowed modes: ' + str(
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

# I/O specifications
flags.DEFINE_string('path', DATA_PATH, 'specify the path to where tfrecords are stored, defaults to "../data/"')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
