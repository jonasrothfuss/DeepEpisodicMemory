import tensorflow as tf
import numpy as np
from settings import FLAGS
from utils.io_handler import generate_batch_from_dir



def create_batch_and_feed(initializer, feeding_model):
  # TODO
  """

  :param initializer:
  :param feed_model:
  :return:
  """
  assert FLAGS.pretrained_model
  feed_batch = generate_batch_from_dir(FLAGS.feeding_input_dir, suffix='*.jpg')
  print("feed batch has shape: " + str(feed_batch.shape))
  hidden_repr = feed(feed_batch, initializer=initializer, feeding_model=feeding_model)

  return np.array(np.squeeze(hidden_repr))



def feed(feed_batch, initializer, feeding_model):
  '''
  feeds the videos inherent feed_batch trough the network provided in feed_model
  :param feed_batch: 5D Tensor (batch_size, num_frames, width, height, num_channels)
  :param initializer:
  :param feed_model:
  :return:
  '''
  assert feeding_model is not None and initializer is not None
  assert feed_batch.ndim == 5


  tf.logging.info(' --- Starting feeding --- ')

  feed_dict = {feeding_model.learning_rate: 0.0, feeding_model.feed_batch: feed_batch}

  hidden_repr = initializer.sess.run([feeding_model.hidden_repr], feed_dict)


  return hidden_repr