import tensorflow as tf


def feed(feed_batch, initializer, feed_model):
  '''
  feeds the videos inherent feed_batch trough the network provided in feed_model
  :param feed_batch: 5D Tensor (batch_size, num_frames, width, height, num_channels)
  :param initializer:
  :param feed_model:
  :return:
  '''
  assert feed_model is not None and initializer is not None
  assert feed_batch.ndim == 5

  tf.logging.info(' --- Starting feeding --- ')

  feed_dict = {feed_model.learning_rate: 0.0, feed_model.feed_batch: feed_batch}

  hidden_repr = initializer.sess.run([feed_model.hidden_repr], feed_dict)


  return hidden_repr