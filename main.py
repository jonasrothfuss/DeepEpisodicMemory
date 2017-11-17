import tensorflow as tf
from tensorflow.python.platform import app

from core.Initializer import Initializer
from core.Model import create_model
from core.development_op import train, validate
from core.production_op import feed
from settings import FLAGS
from utils.io_handler import create_session_dir, create_subfolder, write_metainfo, generate_batch_from_dir


def main(argv):
  # train model is required in every case
  train_model = create_model(mode='train')
  if FLAGS.mode is "train_mode" or FLAGS.mode is "valid_mode":
    val_model = create_model(mode='valid', train_model_scope=train_model.scope)
  elif FLAGS.mode is "feeding_mode":
    feeding_model = create_model(mode='feeding', train_model_scope=train_model.scope)

  initializer = Initializer()
  initializer.start_session()

  # ---- training ----- #
  if FLAGS.mode is "train_mode":
    """either create new directory or reuse old one"""
    if not FLAGS.pretrained_model:
      output_dir = create_session_dir(FLAGS.output_dir)
    else:
      output_dir = FLAGS.pretrained_model
      print('Reusing provided session directory:', output_dir)

    tf.logging.info(' --- ' + FLAGS.mode.capitalize() + ' --- ')
    write_metainfo(output_dir, train_model.model_name, FLAGS)
    train(output_dir, initializer, train_model, val_model)

  # ---- validation  ----- #
  if FLAGS.mode is "valid_mode":
    assert FLAGS.pretrained_model
    output_dir = FLAGS.pretrained_model
    tf.logging.info(' --- ' + FLAGS.mode.capitalize() + ' --- ')
    print('Reusing provided session directory:', output_dir)
    subdir = create_subfolder(output_dir, 'valid_run')
    print('Storing validation data in:', subdir)

  # ---- feeding  ----- #
  if FLAGS.mode is "feeding_mode":
    tf.logging.info(' --- ' + FLAGS.mode.capitalize() + ' --- ')
    batch_input_dir = '/common/homes/students/rothfuss/Documents/training_tests/input_data/images/2'
    feed_batch = generate_batch_from_dir(batch_input_dir, suffix='*.jpg')
    print(feed_batch.shape)
    assert FLAGS.pretrained_model
    hidden_repr = feed(feed_batch, initializer, feeding_model)
    print(hidden_repr)


if __name__ == '__main__':
  app.run()