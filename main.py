import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.platform import app

from core.Initializer import Initializer
from core.Model import create_model
from core.development_op import train, validate
from core.production_op import feed, create_batch_and_feed
from settings import FLAGS
from core.Memory import Memory
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
  initializer.start_saver()

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
    if FLAGS.dump_dir:
      output_dir = FLAGS.dump_dir
    else:
      output_dir = create_subfolder(output_dir, 'valid_run')
    print('Storing validation data in:', output_dir)

    tf.logging.info(' --- ' + FLAGS.mode.capitalize() + ' --- ')
    validate(output_dir, initializer, val_model)

  # ---- feeding  ----- #
  if FLAGS.mode is "feeding_mode":
    tf.logging.info(' --- ' + FLAGS.mode.capitalize() + ' --- ')

    """ scenario 1: feed input from a directory to create hidden representations of query  """
    hidden_repr = create_batch_and_feed(initializer, feeding_model)
    print("output of model has shape: " + str(np.shape(hidden_repr)))

    """ scenario 2: load the memory and query it with the hidden reps to get nearest neighbours  """
    # alternatively, run a validation with the 'memory_prep' VALID MODE (settings) and set memory path in settings
    assert FLAGS.memory_path
    memory_df = pd.read_pickle(FLAGS.memory_path)
    memory = Memory(memory_df, '/common/homes/students/rothfuss/Documents/example/base_dir')

    # choose e.g. first hidden representation
    query = hidden_repr[0]
    _, cos_distances, _, paths = memory.matching(query)
    print(cos_distances, paths)



if __name__ == '__main__':
  app.run()