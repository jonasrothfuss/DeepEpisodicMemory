import tensorflow as tf
from settings import FLAGS
from utils.helpers import get_iter_from_pretrained_model, remove_items_from_dict

class Initializer:
  def __init__(self):
    self.status = False
    self.sess = None
    self.threads = None
    self.coord = None
    self.saver = None
    self.saver_restore = None
    self.itr_start = 0

  def start_session(self):
    """Starts a session and initializes all variables. Provides access to session and coordinator"""
    # Start Session and initialize variables
    self.status = True

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    self.sess = tf.Session()
    self.sess.run(init_op)

    # Start input enqueue threads
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

  def stop_session(self):
    """Stops a current session."""
    if self.sess and self.coord:
      self.coord.join(self.threads)
      self.sess.close()
      self.status = False

  def start_saver(self):
    """Constructs a saver and if pretrained model given, loads the model."""
    print('Constructing saver')
    self.saver = tf.train.Saver(max_to_keep=0)

    # restore dumped model if provided
    if FLAGS.pretrained_model:
      print('Restore model from: ' + str(FLAGS.pretrained_model))
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.pretrained_model)
      self.itr_start = get_iter_from_pretrained_model(latest_checkpoint) + 1
      print('Start with iteration: ' + str(self.itr_start))

      if FLAGS.exclude_from_restoring is not None:
        vars_to_exclude = str(FLAGS.exclude_from_restoring).replace(' ','').split(',')
        global_vars = dict([(v.name, v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="train_model")])
        global_vars = remove_items_from_dict(global_vars, vars_to_exclude)
        self.saver_restore = tf.train.Saver(var_list=list(global_vars.values()), max_to_keep=0)
        self.saver_restore.restore(self.sess, latest_checkpoint)
      else:
        self.saver.restore(self.sess, latest_checkpoint)

    return self.saver