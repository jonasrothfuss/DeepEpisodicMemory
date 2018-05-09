import datetime as dt
import os
import time

import numpy as np

import data_postp.metrics as metrics
import data_postp.similarity_computations as similarity_computations
import utils.helpers as helpers
from core.Model import *
from data_prep.TFRW2Images import createGif
from utils.io_handler import create_subfolder, store_output_frames_as_gif, store_latent_vectors_as_df, \
  store_encoder_latent_vector, file_paths_from_directory, write_file_with_append, bgr_to_rgb



def train(output_dir, initializer, train_model, val_model):
  assert train_model is not None and val_model is not None and initializer is not None

  saver = initializer.start_saver()

  summary_writer = tf.summary.FileWriter(output_dir, graph=initializer.sess.graph, flush_secs=10)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info(' --- Start Training --- ')
  tf.logging.info(' Iteration, Train_Loss ')

  elapsed_time = 0

  ''' main training loop '''
  try:
    for itr in range(initializer.itr_start, initializer.itr_start + FLAGS.num_iterations):
      try:
        if initializer.coord.should_stop():
          break

        #Training Step on batch
        learning_rate = helpers.learning_rate_decay(FLAGS.learning_rate, itr, decay_factor=FLAGS.learning_rate_decay)
        feed_dict = {train_model.learning_rate: learning_rate, train_model.elapsed_time: float(elapsed_time)}

        t = time.time()
        train_loss, _, train_summary_str = initializer.sess.run([train_model.loss, train_model.train_op, train_model.sum_op], feed_dict)
        elapsed_time = time.time() - t

        #validation
        if itr % FLAGS.valid_interval == 1:

          feed_dict = {val_model.learning_rate: 0.0}

          # summary and log
          val_loss, val_summary_str = initializer.sess.run([val_model.loss, val_model.sum_op], feed_dict)

          summary_writer.add_summary(val_summary_str, itr)

          #Print validation loss
          tf.logging.info(' Validation loss at step ' + str(itr) + ':    ' + str(val_loss))

        #dump summary
        if itr % FLAGS.summary_interval == 1:
          summary_writer.add_summary(train_summary_str, itr)

        #save model checkpoint
        if itr % FLAGS.save_interval == 1:
          save_path = saver.save(initializer.sess, os.path.join(output_dir, 'model'), global_step=itr) #TODO also implement save operation in Initializer class
          tf.logging.info('Saved Model to: ' + str(save_path))

        #Print iteration and loss
        tf.logging.info(' ' + str(itr) + ':    ' + str(train_loss) + ' | %.2f sec'%(elapsed_time))
      except Exception as e:
        tf.logging.info('Training iteration ' + str(itr) + 'failed: ' + str(e.message))
  except tf.errors.OutOfRangeError:
    tf.logging.info('Done training -- iterations limit reached')
  finally:
    # When done, ask the threads to stop.
    initializer.coord.request_stop()

  tf.logging.info(' Saving Model ... ')
  saver.save(initializer.sess, os.path.join(output_dir, 'model'), global_step=initializer.itr_start + FLAGS.num_iterations)

  # necessary for outer (train manager) loop to prevent variable conflicts with previously used graph
  tf.reset_default_graph()
  # Wait for threads to finish.
  initializer.stop_session()

def validate(output_dir, initializer, val_model):
  """ feeds validation batch through the model and stores produced frame sequence as gifs to output_dir
    :param
      output_dir: path to output directory where validation summary and gifs are stored
  """
  assert val_model is not None and initializer is not None

  initializer.start_saver()

  # Calculate number of validation samples
  valid_filenames = file_paths_from_directory(FLAGS.tf_records_dir, 'valid*')
  num_valid_samples = input.get_number_of_records(valid_filenames)
  print('Detected %i validation samples' % num_valid_samples)


  tf.logging.info(' --- Start validation --- ')

  try:
    feed_dict = {val_model.learning_rate: 0.0}

    val_loss, val_summary_str, output_frames, hidden_representations, labels, metadata, orig_frames = initializer.sess.run(
      [val_model.loss, val_model.sum_op, val_model.output_frames, val_model.hidden_repr, val_model.label, val_model.metadata, val_model.val_batch], feed_dict)

    if 'vector' in FLAGS.valid_mode:
      # store encoder latent vector for analysing

      hidden_repr_dir = create_subfolder(output_dir, 'hidden_repr')
      store_encoder_latent_vector(hidden_repr_dir, hidden_representations, labels, True)

    if 'gif' in FLAGS.valid_mode:
      # summary and log
      val_model.iter_num = 1
      createGif(np.asarray(orig_frames)[:, FLAGS.image_range_start:FLAGS.image_range_start + FLAGS.encoder_length + FLAGS.decoder_future_length,:,:,:3], labels, output_dir)
      tf.logging.info('Converting validation frame sequences to gif')
      store_output_frames_as_gif(np.asarray(output_frames)[:,:,:,:,:3], labels, output_dir)
      tf.logging.info('Dumped validation gifs in: ' + str(output_dir))

    if 'similarity' in FLAGS.valid_mode:
      print(str(similarity_computations.compute_hidden_representation_similarity(hidden_representations, labels, 'cos')))

    if 'memory_prep' in FLAGS.valid_mode:
      # evaluate multiple batches to cover all available validation samples
      video_file_path = []
      num_val_batches_required = (num_valid_samples // (FLAGS.valid_batch_size * FLAGS.num_gpus)) + int(
        (num_valid_samples % (FLAGS.valid_batch_size * FLAGS.num_gpus)) != 0)
      for i in range(num_val_batches_required):
        hidden_representations_new, labels_new, metadata_new, orig_frames = initializer.sess.run(
          [val_model.hidden_repr, val_model.label, val_model.metadata, val_model.val_batch], feed_dict)

        video_file_path.extend(["original_clip_%s.gif" % str(l.decode('utf-8')) for l in labels])
        hidden_representations = np.concatenate((hidden_representations, hidden_representations_new))
        labels = np.concatenate((labels, labels_new))
        metadata = np.concatenate((metadata, metadata_new))
        createGif(np.asarray(orig_frames)[:,:,:, :, :3], labels, output_dir)

      store_latent_vectors_as_df(output_dir, hidden_representations, labels, metadata, video_file_paths=np.asarray(video_file_path))

    if 'data_frame' in FLAGS.valid_mode:
      #evaluate multiple batches to cover all available validation samples
      num_val_batches_required = (num_valid_samples//(FLAGS.valid_batch_size * FLAGS.num_gpus)) + int((num_valid_samples%(FLAGS.valid_batch_size * FLAGS.num_gpus))!=0)
      for i in range(num_val_batches_required):
        hidden_representations_new, labels_new, metadata_new = initializer.sess.run([val_model.hidden_repr, val_model.label, val_model.metadata], feed_dict)
        hidden_representations = np.concatenate((hidden_representations, hidden_representations_new))
        labels = np.concatenate((labels, labels_new))
        metadata = np.concatenate((metadata, metadata_new))

      store_latent_vectors_as_df(output_dir, hidden_representations, labels, metadata)

    if 'psnr' in FLAGS.valid_mode:
      log_file = os.path.join(output_dir, 'psnr_log_' + str(dt.datetime.now()) + ".txt")

      psnr_reconstruction = []
      psnr_future = []

      orig_rec_mean = []
      orig_fut_mean = []


      num_val_batches_required = (num_valid_samples // (FLAGS.valid_batch_size * FLAGS.num_gpus)) + int(
        (num_valid_samples % (FLAGS.valid_batch_size * FLAGS.num_gpus)) != 0)
      for i in range(num_val_batches_required):
        print("processing batch " + str(i+1) + " of " + str(num_val_batches_required) + " total batches.")
        output_frames, orig_frames = initializer.sess.run([val_model.output_frames, val_model.val_batch], feed_dict)
        video_count = orig_frames.shape[0]

        for i in range(video_count):
          orig_rec_video_frames = np.asarray(orig_frames)[i, (FLAGS.image_range_start + FLAGS.encoder_length - FLAGS.decoder_reconst_length): (FLAGS.image_range_start +FLAGS.encoder_length), :, :, :3]
          orig_fut_video_frames = np.asarray(orig_frames)[i, (FLAGS.image_range_start + FLAGS.encoder_length):(FLAGS.image_range_start + FLAGS.encoder_length + FLAGS.decoder_future_length), :, :, :3]
          recon_video_frames = np.asarray(output_frames)[(FLAGS.encoder_length - FLAGS.decoder_reconst_length):FLAGS.encoder_length, i, :, :, :3]
          recon_video_frames = recon_video_frames.round().astype('uint8')
          future_video_frames = np.asarray(output_frames)[(FLAGS.encoder_length):(FLAGS.encoder_length + FLAGS.decoder_future_length), i, :, :, :3]
          future_video_frames = future_video_frames.round().astype('uint8')

          #for j in range(recon_video_frames.shape[0]):
          #  plt.imsave(os.path.join(output_dir, 'pic_' + str(i) + "_" + str(j)), recon_video_frames[j])

          psnr_reconstruction.append(np.asarray([metrics.peak_signal_to_noise_ratio(orig_frame, bgr_to_rgb(recon_frame), color_depth=255) for orig_frame, recon_frame in zip(orig_rec_video_frames, recon_video_frames)]))
          psnr_future.append(np.asarray([metrics.peak_signal_to_noise_ratio(orig_fut_frame, bgr_to_rgb(fut_frame), color_depth=255) for orig_fut_frame, fut_frame in zip(orig_fut_video_frames, future_video_frames)]))

          """ mean computations """
          # we compute the mean of a video sequence over its 5 rgb-frames (5,128,128,3), yielding a mean frame of shape (128,128,3)
          mean_frame_orig_rec = np.mean([frame for frame in orig_rec_video_frames], axis=0)
          mean_frame_orig_fut = np.mean([frame for frame in orig_fut_video_frames], axis=0)

          # we compute the psnr between the mean frame and each of the 5 rgb-frames
          orig_rec_mean.append(np.asarray([metrics.peak_signal_to_noise_ratio(mean_frame_orig_rec, bgr_to_rgb(recon_frame), color_depth=255) for recon_frame
                                          in recon_video_frames]))

          orig_fut_mean.append(np.asarray([metrics.peak_signal_to_noise_ratio(mean_frame_orig_fut, bgr_to_rgb(fut_frame), color_depth=255) for fut_frame
                                           in future_video_frames]))



      psnr_reconstruction_means = np.mean(np.stack(psnr_reconstruction), axis=0)
      psnr_rec_stds = np.std(np.stack(psnr_reconstruction), axis=0)

      psnr_future_means = np.mean(np.stack(psnr_future), axis=0)
      psnr_fut_stds = np.std(np.stack(psnr_future), axis=0)

      orig_rec_means = np.mean(np.stack(orig_rec_mean), axis=0)
      orig_rec_std = np.std(np.stack(orig_rec_mean), axis=0)

      orig_fut_means = np.mean(np.stack(orig_fut_mean), axis=0)
      orig_fut_std = np.std(np.stack(orig_fut_mean), axis=0)
      

      print(psnr_reconstruction_means)
      print(psnr_future_means)
      print(psnr_rec_stds)
      print(psnr_fut_stds)
      print(orig_rec_means)
      print(orig_fut_means)
      print(orig_rec_std)
      print(orig_fut_std)
      write_file_with_append(log_file, "mean psnr recon: " + str(psnr_reconstruction_means) + "\nmean psnr future: " + str(psnr_future_means) +
                                      "\nstd psnr recon: " + str(psnr_rec_stds) + "\nstd psnr future: " + str(psnr_fut_stds) +
                                      "\nmean of psnr between orig rec mean and rec frames: " + str(orig_rec_means) + "\nmean psnr between orig fut mean and fut frames: " + str(orig_fut_means) +
                                      "\nstd of psnr between orig rec mean and rec frames: " + str(orig_rec_std) + "\nstd of psnr between orig fut mean and fut frames: " + str(orig_fut_std)
                             )

      print("mean psnr recon: " + str(psnr_reconstruction_means) + "\nmean psnr future: " + str(psnr_future_means))
      tf.logging.info('Added psnr values to log file ' + str(log_file))

    if 'count_trainable_weights' in FLAGS.valid_mode:
      total_parameters = 0
      for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
          print(dim)
          variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
      print("Total parameters are:" + total_parameters)

    #summary_writer.add_summary(val_summary_str, 1)


  except tf.errors.OutOfRangeError:
    tf.logging.info('Done producing validation results -- iterations limit reached')
  except Exception as e:
    print("Exception occured:", e)
  finally:
    # When done, ask the threads to stop.
    initializer.coord.request_stop()

  # Wait for threads to finish.
  initializer.stop_session()
