import os, re, math

def get_iter_from_pretrained_model(checkpoint_file_name):
  ''' extracts the iterator count of a dumped checkpoint from the checkpoint file name
  :param checkpoint_file_name: name of checkpoint file - must contain a
  :return: iterator number
  '''
  file_basename = os.path.basename(checkpoint_file_name)
  assert re.compile('[A-Za-z0-9]+[-][0-9]+').match(file_basename)
  idx = re.findall(r'-\b\d+\b', file_basename)[0][1:]
  return int(idx)


def learning_rate_decay(initial_learning_rate, itr, decay_factor=0.0):
  return initial_learning_rate * math.e**(- decay_factor * itr)
