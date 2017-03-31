from math import*
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
import sklearn.metrics.pairwise as sk_pairwise



FLAGS = flags.FLAGS
flags.DEFINE_integer('numVideos', 1000, 'Number of videos stored in one single tfrecords file')


def compute_hidden_representation_similarity_matrix(hidden_representations, labels, type):
  if type == 'cos':
    return compute_cosine_similarity_matrix(hidden_representations, labels)


def compute_cosine_similarity_matrix(hidden_representations, labels):
  assert hidden_representations.size > 0 and labels.size > 0
  matrix = np.zeros(shape=(hidden_representations.shape[0], hidden_representations.shape[0]))

  for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
      matrix[row][column] = compute_cosine_similarity(hidden_representations[row], hidden_representations[column])

  return matrix


def compute_hidden_representation_similarity(hidden_representations, labels, type=None, video_id_a_string = None, video_id_b_string = None):
  """Computes the similarity between two or all vectors, depending on the number of arguments with which the function is called.

  :return if type: 'cos' -> scalar between 0 and 1
  :param hidden_representations: numpy array containing the activations
  :param labels: the corresponding video id's
  :param type: specifies the type of similarity to be computed, possible values: 'cos' |
  :param video_id_x_string: describes the vector video_id to be used for the computation, needs to be existing in 'labels'


  Similarity between two vectors: specify all four arguments, including video_id_a_string and video_id_b_string, e.g.
  video_id_a_string=b'064195' and video_id_a_string=b'004323'

  Similarity between all vectors: specify only the first three arguments (leave out video id's)
  """
  assert hidden_representations.size > 0 and labels.size > 0

  if type == 'cos':
    if video_id_a_string is not None or video_id_b_string is not None:
      # compute similarity for the two given video_id's
      video_id_a_idx = np.where(labels == video_id_a_string)
      video_id_b_idx = np.where(labels == video_id_b_string)

      vector_a = hidden_representations[video_id_a_idx[0]]
      vector_b = hidden_representations[video_id_b_idx[0]]

      return compute_cosine_similarity(vector_a, vector_b)
      # return sk_pairwise.cosine_similarity(vector_a.reshape(-1, 1) vector_b.reshape(-1, 1))

    else:
      # compute similarity matrix for all vectors
      return compute_hidden_representation_similarity_matrix(hidden_representations, labels, 'cos')


def compute_cosine_similarity(vector_a, vector_b):
  if vector_a.ndim > 2:
    vector_a = vector_a.flatten()
  if vector_b.ndim > 2:
    vector_b = vector_b.flatten()

  numerator = sum(a * b for a, b in zip(vector_a, vector_b))
  denominator = square_rooted(vector_a) * square_rooted(vector_b)
  return round(numerator / float(denominator), 3)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def visualize_hidden_representations(hidden_representations, labels):
  X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
  model = TSNE(n_components=2, random_state=0)
  data = model.fit_transform(X)

  # plot the result
  plt.scatter(data[:, 0], data[:, 1])
  plt.show()


def main():
  visualize_hidden_representations()
  #app.run()

if __name__ == "__main__":
  main()