from math import*
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import numpy as np
from matplotlib import pyplot as plt
import collections
import sklearn
from sklearn.manifold import TSNE
import scipy
import sklearn.metrics.pairwise as sk_pairwise
import pandas as pd

PICKLE_FILE_DEFAULT = '/data/rothfuss/training/03-28-17_15-50/valid_run/hidden_repr_df_04-04-17_15-18-53.pickle'

FLAGS = flags.FLAGS
flags.DEFINE_integer('numVideos', 1000, 'Number of videos stored in one single tfrecords file')
flags.DEFINE_string('pickle_file', PICKLE_FILE_DEFAULT, 'path of panda dataframe pickle file ')


def compute_hidden_representation_similarity_matrix(hidden_representations, labels, type):
  if type == 'cos':
    return compute_cosine_similarity_matrix(hidden_representations, labels)


def compute_cosine_similarity_matrix(hidden_representations, labels):
  assert hidden_representations.size > 0 and labels.size > 0
  distance_matrix = np.zeros(shape=(hidden_representations.shape[0], hidden_representations.shape[0]))
  class_overlap_matrix = np.zeros(shape=(hidden_representations.shape[0], hidden_representations.shape[0]))

  for row in range(distance_matrix.shape[0]):
    print(row)
    for column in range(distance_matrix.shape[1]):
      distance_matrix[row][column] = compute_cosine_similarity(hidden_representations[row], hidden_representations[column])
      class_overlap_matrix[row][column] = labels[row] == labels[column]
  return distance_matrix, class_overlap_matrix


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

def mean_vector(vector_list):
  """Computes the mean vector from a list of ndarrays
  :return mean_vector - ndarray
  :param vector_list - list of ndarrays with the same shape
  """
  assert isinstance(vector_list, collections.Iterable) and all(type(v) is np.ndarray for v in vector_list)
  mean_vector = np.zeros(vector_list[0].shape)
  for v in vector_list:
    mean_vector += v
  mean_vector = mean_vector / len(vector_list)
  assert type(mean_vector) is np.ndarray
  return mean_vector

def df_col_to_matrix(panda_col):
  """Converts a pandas dataframe column wherin each element in an ndarray into a 2D Matrix
  :return ndarray (2D)
  :param panda_col - panda Series wherin each element in an ndarray
  """
  panda_col = panda_col.map(lambda x: x.flatten())
  return np.vstack(panda_col)

def principal_components(df_col, n_components=2):
  pca = sklearn.decomposition.PCA(n_components)
  X = df_col_to_matrix(df_col) #reshape dataframe column consisting of ndarrays as 2D matrix
  return pca.fit_transform(X)

def distance_classifier(df):
  classes = list(set(df['shape']))
  mean_vectors = [mean_vector(list(df[df['shape']==c]['hidden_repr'])) for c in classes]
  for i, c in enumerate(classes):
    df['dist_euc_'+str(i)] = (df['hidden_repr'] - mean_vectors[i]).apply(lambda x: np.linalg.norm(x.flatten()))
    df['dist_cos_'+str(i)] = [scipy.spatial.distance.cosine(v.flatten(), mean_vectors[i].flatten()) for v in df['hidden_repr']]

  df['class'] = [classes.index(s) for s in df['shape']]
  df['eucl_class_pred'] = df[['dist_euc_'+str(i) for i in range(len(classes))]].idxmin(axis=1).apply(lambda x: int(x[-1]))
  df['cos_class_pred'] = df[['dist_cos_' + str(i) for i in range(len(classes))]].idxmin(axis=1).apply(lambda x: int(x[-1]))

  df['eucl_correct'] = df['eucl_class_pred'] == df['class']
  df['cos_correct'] = df['cos_class_pred'] == df['class']

  print('Accuracy with euclidian distance:', df['eucl_correct'].mean())
  print('Accuracy with cosine distance:', df['cos_correct'].mean())
  return df

def logistic_regression(df):
  X = df_col_to_matrix(df['hidden_repr'])
  Y = df['shape']
  lr = sklearn.linear_model.LogisticRegression()
  lr.fit(X,Y)
  return lr.score(X,Y)

def svm(df):
  X = df_col_to_matrix(df['hidden_repr'])
  Y = df['shape']
  svc = sklearn.svm.SVC(kernel='linear')
  svc.fit(X,Y)
  return svc.score(X, Y)

def avg_distance(df):
  same_class_dist_array = []
  out_class_dist_array = []
  vectors = list(df['hidden_repr'])
  labels = list(df['shape'])
  for i, (v1, l1) in enumerate(zip(vectors, labels)):
    print(i)
    for v2, l2 in zip(vectors, labels):
      distance = compute_cosine_similarity(v1, v2)
      if l1==l2:
        same_class_dist_array.append(distance)
      else:
        out_class_dist_array.append(distance)
  return np.mean(same_class_dist_array), np.mean(out_class_dist_array)

def main():
  #visualize_hidden_representations()
  #app.run()
  df = pd.read_pickle(FLAGS.pickle_file)
  #print(df)
  #result = svm(df)
  #print(result)
  print(avg_distance(df))

if __name__ == "__main__":
  main()