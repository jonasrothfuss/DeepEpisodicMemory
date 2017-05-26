from math import*
import os
from datetime import datetime
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#from matplotlib import pyplot as plt
import collections
import sklearn
from sklearn.manifold import TSNE
import scipy
import sklearn.metrics.pairwise as sk_pairwise
import pandas as pd
import itertools
import seaborn as sn

# PICKLE_FILE_DEFAULT = './metadata_and_hidden_rep_df.pickle' #'/localhome/rothfuss/data/df.pickle'
#/localhome/rothfuss/training/04-27-17_20-40/valid_run/metadata_and_hidden_rep_df_05-04-17_09-06-48.pickle
#PICKLE_FILE_DEFAULT = '/localhome/rothfuss/training/04-27-17_20-40/valid_run/metadata_and_hidden_rep_df_05-04-17_09
# -06-48.pickle'
#PICKLE_FILE_DEFAULT ='/Users/fabioferreira/Google Drive/Studium/Master/Praxis der Forschung/private
# repository/DeepEpisodicMemory/data/hidden_repr_df_04-04-17_15-18-53.pickle'
PICKLE_FILE_DEFAULT = '/Users/fabioferreira/Google Drive/Studium/Master/Praxis der Forschung/private repository/DeepEpisodicMemory/data/metadata_and_hidden_rep_df_05-04-17_09-06-48.pickle'
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
  "vectors similar: cosine similarity is 1"
  if vector_a.ndim > 2:
    vector_a = vector_a.flatten()
  if vector_b.ndim > 2:
    vector_b = vector_b.flatten()

  numerator = sum(a * b for a, b in zip(vector_a, vector_b))
  denominator = square_rooted(vector_a) * square_rooted(vector_b)
  return round(numerator / float(denominator), 3)

def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def visualize_hidden_representations(pickle_hidden_representations):
    # shatter 1000x8x8x16 to 1000x1024 dimensions
    labels = list(pickle_hidden_representations['shape'])
    values = df_col_to_matrix(pickle_hidden_representations['hidden_repr'])
    Y_data = np.asarray(labels)


    for i, entry in enumerate(Y_data):
        if  entry == 'square':
            Y_data[i] = 's'
        if entry == 'circular':
            Y_data[i] = 'o'
        if entry == 'triangle':
            Y_data[i] = '^'

    model = TSNE(n_components=23, random_state=0, method='exact')
    data = model.fit_transform(values)

    for xp, yp, m in zip(data[:, 0], data[:, 1], Y_data):
        plt.scatter([xp],[yp], marker=m)

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
  """Performs a Principal Component Analysis (fit to the data) and then returns the returns the data transformed corresponding to the n_components
  first principal components

  :return ndarray - provided data projected onto the first n principal components
  :param df_col: pandas Series (df column)
  :param n_components: number of principal components that shall be used for the data transformatation (dimensionality reduction)
  """
  pca = sklearn.decomposition.PCA(n_components)
  X = df_col_to_matrix(df_col) #reshape dataframe column consisting of ndarrays as 2D matrix
  return pca.fit_transform(X)

def distance_classifier(df):
  """Handmade classifier that computes the center of each class and then assigns each datapoint the class with the closest centerpoint
   """
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
  """ Fits a logistic regression model (MaxEnt classifier) on the data and returns the training accuracy"""
  X = df_col_to_matrix(df['hidden_repr'])
  Y = df['shape']
  lr = sklearn.linear_model.LogisticRegression()
  lr.fit(X,Y)
  return lr.score(X,Y)

def svm(df):
  """ Fits a SVM with linear kernel on the data and returns the training accuracy"""
  X = df_col_to_matrix(df['hidden_repr'])
  Y = df['shape']
  svc = sklearn.svm.SVC(kernel='linear')
  svc.fit(X,Y)
  return svc.score(X, Y)

def avg_distance(df, similarity_type = 'cos'):
  """ Computes the average pairwise distance (a: euclidean, b:cosine) of data instances within
  the same class and with different class labels
  :returns avg pairwise distance of data instances with the same class label
  :returns avg pairwise distance of data instances with different class label
  :param dataframe including the colummns hidden_repr and shape
  :param similarity_type - either 'cos' or 'euc'
  """
  assert similarity_type in ['cos', 'euc']
  assert 'hidden_repr' in list(df) and 'shape' in list(df)

  same_class_dist_array = []
  out_class_dist_array = []
  vectors = list(df['hidden_repr'])
  labels = list(df['shape'])
  for i, (v1, l1) in enumerate(zip(vectors, labels)):
    print(i)
    for v2, l2 in zip(vectors, labels):
      if similarity_type == 'cos':
        distance = compute_cosine_similarity(v1, v2)
      elif similarity_type == 'euc':
        distance = np.sqrt(np.sum((v1.flatten() - v2.flatten())**2))
      if l1==l2:
        same_class_dist_array.append(distance)
      else:
        out_class_dist_array.append(distance)
  return np.mean(same_class_dist_array), np.mean(out_class_dist_array)

def similarity_matrix(df, df_label_col,  similarity_type= 'cos'):
  assert 'hidden_repr' in list(df) and df_label_col in list(df)
  assert similarity_type in ['cos', 'euc']
  labels = list(set(df[df_label_col]))
  n = len(labels)
  sim_matrix = np.zeros([n, n])
  for i in range(n):
    for j in range(n):
      print(i,j, labels[i], labels[j])
      vectors1 = list(df[df[df_label_col] == labels[i]]['hidden_repr'])
      vectors2 = list(df[df[df_label_col] == labels[j]]['hidden_repr'])
      sim = []
      for k, v1 in enumerate(vectors1):
        for v2 in vectors2:
          if similarity_type is 'cos':
            sim.append(compute_cosine_similarity(v1, v2))
          else:
            sim.append(np.sqrt(np.sum((v1.flatten() - v2.flatten())**2)))
      assert(len(sim)==len(vectors1)*len(vectors2))
      print(np.mean(sim))
      sim_matrix[i,j] = np.mean(sim)
  df_cm = pd.DataFrame(sim_matrix, index=labels, columns=labels)
  print(df_cm)
  df_cm.to_pickle(os.path.join(os.path.dirname(FLAGS.pickle_file),'sim_matrix_' + similarity_type +'.pickle'))
  plt.figure(figsize=(64, 64))
  sn.set(font_scale=10)
  ax = sn.heatmap(df_cm, annot=True, annot_kws={"size":90})
  if n > 5:
    #rotate x-axis labels
    for item in ax.get_xticklabels():
      item.set_rotation(90)
  heatmap_file_name = os.path.join(os.path.dirname(FLAGS.pickle_file),'sim_matrix_' + df_label_col + '_'+  similarity_type +'.png')
  plt.savefig(heatmap_file_name, dpi=100)
  print('Dumped Heatmap to:', heatmap_file_name)
  plt.show()
  return sim_matrix

def plot_similarity_shape_motion_matrix(df):
  """This function computes the large square confusion matrix with the dimensions being of shape (num_shapes * num_directions) and plots it using matplotlib"""

  different_shapes = df['shape'].unique()
  different_directions = df['motion_location'].unique()
  large_confusion_matrix = np.zeros(shape=(len(different_shapes) * len(different_directions), len(different_shapes) * len(different_directions)))


  x = 0
  y = 0

  for (shape1, shape2) in list(itertools.product(different_shapes, different_shapes)):
    print('x:', x)
    print('y:', y)
    if(shape1 == 'circular' and shape2 == 'circular'): x = 0; y = 0
    if(shape1 == 'circular' and shape2 == 'triangle'): x = 0; y = 8
    if(shape1 == 'circular' and shape2 == 'square'): x = 0; y = 16
    if(shape1 == 'triangle' and shape2 == 'circular'): x = 8; y = 0
    if(shape1 == 'triangle' and shape2 == 'triangle'): x = 8; y = 8
    if(shape1 == 'triangle' and shape2 == 'square'): x = 8; y = 16
    if(shape1 == 'square' and shape2 == 'circular'): x = 16; y = 0
    if(shape1 == 'square' and shape2 == 'triangle'): x = 16; y = 8
    if(shape1 == 'square' and shape2 == 'square'): x = 16; y = 16
    print('computing matrix for: ' + shape1, shape2)
    intermediate_matrix = compute_small_confusion_matrix(df, shape1, shape2)
    print('inserting small into large confusion matrix')
    large_confusion_matrix[x:x+intermediate_matrix.shape[0], y:y+intermediate_matrix.shape[1]] = intermediate_matrix


  #print(large_confusion_matrix)

  df_cm = pd.DataFrame(large_confusion_matrix, index = [i for i in np.concatenate([different_directions, different_directions, different_directions])],
                       columns = [i for i in np.concatenate([different_directions, different_directions, different_directions])])
  plt.figure(figsize=(64,64))
  sn.heatmap(df_cm, annot=True)
  heatmap_file_name = os.path.join(os.path.dirname(FLAGS.pickle_file),'large_sim_matrix.png')
  plt.savefig(heatmap_file_name, dpi=100)
  sn.plt.show()

def compute_small_confusion_matrix(df, shape1, shape2):
  """Used to compute a square confusion matrix with the dimensions being of shape
  (num_directions_of_shape1 * num_directions_of_shape2). It
   1st: computes the similarity between every vector given by df,
        i.e. for every motion combination of the individual shapes
   2nd: computes the mean values for every motion combination
   3rd: assigns the mean values to a numpy array and returns it
  """

  subset_of_shape_1 = df.loc[df['shape'] == shape1]
  subset_of_shape_2 = df.loc[df['shape'] == shape2]

  different_directions = subset_of_shape_1['motion_location'].unique()

  vectors_1 = list(subset_of_shape_1['hidden_repr'])
  vectors_2 = list(subset_of_shape_2['hidden_repr'])
  labels_1 = list(subset_of_shape_1['shape'])
  labels_2 = list(subset_of_shape_2['shape'])
  motions_1 = list(subset_of_shape_1['motion_location'])
  motions_2 = list(subset_of_shape_2['motion_location'])

  # initialize the resulting matrix
  confusion_matrix = np.zeros(shape=(len(different_directions), len(different_directions)))

  # use a dictionary of lists for all shape-motion combinations to collect the similarity values (keys are for example 'circular_circular_top_leftbottom')
  similarity_lists = {}
  for (shape1, shape2, direction1, direction2) in list(itertools.product([shape1], [shape2], different_directions, different_directions)):
    # adding to list is slow but no extra increment needed for number of elements
    similarity_lists[str(shape1) + '_' + str(shape2) + '_' + str(direction1) + '_' + str(direction2)] = []

  # iterate through the hidden representations, assign the to the dictionary lists appropriately
  for i, (v1, l1, m1) in enumerate(zip(vectors_1, labels_1, motions_1)):
    for j, (v2, l2, m2) in enumerate(zip(vectors_2, labels_2, motions_2)):
      distance = compute_cosine_similarity(v1, v2)
      similarity_lists.get(str(l1) + '_' + str(l2) + '_' + str(m1) + '_' + str(m2)).append(distance)

  # compute the mean for every shape-motion combination list
  similarity_lists_means = {}
  for k, v in similarity_lists.items():
    if float(sum(v)) is not None:
      similarity_lists_means[k] = float(sum(v)) / len(v)

  # assign the means to the previously initialized matrix
  for i, direction1 in enumerate(different_directions):
    for j, direction2 in enumerate(different_directions):
      confusion_matrix[i,j] = similarity_lists_means.get(str(shape1) + '_' + str(shape2) + '_' + str(direction1) + '_' + str(direction2))

  return confusion_matrix

def classifier_analysis(df):
  string_to_dump = str(datetime.now()) + '\n' + '---- SVM ----\nAccuracy: ' + str(svm(df)) + '\n' \
                   + '---- LogisticRegression ----' + '\n' + 'Accuracy: ' + str(logistic_regression(df)) + '\n'
  dump_file_name = os.path.join(os.path.dirname(FLAGS.pickle_file),'classifier_analysis' + '.txt')
  print(string_to_dump)
  with open(dump_file_name, 'w') as file:
    file.write(string_to_dump)


def main():
  #visualize_hidden_representations()
  #app.run()
  df = pd.read_pickle(FLAGS.pickle_file)
  print(df)
  visualize_hidden_representations(df)


  #similarity_matrix(df, "shape")
  #similarity_matrix(df, "motion_location")
  #classifier_analysis(df)
  #plot_similarity_shape_motion_matrix(df)


  #print(similarity_matrix(df, 'shape'))

  #plot_similarity_shape_motion_matrix(df)
  #print(svm(df))
  #print(logistic_regression(df))
  #print(avg_distance(df, 'cos'))

if __name__ == "__main__":
  main()
