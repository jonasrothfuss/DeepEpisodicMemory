import os, sklearn, collections
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals import joblib
from utils import io_handler
import pandas as pd
import numpy as np
from data_postp import similarity_computations


class Memory:

  def __init__(self, memory_df, label_col="category", inter_class_pca_path=None):
    ''' Initializes the Memory class
    :param memory_df: pandas dataframe that contains the hidden_reps, labels and video_paths of the episodes
    :param label_col:
    '''
    assert label_col in memory_df.columns, str(label_col) + ' must be a dataframe category'
    assert isinstance(memory_df, pd.DataFrame)

    self.memory_df = memory_df
    self.hidden_reps = np.stack([h.flatten() for h in memory_df['hidden_repr']])
    self.labels = memory_df[label_col]

    #get fitted PCA object
    if inter_class_pca_path:
      assert os.path.isfile(inter_class_pca_path)
      self.inter_class_pca = joblib.load(inter_class_pca_path)
    else:
      self.inter_class_pca = fit_inter_class_pca(self.hidden_reps, self.labels, n_components=50, verbose=False)

    # PCA transform the hidden_reps
    self.hidden_reps_transformed = self.inter_class_pca.transform(self.hidden_reps)

  def store_episodes(self, ids, hidden_reps, metadata_dicts, video_file_paths):
    '''
    stores provided episodes in mongo database (memory)
    :param ids:
    :param hidden_reps:
    :param metadata_dicts:
    :param video_file_path:
    :return mongodb_ids corresponding to the persisted documents
    '''
    assert len(ids) == len(hidden_reps) == len(metadata_dicts) == len(video_file_paths)
    assert all([os.path.isfile(path) for path in video_file_paths])

    #TODO: store as document in mongodb

    #for i, id in enumerate(ids):
    #  if id not in self.memory_df:

  def get_episode(self, id):
    '''
    queries a single episode by its id
    :return: tuple of four objects (id, hidden_reps, metadata_dicts, video_episode)
    '''
    # TODO
    #pass

  def matching(self, query_hidden_repr, n_closest_matches = 5, use_transform=True):
    '''
    finds the closest vector matches (cos_similarity) for a given query vector
    :param query_hidden_repr: the query vector
    :param n_clostest_matches: (optional) the number of closest matches returned
    :param use_transform: boolean that denotes whether the matching shall performed on transformed hidden vectors
    :return: two arrays, the first containing the n_clostest_matches and the second containing the hidden_reps
    '''

    query_hidden_repr = np.expand_dims(query_hidden_repr, axis=0)
    if use_transform:
      memory_hidden_reps = self.hidden_reps_transformed
      query_hidden_repr = self.inter_class_pca.transform(query_hidden_repr)
    else:
      memory_hidden_reps = self.hidden_reps
    assert memory_hidden_reps.ndim == 2 #memory_hidden_reps must have shape (n_episodes, n_dim_repr)
    assert query_hidden_repr.ndim == 2 #query_hidden_repr must have shape (1, n_dim_repr)
    assert memory_hidden_reps.shape[1] == query_hidden_repr.shape[1]

    cos_distances = pairwise_distances(memory_hidden_reps, query_hidden_repr, metric='cosine')[:,0] #shape(n_episodes, 1)
    # get indices of n maximum values in ndarray
    indices_closest = cos_distances.argsort()

    return indices_closest, cos_distances[indices_closest], memory_hidden_reps[indices_closest]

def mean_vectors_of_classes(hidden_reps, labels):
  """
  Computes mean vector for each class in class_column
  :param hidden_reps: list of hidden_vectors
  :param labels: list of labels corresponding to the hidden_reps
  :return: dataframe with labels as index and mean vectors for each class
  """
  vector_dict = collections.defaultdict(list)
  for label, vector in zip(labels, hidden_reps):
    vector_dict[label].append(vector)
  return pd.DataFrame.from_dict(dict([(label, np.mean(vectors, axis=0)) for label, vectors in vector_dict.items()]),
                                orient='index')

def fit_inter_class_pca(hidden_reps, labels, n_components=50, verbose=False, dump_path=None):
  '''
  Fits a PCA on mean vectors of classes denoted by self.labels
  :param n_components: number of pca components
  :param verbose: verbosity
  :param dump_path: if provided, the pca object is dumped to the provided path
  :return pca: fitted pca object
  '''
  mean_vectors = mean_vectors_of_classes(hidden_reps, labels)
  pca = sklearn.decomposition.PCA(n_components).fit(mean_vectors)
  if verbose:
    print("PCA (n_components= %i: relative variance explained:" % n_components, np.sum(pca.explained_variance_ratio_))
  if dump_path:
    joblib.dump(pca, dump_path)
  return pca

if __name__ == '__main__':
  memory_df = pd.read_pickle('/common/homes/students/rothfuss/Documents/Episodic_Memory/Armar_Experiences/metadata_and_hidden_rep_df_11-24-17_14-00-34.pickle')
  #m = Memory(memory_df)
  #query = m.hidden_reps[2,:]
  #print(m.matching(query))
  #result = m.matching(m.hidden_reps[1,:])

  video_names = ["original_clip_%s.gif"%str(i) for i in memory_df['id']]
  print(video_names)
  files = io_handler.files_from_directory('/common/homes/students/rothfuss/Documents/Episodic_Memory/Armar_Experiences', '*.gif')
  files = sorted(files)

  print(set(video_names)-set(files))