import os, sklearn
from sklearn.metrics.pairwise import pairwise_distances
from pymongo import MongoClient
import gridfs
import pandas as pd
import numpy as np

class Memory:

  def __init__(self, memory_df):
    #TODO establish connection to mongodb
    #self.client = MongoClient()
    #self.db = self.client.episodic_memory
    #self.fs = gridfs.GridFS(self.db)
    self.memory_df = memory_df
    self.hidden_reps = np.stack([h.flatten() for h in memory_df['hidden_repr']])
    # add pca transf as pickle

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



  def get_all_hidden_reps(self):
    '''
    queries all episodes
    :return: triple of three lists of the same length (ids, hidden_reps, metadata_dicts)
    '''
    #TODO
    pass
    #return (ids, hidden_reps, metadata_dicts)

  def get_episode(self, id):
    '''
    queries a single episode by its id
    :return: tuple of four objects (id, hidden_reps, metadata_dicts, video_episode)
    '''
    # TODO
    #pass

  def matching(self, query_hidden_repr, n_closest_matches = 5):
    '''
    finds the closest vector matches (cos_similarity) for a given query vector
    :param query_hidden_repr: the query vector
    :param n_clostest_matches: (optional) the number of closest matches returned
    :return: two arrays, the first containing the n_clostest_matches and the second containing the hidden_reps
    '''
    #ids, memory_hidden_reps,  = self.get_all_hidden_reps()

    query_hidden_repr = np.expand_dims(query_hidden_repr, axis=0)

    assert self.hidden_reps.ndim == 2 #memory_hidden_reps must have shape (n_episodes, n_dim_repr)
    assert query_hidden_repr.ndim == 2 #query_hidden_repr must have shape (1, n_dim_repr)

    cos_distances = pairwise_distances(self.hidden_reps, query_hidden_repr)[:,0] #shape(n_episodes, 1)
    # get indices of n maximum values in ndarray
    indices_closest = cos_distances.argsort()[-n_closest_matches:][::-1]


    return cos_distances[indices_closest], self.hidden_reps[indices_closest]



if __name__ == '__main__':
  memory_df = pd.read_pickle('/data/rothfuss/data/ArmarExperiences/hidden_reps/armar_experiences_20bn_memory.pickle')
  m = Memory(memory_df)
  result = m.matching(m.hidden_reps[1,:])
  print(result)
