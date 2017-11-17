import os
from pymongo import MongoClient
import gridfs

class Memory:

  def __init__(self, collection_name):
    #TODO establish connection to mongodb
    self.client = MongoClient()
    self.db = self.client.episodic_memory
    self.fs = gridfs.GridFS(self.db)

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

  def get_all_hidden_reps(self):
    '''
    queries all episodes
    :return: triple of three lists of the same length (ids, hidden_reps, metadata_dicts)
    '''
    #TODO
    pass

  def get_episode(self, id):
    '''
    queries a single episode by its id
    :return: tuple of four objects (id, hidden_reps, metadata_dicts, video_episode)
    '''
    # TODO
    pass