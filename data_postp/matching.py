from math import *
import os, collections, scipy, itertools, multiprocessing, shutil
from pprint import pprint
import numpy as np
import scipy, pickle, json
from ast import literal_eval as make_tuple
import sklearn, sklearn.ensemble
import pandas as pd
import utils.io_handler as io_handler

from data_postp.similarity_computations import df_col_to_matrix, transform_vectors_with_inter_class_pca, compute_cosine_similarity, inter_class_pca, top_n_accuracy



#PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/valid_run/metadata_and_hidden_rep_df_08-14-17_17-38-26_all_augmented_valid.pickle'
PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/valid_run/metadata_and_hidden_rep_df_08-12-17_02-50-16_selected_10_classes_eren_augmented_valid.pickle'
#PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/valid_run/matching/metadata_and_hidden_rep_df_08-09-17_21-19-52_half_actions_new_episodes_optical_flow_valid.pickle'

ARMAR_EXPERIENCES_BASE_DIR = '/data/rothfuss/data/ArmarExperiences/video_frames'
BASE_DIR_20BN = '/PDFData/rothfuss/data/20bn-something-something-v1'
PICKLE_ARMAR_EXPERIENCES_MEMORY = '/data/rothfuss/data/ArmarExperiences/hidden_reps/hidden_repr_memory.pickle'
PICKLE_ARMAR_EXPERIENCES_MEMORY_AUGMENTED = '/data/rothfuss/data/ArmarExperiences/hidden_reps/hidden_repr_memory_augmented.pickle'
PICKLE_ARMAR_EXPERIENCES_QUERY = '/data/rothfuss/data/ArmarExperiences/hidden_reps/hidden_repr_query.pickle'
#PICKLE_ARMAR_EXPERIENCES_QUERY = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/valid_run/matching/metadata_and_hidden_rep_df_08-10-17_12-17-36_half_actions_old_episodes_optical_flow_valid.pickle'
PICKLE_ARMAR_EXPERIENCES_HALF_ACTION = '/data/rothfuss/data/ArmarExperiences/hidden_reps/hidden_repr_query_half.pickle'

def closest_vector_analysis_composite(df, df_query, base_dir, target_dir, n_pca_matching=20, n_pca_classifier=50,
                                                 class_column='category', n_closest_matches=5, lambda_weight=0.5, augmented=False, output_type='gif'):

  assert output_type in ['gif', 'png'], "output type must be either gif or png"
  #Preparation Part 1: prepare classifier and classifier_pca
  #train logistic regression on pca components of hidden_reps in df --> returns classifier and pca object
  classifier, pca_classifier, _, _ = train_and_dump_classifier(df, class_column=class_column, n_components=n_pca_classifier)

  # generate pca transformed query df for classifiaction pca
  df_query_classification = df_query.copy()
  transformed_vectors_as_matrix = pca_classifier.transform(df_col_to_matrix(df_query['hidden_repr']))
  df_query_classification['hidden_repr'] = np.split(transformed_vectors_as_matrix, transformed_vectors_as_matrix.shape[0])

  #Preparation Part 2: prepare pca transform for matching
  df_pca_matching, df_query_matching, pca_matching = transform_vectors_with_inter_class_pca(df, df_2=df_query, n_components=n_pca_matching, return_pca_object=True)

  # removing augmented samples from df_pca_matching (videos with suffix _9 are originals)
  if augmented:
    df_pca_matching = df_pca_matching[df_pca_matching["id"].str.contains("_9")]
    #df_query_matching = df_query_matching[df_query_matching["id"].str.contains("_9")]

  #make parent dir for files
  target_dir = os.path.join(target_dir, 'matching_pca_%i_classifier_%i_%.2f'%(n_pca_matching,n_pca_classifier,lambda_weight))
  print(target_dir)
  os.mkdir(target_dir)
  print("Created directory:", target_dir)
  # Production: Iterate over queries in df_query
  for hidden_repr_matching, hidden_repr_classification, label, category in zip(df_query_matching['hidden_repr'], df_query_classification['hidden_repr'],
                                                                               df_query_matching['label'], df_query_matching[class_column]):
    try:
      class_prob_dict = dict(zip(classifier.classes_, classifier.predict_proba(hidden_repr_classification).flatten().tolist()))
      pprint(sorted(class_prob_dict.items(), key=lambda tup: tup[1], reverse=True)[:5])
      composite_scores= []
      for v, l, v_id in zip(df_pca_matching['hidden_repr'], df_pca_matching[class_column], df_pca_matching['video_id']):
        class_prob = class_prob_dict[l]
        cos_sim = compute_cosine_similarity(v, hidden_repr_matching)
        composite_score = (1-lambda_weight) * class_prob + lambda_weight * cos_sim
        composite_scores.append((composite_score, l, int(v_id)))

      sorted_scores = sorted(composite_scores, key=lambda tup: tup[0], reverse=True)

      closest_vectors = sorted_scores[:n_closest_matches]

      label = label.replace("_9", "")
      print(label, category, n_closest_matches)
      #pprint(closest_vectors)

      # File transfer - make directiry label_dir and copy the matched videos into the directory
      label_dir = os.path.join(target_dir, label)
      os.mkdir(label_dir)
      try:
        if output_type is 'gif':
          io_handler.convert_frames_to_gif(os.path.join(base_dir, str(label)), image_type='.jpg', fps=15, gif_file_path=os.path.join(label_dir, 'query_' + str(label) + ': ' + category))
        else:
          shutil.copytree(os.path.join(base_dir, str(label)), os.path.join(label_dir, 'query_' + str(label) + ': ' + category))
      except Exception as e:
        print(e)
        pass
      for i, (cos_dist, v_label, v_id) in enumerate(closest_vectors):
        try:
          if output_type is 'gif':
            io_handler.convert_frames_to_gif(os.path.join(base_dir, str(v_id)), image_type='.jpg', fps=15,
                                             gif_file_path=os.path.join(label_dir, "match_%i:%s_%.4f_%s" % (i, str(v_id), cos_dist, v_label)))
          else:
            shutil.copytree(os.path.join(base_dir, str(v_id)), os.path.join(label_dir, str(v_id)))
        except Exception as e:
          pass
        if output_type is 'png':
          os.rename(os.path.join(label_dir, str(v_id)),
                    os.path.join(label_dir, "match_%i:%s_%.4f_%s" % (i, str(v_id), cos_dist, v_label)))
    except Exception as e:
      print(e)

def train_and_dump_classifier(df, dump_path=None, class_column="category", classifier=sklearn.linear_model.LogisticRegression(n_jobs=-1),
                              n_components=500, train_split_ratio=0.8):

  # separate dataframe into test and train split
  if train_split_ratio < 1.0:
    msk = np.random.rand(len(df)) < train_split_ratio
    test_df = df[~msk]
    train_df = df[msk]
  else: #no testing required -> make test_df dummy
    train_df = df.copy()
    test_df = df.copy()

  if n_components > 0:
    pca = inter_class_pca(df, class_column=class_column, n_components=n_components)
    X_train = pca.transform(df_col_to_matrix(train_df['hidden_repr']))
    X_test = pca.transform(df_col_to_matrix(test_df['hidden_repr']))

  else:
    X_train = df_col_to_matrix(train_df['hidden_repr'])
    X_test = df_col_to_matrix(test_df['hidden_repr'])

  y_train = np.asarray(list(train_df[class_column]))
  y_test = np.asarray(list(test_df[class_column]))


  #fit model:
  print("Training classifier")
  classifier.fit(X_train, y_train)

  #calculate accuracy
  if train_split_ratio < 1.0:
    acc = classifier.score(X_test, y_test)
    print("Accuracy:", acc)
    top5_acc = top_n_accuracy(classifier, X_test, y_test, n=3)
    print("Top5 Accuracy:", top5_acc)
  else:
    acc, top5_acc = None, None

  #dump classifier
  if dump_path:
    classifier_dump_path = os.path.join(os.path.dirname(dump_path), 'classifier.pickle')
    with open(classifier_dump_path, 'wb') as f:
      pickle.dump(classifier, f)
    print("Dumped classifier to:", classifier_dump_path)

  return classifier, pca, acc, top5_acc


def episodic_memory_20bn(augmented=False):
  df = pd.read_pickle(PICKLE_FILE_TEST)

  df_val = pd.read_pickle(
    '/data/rothfuss/data/ArmarExperiences/hidden_reps/hidden_repr_query_half.pickle')

  # target_dir = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/valid_run/matching/composite_matching/matching_armar_old'
  target_dir = '/data/rothfuss/data/20bn-something/matching/not_augmented'
  input_image_dir = '/common/temp/toEren/4PdF_ArmarSampleImages/HalfActions/fromEren/Originals'

  if not augmented:
    df = df[df["id"].str.contains("_9")] #SET IF not augmentation wanted

  df['label'] = df['label_x']
  msk = np.random.rand(len(df)) < 0.8
  test_df = df[~msk]
  train_df = df[msk]


  print(len(df), len(test_df), len(train_df))

  df_val['category'] = ['armar_setting' for _ in range(len(df_val))]
  closest_vector_analysis_composite(train_df, test_df, BASE_DIR_20BN, target_dir,
                                    class_column='category',
                                    n_pca_matching=20, n_pca_classifier=200, lambda_weight=0.5, augmented=augmented)

def episodic_memory_armar(augmented=False):
  memory_df = pd.read_pickle(PICKLE_ARMAR_EXPERIENCES_MEMORY)
  #query_df = pd.read_pickle(PICKLE_ARMAR_EXPERIENCES_HALF_ACTION)
  query_df = pd.read_pickle(PICKLE_ARMAR_EXPERIENCES_QUERY)
  query_df['label'] = query_df['id']
  query_df['category'] = query_df['label']

  target_dir = '/data/rothfuss/data/ArmarExperiences/matching/complete_actions/not_augmented'

  closest_vector_analysis_composite(memory_df, query_df, ARMAR_EXPERIENCES_BASE_DIR, target_dir, class_column='category',
                                    n_pca_matching=50, n_pca_classifier=200, lambda_weight=0.5, augmented=augmented)


def main():
  #episodic_memory_armar(augmented=False)
  episodic_memory_20bn(augmented=False)

  matching_dir =  '/data/rothfuss/data/ArmarExperiences/matching/new_half_actions/not_augmented/matching_pca_50_classifier_200_0.50_gif'
  io_handler.frames_to_gif_in_dir_tree(matching_dir)



if __name__ == "__main__":
  main()