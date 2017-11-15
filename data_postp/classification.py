import numpy as np
import pandas as pd
import utils.io_handler as io_handler
import os, collections, scipy, itertools, multiprocessing, shutil
import scipy, pickle, json
import sklearn, sklearn.ensemble
import csv
from datetime import datetime
from pprint import pprint
from tensorflow.python.platform import flags

from data_postp.matching import train_and_dump_classifier
from data_postp.similarity_computations import df_col_to_matrix


PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/validate/metadata_and_hidden_rep_df_08-14-17_16-17-12_test.pickle'
PICKLE_FILE_VALID = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/validate/metadata_and_hidden_rep_df_08-09-17_17-00-24_valid.pickle'



def generate_test_labels_csv(valid_df, test_df, dump_path, n_components=100):
  #train classifier on valid_df

  classifier, pca, _, _ = train_and_dump_classifier(valid_df, class_column="category",
                                                             classifier=sklearn.linear_model.LogisticRegression(n_jobs=-1),
                                                             n_components=n_components, train_split_ratio=0.8)

  test_df = pd.read_pickle(PICKLE_FILE_TEST)

  #PCA transform hidden_reps of test_df
  transformed_vectors_as_matrix = pca.transform(df_col_to_matrix(test_df['hidden_repr']))
  test_df['hidden_repr'] = np.split(transformed_vectors_as_matrix, transformed_vectors_as_matrix.shape[0])
  X_test = df_col_to_matrix(test_df['hidden_repr'])

  #predict labels
  Y = classifier.predict(X_test)

  #generate csv
  result_df = pd.DataFrame(Y, index=test_df['id'])
  result_df.to_csv(dump_path, sep=';')

def calculate_accuracy(valid_df, n_components=[200], n_folds=5,
                       classifier=sklearn.linear_model.LogisticRegression(n_jobs=-1), dump_file_name=None):
  n_components = [n_components] if type(n_components) is not list else n_components
  string_to_dump = ''

  for n_comp in n_components:
    print(type(n_comp))
    acc_array, top5_acc_array = [], []
    print("Training started with %i PCA components and %i folds" %(n_comp, n_folds))
    for i in range(n_folds):
      _, _, acc, top5_acc = train_and_dump_classifier(valid_df, class_column="category", classifier=classifier,
                                                        n_components=n_comp, train_split_ratio=0.8)
      acc_array.append(acc)
      top5_acc_array.append(top5_acc)
    mean_acc, mean_top5_acc = np.mean(acc_array), np.mean(top5_acc_array)

    summary_str = "[%s, %i PCA, %i folds]  acc: %.4f  top5_acc: %.4f"%("LogisticRegression", n_comp, n_folds, mean_acc, mean_top5_acc)
    print(summary_str)
    string_to_dump += string_to_dump + '\n'


    if dump_file_name:
      with open(dump_file_name, 'w') as file:
        file.write(string_to_dump)


def main():
  valid_df, test_df = pd.read_pickle(PICKLE_FILE_VALID), pd.read_pickle(PICKLE_FILE_TEST)
  dump_path = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/validate/test_labels.csv'
  #generate_test_labels_csv(valid_df, test_df, dump_path, n_components=200)

  classifier_analysis_dump_file = '/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/validate/classifier_analysis/classifier_analysis_logistic.txt'
  calculate_accuracy(valid_df, n_components=[100, 150, 200, 250, 300, 400], n_folds=1, dump_file_name=classifier_analysis_dump_file)



if __name__ == "__main__":
  main()