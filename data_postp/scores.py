import numpy as np
import pandas as pd
import sklearn
from data_postp import similarity_computations
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def get_y_true_y_pred(df, true_class_column="true_class", column_keyword="pred_class"):
  assert df is not None
  df_true_classes = df.filter(like=true_class_column)
  df_pred_classes = df.filter(like=column_keyword)
  all_classes = np.unique(pd.concat([df_true_classes, df_pred_classes], axis=1))

  matches = df_pred_classes.isin(df.true_class)

  df_y_true = pd.DataFrame(np.full((len(df.index), len(all_classes)), False), columns=all_classes)
  for i, row in matches.iterrows():
    if row.any():
      """ get the true label, determine location and set to True """
      label = df_true_classes.iloc[i].values
      df_y_true.set_value(i, label, True)


  df_y_pred = pd.DataFrame(np.full((len(df.index), len(all_classes)), False), columns=all_classes)
  for i, row in df_pred_classes.iterrows():
    for entry in row.values:
      df_y_pred.set_value(i, entry, True)

  return df_y_true.as_matrix(), df_y_pred.as_matrix(), all_classes


def compute_average_precision_score(y_true, y_pred, labels):
  assert len(y_true) == len(y_pred)
  """ the array returned contains precisions that are estimated in binary mode, meaining:
  if the true class is also marked true somewhere among the 5 matches, precision is 1/5"""
  y_score = [sklearn.metrics.precision_score(y_true[i], y_pred[i], labels=list(labels), average=None) for i in
       range(len(y_true))]

  """ generate random true/false matrix"""
  a = np.random.rand(np.shape(y_true)[0], np.shape(y_true)[1]) > 0.5
  print(sklearn.metrics.average_precision_score(a, y_pred)) # ~0.5

  return sklearn.metrics.average_precision_score(y_true, y_score)


def get_query_matching_table(df, df_query, class_column='shape', n_closest_matches=5, df_query_label="category", df_query_id="id"):
  """
  Yields a pandas dataframe in which each row contains the n_closest_matches as a result from querying the df for every single
  hidden representation in the df dataframe. In addition, every row contains the true label and the query id.
  :param df: the df to be queried
  :param df_query: the df from which to query
  :return: pandas dataframe with columns ("id", "true_label", "match i pred_class" for i=1,...,n_closest_matches) and
  number of rows equal to df_query rows
  """
  assert df is not None and df_query is not None
  assert 'hidden_repr' in df.columns and class_column in df.columns
  assert 'hidden_repr' in df_query.columns and df_query_label in df_query.columns and df_query_id in df_query.columns

  columns = [["id{}".format(i), "pred_class{}".format(i)] for i in range(1, n_closest_matches + 1)]
  columns = [e for entry in columns for e in entry] # flatten list in list
  columns[:0] = ["id", "true_class"]

  query_matching_df = pd.DataFrame(columns=columns)
  query_matching_df.set_index('id', 'true_class')

  for hidden_repr, label, id in zip(df_query['hidden_repr'], df_query[df_query_label], df_query[df_query_id]):
    closest_vectors = similarity_computations.find_closest_vectors(df, hidden_repr=hidden_repr, class_column=class_column,
                                           n_closest_matches=n_closest_matches)


    matching_results = [[tpl[2], tpl[1]] for tpl in closest_vectors]
    matching_results = sum(matching_results, [])  #flatten
    matching_results[:0] = [id, label]

    row_data = dict(zip(columns, matching_results))
    query_matching_df = query_matching_df.append(row_data, ignore_index=True)

  return query_matching_df


def main():
  df = pd.read_pickle(FLAGS.pickle_file_test)
  df_val = pd.read_pickle('/common/homes/students/rothfuss/Documents/selected_trainings/8_20bn_gdl_optical_flow/valid_run/metadata_and_hidden_rep_df_08-12-17_02-50-16_selected_10_classes_eren_augmented_valid.pickle')

  df, df_val = similarity_computations.transform_vectors_with_inter_class_pca(df, df_val, class_column='category', n_components=50)
  df = get_query_matching_table(df[:500], df_val[:500], class_column="category")
  y_true, y_pred, labels = get_y_true_y_pred(df)
  aps = compute_average_precision_score(y_true, y_pred, labels)
  print("the mean average precision score is: ", aps)


if __name__ == "__main__":
  main()
