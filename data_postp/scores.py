import numpy as np
import pandas as pd
import sklearn
from data_postp import similarity_computations
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def compute_mean_average_precision(df, match_matrix=False, column_keyword="pred_class"):
  assert df is not None
  """
  either provide a pandas dataframe with shape (n, m) with n being the number of queries, m being the number of columns
  or provide a matrix with true/false values with the same shape as above representing the matching between true class labels
  and predicted class labels.
  """
  if not match_matrix:
    df_pred_classes = df.filter(like=column_keyword)
    n_relevant_documents = len(df_pred_classes.columns)
    matches = df_pred_classes.isin(df.true_class).as_matrix()
  else:
    n_relevant_documents = np.shape(df)[1]
    matches = df

  P = np.zeros(shape=matches.shape)
  for k in range(n_relevant_documents):
    P[:, k] = np.mean(matches[:, :k], axis=1)

  return np.mean(np.multiply(P, matches))


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
  valid_file="/common/homes/students/ferreira/Documents/metadata_and_hidden_rep_df_08-09-17_17-00-24_valid.pickle"
  df = pd.read_pickle(valid_file)

  # create own train/test split
  msk = np.random.rand(len(df)) < 0.8
  test_df = df[~msk]
  print("number of test samples: ", np.shape(test_df)[0])
  train_df = df[msk]
  print("number of train samples: ", np.shape(train_df)[0])

  df, df_val = similarity_computations.transform_vectors_with_inter_class_pca(train_df, test_df, class_column='category', n_components=50)
  df = get_query_matching_table(df[:100], df_val[:100], class_column="category")
  print(compute_mean_average_precision(df))



if __name__ == "__main__":
  main()
