from math import *
import os, collections, scipy, itertools, multiprocessing
from datetime import datetime
from pprint import pprint
from tensorflow.python.platform import flags
import matplotlib as mpl; mpl.use('Agg'); from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sn
import scipy
import json

import sklearn, sklearn.ensemble
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

#from utils.io_handler import store_plot, export_plot_from_pickle, insert_general_classes_to_20bn_dataframe, remove_rows_from_dataframe
import utils.io_handler as io_handler

NUM_CORES = multiprocessing.cpu_count()


#PICKLE_FILE_TRAIN = '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise_20bn_v2/valid_run/metadata_and_hidden_rep_from_train_clean_grouped.pickle'
PICKLE_FILE_TRAIN = '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise_20bn_v2/valid_run/metadata_and_hidden_rep_df_07-13-17_09-47-43_cleaned_grouped.pickle'
PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise_20bn_v2/valid_run/metadata_and_hidden_rep_df_07-13-17_09-47-43.pickle'
#PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise_20bn_v2/valid_run/metadata_and_hidden_rep_from_test_clean_grouped.pickle'
#PICKLE_FILE_TEST = '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise_20bn_v2/valid_run/metadata_and_hidden_rep_from_test_clean.pickle'
PICKLE_DIR_MAIN = '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise_20bn_v2/valid_run/'
MAPPING_DOCUMENT = '/PDFData/rothfuss/data/20bn-something/something-something-grouped.csv'


FLAGS = flags.FLAGS
flags.DEFINE_string('pickle_file_train', PICKLE_FILE_TRAIN, 'path of panda dataframe pickle training file')
flags.DEFINE_string('pickle_file_test', PICKLE_FILE_TEST, 'path of panda dataframe pickle training file')
flags.DEFINE_string('mapping_csv', MAPPING_DOCUMENT, 'path to the csv mapping file which specifies the subclass -> class relation')
flags.DEFINE_string('pickle_dir_main', PICKLE_DIR_MAIN, 'path to the main validation directory of a training')


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
            distance_matrix[row][column] = compute_cosine_similarity(hidden_representations[row],
                                                                     hidden_representations[column])
            class_overlap_matrix[row][column] = labels[row] == labels[column]
    return distance_matrix, class_overlap_matrix


def compute_hidden_representation_similarity(hidden_representations, labels, type=None, video_id_a_string=None,
                                             video_id_b_string=None):
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
    if vector_a.ndim >= 2:
        vector_a = vector_a.flatten()
    if vector_b.ndim >= 2:
        vector_b = vector_b.flatten()

    numerator = sum(a * b for a, b in zip(vector_a, vector_b))
    denominator = square_rooted(vector_a) * square_rooted(vector_b)
    return round(numerator / float(denominator), 3)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def visualize_hidden_representations_tsne(pickle_hidden_representations):
    # shatter 1000x8x8x16 to 1000x1024 dimensions
    labels = list(pickle_hidden_representations['shape'])
    values = df_col_to_matrix(pickle_hidden_representations['hidden_repr'])
    Y_data = np.asarray(labels)

    for i, entry in enumerate(Y_data):
        if entry == 'square':
            Y_data[i] = 's'
        if entry == 'circular':
            Y_data[i] = 'o'
        if entry == 'triangle':
            Y_data[i] = '^'

    model = TSNE(n_components=128, random_state=0, method='exact')
    data = model.fit_transform(values)

    for xp, yp, m in zip(data[:, 0], data[:, 1], Y_data):
        plt.scatter([xp], [yp], marker=m)

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
    # return mean_vector

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
    X = df_col_to_matrix(df_col)  # reshape dataframe column consisting of ndarrays as 2D matrix
    return pca.fit_transform(X)

def distance_classifier(df):
    """Handmade classifier that computes the center of each class and then assigns each datapoint the class with the closest centerpoint
     """
    classes = list(set(df['shape']))
    mean_vectors = [mean_vector(list(df[df['shape'] == c]['hidden_repr'])) for c in classes]
    for i, c in enumerate(classes):
        df['dist_euc_' + str(i)] = (df['hidden_repr'] - mean_vectors[i]).apply(lambda x: np.linalg.norm(x.flatten()))
        df['dist_cos_' + str(i)] = [scipy.spatial.distance.cosine(v.flatten(), mean_vectors[i].flatten()) for v in
                                    df['hidden_repr']]

    df['class'] = [classes.index(s) for s in df['shape']]
    df['eucl_class_pred'] = df[['dist_euc_' + str(i) for i in range(len(classes))]].idxmin(axis=1).apply(
        lambda x: int(x[-1]))
    df['cos_class_pred'] = df[['dist_cos_' + str(i) for i in range(len(classes))]].idxmin(axis=1).apply(
        lambda x: int(x[-1]))

    df['eucl_correct'] = df['eucl_class_pred'] == df['class']
    df['cos_correct'] = df['cos_class_pred'] == df['class']

    print('Accuracy with euclidian distance:', df['eucl_correct'].mean())
    print('Accuracy with cosine distance:', df['cos_correct'].mean())
    return df

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    From scikit-learn.org:
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="lower right")

    text_test_score = 'test score: %.4f' % test_scores_mean[-1]
    text_train_score = 'train score: %.4f' % train_scores_mean[-1]

    ax.text(0.05, 0.05, text_test_score + '\n' + text_train_score, horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    return plt

def svm_fit_and_score(hidden_representations_pickle, class_column="shape", type="linear", CV=False, plot=False):
    """
    The function:
        1. splits the data from the pickle file into 80% train data and 20% test data
        2. does a shuffle-split-cv on the 80% train data for optimizing parameter C of the linear SVM
        3. uses the optimized and trained linear SVM for a final evaluation on the 20% test data

    The C hyperparameter is cross-validated with the shuffle strategy
    (alternative to k-fold), from scikit-learn.org:
        The ShuffleSplit iterator will generate a user defined number of independent train / test dataset splits.
        Samples are first shuffled and then split into a pair of train and test sets.
        ShuffleSplit is thus a good alternative to KFold cross validation that allows both finer control on the number
        of iterations and the proportion of samples on each side of the train / test split.

    explanation of plot output
            Since the training data is randomly split into a training and a test set 10 times (param n_iter=10),
            each point on the training-score curve is the average over 10 scores in which the model was trained and
            evaluated on the first x training samples. Each point on the shuffle-split-cross-validation score curve is the
            average over 10 scores where the model was trained on the first x training samples and evaluated on all
            examples of the test set.


    :param hidden_representations_pickle: the pickle file containing the X and Y data
    :param type: specifies the type of SVM being used. Possible values: "linear" or "rbf" will default to linear SVM.
    :param CV: indicates if CV + Gridsearch for optimizing hyperparams shall be performed, if not default hyperparams are used
    :param plot: indicates if plot for CV is generated (requires CV to be True)
    :return: stores the cross-validation plot with the learning curves and returns the accuracy from the
    evaluation on the 20% test data
    """
    assert type in ["linear", "rbf"],  "No known SVM type specified. Known types are linear or rbf"
    assert not plot or CV

    labels = list(hidden_representations_pickle[class_column])
    values = df_col_to_matrix(hidden_representations_pickle['hidden_repr'])
    Y_data = np.asarray(labels)

    # prepare shuffle split cross-validation, set random_state to a arbitrary constant for recomputability
    X_train, X_test, y_train, y_test = train_test_split(values, Y_data, test_size=0.2, random_state=0)

    estimator = OneVsOneClassifier(SVC(kernel=type, verbose=True), n_jobs=-1) #if type is 'rbf' else LinearSVC(verbose=True)

    if CV:
      cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
      C_values = [0.1, 1, 10, 100, 1000]
      gammas = np.logspace(-6, -1, 10)

      if type is "linear":
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(C=C_values))
        classifier.fit(X_train, y_train)
        estimator = estimator.set_params(C=classifier.best_estimator_.C)
      else: # rbf
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas, C=C_values))
        classifier.fit(X_train, y_train)
        estimator = estimator.set_params(gamma=classifier.best_estimator_.gamma, C=classifier.best_estimator_.C)

      title = 'Learning Curves (SVM (kernel=%s), $\gamma=%.6f$, C=%.1f, %i shuffle splits, %i train samples)' % (type,
        classifier.best_estimator_.gamma, classifier.best_estimator_.C, cv.get_n_splits(), X_train.shape[0] * (1 - cv.test_size))
      print(title)
      file_name = 'svm_linear_%ishuffle_splits_%.2ftest_size' % (cv.n_splits, cv.test_size)


    else:
      classifier = estimator.fit(X_train, y_train)


    #do final test with remaining data and store the
    test_accuracy = classifier.score(X_test, y_test)

    if plot:
      plt = plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
      plt.axes().annotate('test accuracy (on remaining %i samples): %.2f' % (len(X_test), test_accuracy),
                          xy=(0.05, 0.0),
                          xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')

      io_handler.store_plot(FLAGS.pickle_file, file_name)
    return test_accuracy

def svm_train_test_separate(train_df, test_df, class_column="shape", type="linear"):
  assert type in ["linear", "rbf"],  "No known SVM type specified. Known types are linear or rbf"

  X_train = df_col_to_matrix(train_df['hidden_repr'])
  y_train = np.asarray(list(train_df[class_column]))

  X_test = df_col_to_matrix(test_df['hidden_repr'])
  y_test= np.asarray(list(test_df[class_column]))

  if type=='linear':
    estimator = OneVsOneClassifier(sklearn.svm.LinearSVC(verbose=True, max_iter=2000), n_jobs=-1)
  else:
    estimator = OneVsOneClassifier(SVC(kernel=type, verbose=True), n_jobs=-1)  # if type is 'rbf' else LinearSVC(verbose=True)
  classifier = estimator.fit(X_train, y_train)

  test_accuracy = classifier.score(X_test, y_test)
  return test_accuracy

def logistic_regression_fit_and_score(df, class_column="shape"):
    """ Fits a logistic regression model (MaxEnt classifier) on the data and returns the test accuracy"""
    labels = list(df[class_column])
    values = df_col_to_matrix(df['hidden_repr'])
    Y_data = np.asarray(labels)

    # prepare train/test split
    X_train, X_test, y_train, y_test = train_test_split(values, Y_data, test_size=0.2, random_state=0)

    # train logistic regression model
    lr = sklearn.linear_model.LogisticRegression()
    lr = lr.fit(X_train, y_train)

    # after hyperparameter search with cv, do final test with remaining data and store the plot
    test_accuracy = lr.score(X_test, y_test)

    return test_accuracy

def knn_fit_and_score(train_df, test_df, class_column="shape", CV=False, PCA=False, n_pca_components=500):
  # prepare train/test split
  if PCA:
    pca = inter_class_pca(test_df, class_column='category', n_components=n_pca_components)
    X_train = pca.transform(df_col_to_matrix(train_df['hidden_repr']))
    X_test = pca.transform(df_col_to_matrix(test_df['hidden_repr']))
  else:
    X_train = df_col_to_matrix(train_df['hidden_repr'])
    X_test = df_col_to_matrix(test_df['hidden_repr'])

  y_train = np.asarray(list(train_df[class_column]))
  y_test = np.asarray(list(test_df[class_column]))

  estimator = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)

  if CV:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    weights = ['uniform', 'distance']
    n_neighbors = [8, 15, 30, 35] # 10, 12, 14, 16, 18, 20, 25,
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(weights=weights, n_neighbors=n_neighbors))
    print(X_train, y_train)

    classifier.fit(X_train, y_train)
    estimator = estimator.set_params(weights=classifier.best_estimator_.weights, n_neighbors=classifier.best_estimator_.n_neighbors)
    print("best parameters set. weights: %i, n: %i" % classifier.best_estimator_.weights, classifier.best_estimator_.n_neighbors)
  else:
    classifier = estimator.fit(X_train, y_train)

  # do final test with remaining data and store the
  test_accuracy = classifier.score(X_test, y_test)
  print(test_accuracy)

  return test_accuracy


def gradient_boosting_fit_and_score(df, class_column="shape"):
  """ Fits a logistic regression model (MaxEnt classifier) on the data and returns the test accuracy"""
  labels = list(df[class_column])
  values = df_col_to_matrix(df['hidden_repr'])
  Y_data = np.asarray(labels)

  # prepare train/test split
  X_train, X_test, y_train, y_test = train_test_split(values, Y_data, test_size=0.2, random_state=0)

  # train logistic regression model
  gb = sklearn.ensemble.GradientBoostingClassifier(verbose=True)
  gb = gb.fit(X_train, y_train)

  # after hyperparameter search with cv, do final test with remaining data and store the plot
  test_accuracy = gb.score(X_test, y_test)

  return test_accuracy


def export_random_forest_plot(df, class_column="shape", plot=False):
    X = df_col_to_matrix(df['hidden_repr'])
    Y = df[class_column]

    # prepare shuffle split cross-validation, set random_state to a arbitrary constant for recomputability
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    n_estimators = [10, 30, 50, 80, 100]
    max_depth = [2, 5, 10, 15, 20, 25, 30, 35]

    rf = sklearn.ensemble.RandomForestClassifier()

    classifier = GridSearchCV(estimator=rf, cv=cv, param_grid=dict(n_estimators=n_estimators, max_depth=max_depth))
    # run fit with all hyperparameter values
    classifier.fit(X_train, y_train)
    rf = rf.set_params(n_estimators=classifier.best_estimator_.n_estimators,
                       max_depth=classifier.best_estimator_.max_depth)


    title = 'Learning Curves (Random Forest, n_trees=%i, depth=%i, %i shuffle splits, %i train samples)' %  (
        classifier.best_estimator_.n_estimators, classifier.best_estimator_.max_depth, cv.get_n_splits(),
        X_train.shape[0] * (1 - cv.test_size)) if plot else 0

    # train random forest
    plt = plot_learning_curve(rf, title, X, Y, cv=cv)

    file_name = 'random_forest_%ishuffle_splits_%.2ftest_size' % (cv.n_splits, cv.test_size)

    test_accuracy = classifier.score(X_test, y_test)

    if plot:
        plt.axes().annotate('test accuracy (on remaining %i samples): %.2f' % (len(X_test), test_accuracy), xy=(0.05, 0.0),
                        xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')

        io_handler.store_plot(FLAGS.pickle_file, file_name)

    return test_accuracy


def export_decision_tree_plot(df, class_column="shape", plot=False):
    X = df_col_to_matrix(df['hidden_repr'])
    Y = df[class_column]

    # prepare shuffle split cross-validation, set random_state to a arbitrary constant for recomputability
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    dt = sklearn.tree.DecisionTreeClassifier()

    classifier = GridSearchCV(estimator=dt, cv=cv)
    # run fit with all hyperparameter values
    classifier.fit(X_train, y_train)


    title = 'Learning Curves (Decision Tree, %i shuffle splits, %i samples)' % (
        cv.get_n_splits(), len(X_train)) if plot else 0

    # train decision tree
    plt = plot_learning_curve(dt, title, X_train, y_train, cv=cv)

    test_accuracy = classifier.score(X_test, y_test)

    if plot:
        file_name = 'decision_tree_%ishuffle_splits_%.2ftest_size' % (cv.n_splits, cv.test_size)

        io_handler.store_plot(FLAGS.pickle_file, file_name)

    return test_accuracy


def avg_distance(df, similarity_type='cos'):
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
                distance = np.sqrt(np.sum((v1.flatten() - v2.flatten()) ** 2))
            if l1 == l2:
                same_class_dist_array.append(distance)
            else:
                out_class_dist_array.append(distance)
    return np.mean(same_class_dist_array), np.mean(out_class_dist_array)


def similarity_matrix(df, df_label_col, similarity_type='cos', vector_type = "", file_name="sim_matrix_font_large_", plot_options=((64, 64), 15, 15)):
  """
  Computes a matrix the mean pairwise cosine similarities of the hidden vectors belongining to different classes

  :param df: pandas dataframe that contains the hidden vectors in a column called 'hidden_repr' and the corresponding class labels
  :param df_label_col: name of the column with the class labels
  :param similarity_type: denotes the similarity metric that is used (so far only 'cos' is supported)
  :param vector_type: optional string that is included in the name of the dumped files
  :param plot_options: list of settings for matplotlib and seaborn. First list element specifies figure size as a
  tuple e.g. 64x64. Second list element specifies font_scale for seaborn as a single integer, e.g. 15. Third list
  element specifies annotation font size as a single integer, e.g. 15)
  :return: inter class cosine similarity matrix (ndarray with shape (n_classes, n_classes))
  """
  assert 'hidden_repr' in list(df) and df_label_col in list(df)
  assert similarity_type in ['cos'] #euclidean no longer supported
  labels = list(sorted(set(df[df_label_col])))
  n = len(labels)
  sim_matrix = np.zeros([n, n])
  for i, j in itertools.product(range(n), range(n)):
    if i <= j: # since similarity matrix is mirrored along the diagonal, only compute one half
      print(i, j, labels[i], labels[j])
      vectors1 = np.asmatrix([v.flatten() for v in list(df[df[df_label_col] == labels[i]]['hidden_repr'])])
      vectors2 = np.asmatrix([v.flatten() for v in list(df[df[df_label_col] == labels[j]]['hidden_repr'])])
      kernel_matrix = sklearn.metrics.pairwise.cosine_similarity(vectors1, vectors2)
      sim_matrix[i, j] = np.mean(kernel_matrix)

  #mirror the second half
  for i, j in itertools.product(range(n), range(n)):
    if i > j:
      sim_matrix[i, j] = sim_matrix[j, i]

  df_cm = pd.DataFrame(sim_matrix, index=labels, columns=labels)
  #print(df_cm)
  df_cm.to_pickle(os.path.join(os.path.dirname(FLAGS.pickle_file), 'sim_matrix_' + similarity_type + '_' + vector_type + '.pickle'))
  plt.figure(figsize=plot_options[0])
  sn.set(font_scale=plot_options[1])
  ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": plot_options[2]})
  if n > 5:
      # rotate x-axis labels
      for item in ax.get_xticklabels():
          item.set_rotation(90)
  heatmap_file_name = os.path.join(os.path.dirname(FLAGS.pickle_file),
                                   file_name + df_label_col + '_' + similarity_type + '_' + vector_type + '.png')
  plt.savefig(heatmap_file_name, dpi=100)
  print('Dumped Heatmap to:', heatmap_file_name)
  plt.show()
  return sim_matrix


def plot_similarity_shape_motion_matrix(df):
    """This function computes the large square confusion matrix with the dimensions being of shape (num_shapes * num_directions) and plots it using matplotlib"""

    different_shapes = df['shape'].unique()
    different_directions = df['motion_location'].unique()
    large_confusion_matrix = np.zeros(
        shape=(len(different_shapes) * len(different_directions), len(different_shapes) * len(different_directions)))

    x = 0
    y = 0

    for (shape1, shape2) in list(itertools.product(different_shapes, different_shapes)):
        print('x:', x)
        print('y:', y)
        if (shape1 == 'circular' and shape2 == 'circular'): x = 0; y = 0
        if (shape1 == 'circular' and shape2 == 'triangle'): x = 0; y = 8
        if (shape1 == 'circular' and shape2 == 'square'): x = 0; y = 16
        if (shape1 == 'triangle' and shape2 == 'circular'): x = 8; y = 0
        if (shape1 == 'triangle' and shape2 == 'triangle'): x = 8; y = 8
        if (shape1 == 'triangle' and shape2 == 'square'): x = 8; y = 16
        if (shape1 == 'square' and shape2 == 'circular'): x = 16; y = 0
        if (shape1 == 'square' and shape2 == 'triangle'): x = 16; y = 8
        if (shape1 == 'square' and shape2 == 'square'): x = 16; y = 16
        print('computing matrix for: ' + shape1, shape2)
        intermediate_matrix = compute_small_confusion_matrix(df, shape1, shape2)
        print('inserting small into large confusion matrix')
        large_confusion_matrix[x:x + intermediate_matrix.shape[0],
        y:y + intermediate_matrix.shape[1]] = intermediate_matrix

    # print(large_confusion_matrix)

    df_cm = pd.DataFrame(large_confusion_matrix, index=[i for i in np.concatenate(
        [different_directions, different_directions, different_directions])], columns=[i for i in np.concatenate(
        [different_directions, different_directions, different_directions])])
    plt.figure(figsize=(64, 64))
    sn.heatmap(df_cm, annot=True)
    heatmap_file_name = os.path.join(os.path.dirname(FLAGS.pickle_file), 'large_sim_matrix.png')
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
    for (shape1, shape2, direction1, direction2) in list(
            itertools.product([shape1], [shape2], different_directions, different_directions)):
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
            confusion_matrix[i, j] = similarity_lists_means.get(
                str(shape1) + '_' + str(shape2) + '_' + str(direction1) + '_' + str(direction2))

    return confusion_matrix


def classifier_analysis(df, class_column='shape'):
    svm_linear_accuracy = svm_fit_and_score(df, class_column=class_column, type='linear')
    svm_rbf_accuracy = svm_fit_and_score(df, class_column=class_column, type='rbf')
    lr_accuracy = logistic_regression_fit_and_score(df, class_column=class_column)
    string_to_dump = str(datetime.now()) + '\n' \
                     + '---- SVM Linear ----\n Accuracy: ' + str(svm_linear_accuracy) + '\n' \
                     + '---- SVM RBF ----\n Accuracy: ' + str(svm_rbf_accuracy) + '\n' \
                     + '---- LogisticRegression ----' + '\n' + 'Accuracy: ' + str(lr_accuracy) + '\n'
    dump_file_name = os.path.join(os.path.dirname(FLAGS.pickle_file), 'classifier_analysis' + '.txt')
    print(string_to_dump)
    with open(dump_file_name, 'w') as file:
        file.write(string_to_dump)


def classifier_analysis_train_test(train_df_path, test_df_path=None, class_column='category'):
  train_df = pd.read_pickle(train_df_path)
  split = False
  # pca fit on all data
  full_data = train_df
  if not test_df_path:
    #create own test split
    msk = np.random.rand(len(train_df)) < 0.8
    test_df = train_df[~msk]
    train_df = train_df[msk]
    split = True
  else:
    test_df = pd.read_pickle(test_df_path)

  #prepare dump file
  dump_file_name = os.path.join(os.path.dirname(train_df_path), 'full_classifier_analysis_' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.txt')

  valid_procedure_spec = {'SVM_Linear': [-1, 200, 500],
                          #'SVM_RBF': [-1, 200, 500],
                         'Logistic_Regression': ([-1, 200, 500]),
                          'KNN': tuple(itertools.product([5, 10, 20, 50], [3, 5, 10, 20, 50]))}

  classifier_dict = {'SVM_Linear': OneVsOneClassifier(sklearn.svm.LinearSVC(verbose=True, max_iter=2000), n_jobs=-1),
                     #'SVM_RBF': OneVsRestClassifier(SVC(kernel='rbf', verbose=True), n_jobs=-1),
                     'Logistic_Regression': sklearn.linear_model.LogisticRegression(n_jobs=-1),
                     'KNN': sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
  }

  configurations = [(classifier_name, param) for classifier_name, params in valid_procedure_spec.items() for param in params]

  string_summary = "--Summary--\nused classifiers: " + str([key for key in valid_procedure_spec.keys()]) + "\nlabel(column) used: " + class_column +\
                    "\ntrain_df used: " + str(train_df_path) + "\ntest_df used: " + str(test_df_path)
  string_summary += "\n" if not split else "\nwith 80/20 split of train_df (100/0 for pca)\n"
  print(string_summary)

  #actually run that shit
  for classifier_name, param in configurations:
    append_write = 'a' if os.path.exists(dump_file_name) else 'w'
    string_to_dump = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' -- ' + classifier_name + ': ' + str(
      param) + ' --> Start Training' + '\n'
    with open(dump_file_name, append_write) as f:
      f.write(string_to_dump)
    print(string_to_dump)

    n_components = param[0] if classifier_name is 'KNN' else param

    if n_components > 0:
      pca = inter_class_pca(full_data, class_column=class_column, n_components=n_components)
      X_train = pca.transform(df_col_to_matrix(train_df['hidden_repr']))
      X_test = pca.transform(df_col_to_matrix(test_df['hidden_repr']))
    else:
      X_train = df_col_to_matrix(train_df['hidden_repr'])
      X_test = df_col_to_matrix(test_df['hidden_repr'])

    y_train = np.asarray(list(train_df[class_column]))
    y_test = np.asarray(list(test_df[class_column]))

    # choose and parametrize classifier model
    estimator = classifier_dict[classifier_name]
    if classifier_name is 'KNN':
      estimator.set_params(n_neighbors=param[1])

    # fit model and calculate accuracy:
    estimator.fit(X_train, y_train)
    acc = estimator.score(X_test, y_test)
    
    #can only perform top_n_acc with classifiers that can be interpreted probabilitically
    if classifier_name is 'Logistic_Regression':
      top_n_acc = top_n_accuracy(estimator, X_test, y_test)
    else:
      top_n_acc = '-'


    string_to_dump = string_summary + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' -- ' + classifier_name + ': ' + str(
      param) + ' --> Training Done' + '\n' + \
                     'Accuracy: ' + str(acc) + '\n' + \
                     'Top-5-Accuracy: ' + str(top_n_acc) + '\n' + \
                     'label used: ' + class_column + '\n'
    with open(dump_file_name, append_write) as f:
      f.write(string_to_dump)
    print(string_to_dump)
    string_summary = ""


def classifier_analysis_train_test_different_splits(train_df_path, class_column='category'):
  train_df = pd.read_pickle(train_df_path)
  # pca fit on all data
  full_data = train_df

  accuracies_dict = {'SVM_Linear': {},
                     'Logistic_Regression': {},
                     'KNN': {}
                     }

  valid_procedure_spec = {'SVM_Linear': [-1, 200, 500],
                          #'SVM_RBF': [-1, 200, 500],
                         'Logistic_Regression': ([-1, 200, 500]),
                          'KNN': tuple(itertools.product([5, 10, 20, 50], [3, 5, 10, 20, 50]))}

  classifier_dict = {'SVM_Linear': OneVsOneClassifier(sklearn.svm.LinearSVC(verbose=True, max_iter=2000), n_jobs=-1),
                     # 'SVM_RBF': OneVsRestClassifier(SVC(kernel='rbf', verbose=True), n_jobs=-1),
                     'Logistic_Regression': sklearn.linear_model.LogisticRegression(n_jobs=-1),
                     'KNN': sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
                     }

  configurations = [(classifier_name, param) for classifier_name, params in valid_procedure_spec.items() for param in
                    params]


  # prepare dump file
  dump_file_name = os.path.join(os.path.dirname(train_df_path), 'full_classifier_analysis_' + str(
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.txt')

  for ratio in np.arange(0.4, 1.0, 0.1):
    append_write = 'a' if os.path.exists(dump_file_name) else 'w'
    # create own test split
    msk = np.random.rand(len(train_df)) < ratio
    test_df = train_df[~msk]
    train_df = train_df[msk]

    string_summary = "\nlabel(column) used: " + class_column + "\ntrain_df used: " + str(train_df_path) +\
                      "\nSplit(train percentage): " + str(ratio) + "\n"

    print(string_summary)

    # actually run that shit
    for classifier_name, param in configurations:
      # assign keys and values to accuracies dict
      print(param)
      accuracies_dict[classifier_name][(param)] = {'top_n_acc': list(), 'acc': list()}

      # knn only with pca due to curse of dimensionality resulting in tuple: (#pca components, #neighbors)
      n_components = param[0] if classifier_name is 'KNN' else param
      if n_components > 0:
        pca = inter_class_pca(full_data, class_column=class_column, n_components=n_components)
        X_train = pca.transform(df_col_to_matrix(train_df['hidden_repr']))
        X_test = pca.transform(df_col_to_matrix(test_df['hidden_repr']))

      else:
        X_train = df_col_to_matrix(train_df['hidden_repr'])
        X_test = df_col_to_matrix(test_df['hidden_repr'])

      y_train = np.asarray(list(train_df[class_column]))
      y_test = np.asarray(list(test_df[class_column]))

      # choose and parametrize classifier model
      estimator = classifier_dict[classifier_name]
      if classifier_name is 'KNN':
        estimator.set_params(n_neighbors=param[1])

      # fit model and calculate accuracy:
      estimator.fit(X_train, y_train)
      acc = estimator.score(X_test, y_test)

      # can only perform top_n_acc with classifiers that can be interpreted probabilitically
      if classifier_name is 'Logistic_Regression':
        top_n_acc = top_n_accuracy(estimator, X_test, y_test)
      else:
        top_n_acc = '-'

      accuracies_dict[classifier_name][param]['top_n_acc'].append(top_n_acc if type(top_n_acc) is str else '%.3f' % float(top_n_acc))
      accuracies_dict[classifier_name][param]['acc'].append('%.3f' % float(acc))


    string_to_dump = string_summary + str(accuracies_dict)
    print("cleared dict")


    with open(dump_file_name, append_write) as f:
      f.write(string_to_dump)
    print(string_to_dump)


def mean_vectors_of_classes(df, class_column='shape'):
  """
  Computes mean vector for each class in class_column
  :param df: dataframe containing hidden vectors + metadata
  :param class_column: column_name corresponding to class labels
  :return: dataframe with classes as index and mean vectors for each class
  """
  assert 'hidden_repr' in df.columns and class_column in df.columns
  labels = list(df[class_column])
  values = df_col_to_matrix(df['hidden_repr'])
  vector_dict = collections.defaultdict(list)
  for label, vector in zip(labels, values):
    vector_dict[label].append(vector)

  return pd.DataFrame.from_dict(dict([(label, np.mean(vectors, axis=0)) for label, vectors in vector_dict.items()]), orient='index')


def inter_class_variance_plot(df, class_column='shape'):
  """
  Plots inter-class variance of mean vectors for each dimension of the hidden representation
  :param df: dataframe containing hidden vectors + metadata
  :param class_column: column_name corresponding to class labels
  """
  assert 'hidden_repr' in df.columns and class_column in df.columns
  variance_df = mean_vectors_of_classes(df, class_column=class_column).var(axis=0)
  plt.figure()
  variance_df.plot(kind='bar')
  plt.show()


def inter_class_pca(df, class_column='shape', n_components=50):
  """
  Performs PCA on mean vactors of classes
  :param df: dataframe containing hidden vectors + metadata
  :param class_column: column_name corresponding to class labels
  :param n_components: number of principal components
  :return: fitted pca sklean object
  """
  assert 'hidden_repr' in df.columns and class_column in df.columns
  mean_vector_df = mean_vectors_of_classes(df, class_column=class_column)
  pca = sklearn.decomposition.PCA(n_components).fit(mean_vector_df)
  relative_variance_explained = np.sum(pca.explained_variance_)/np.sum(mean_vector_df.var(axis=0))
  print("PCA (n_components= %i: relative variance explained:" % n_components, relative_variance_explained, '\n',pca.explained_variance_)
  variance_df = mean_vectors_of_classes(df, class_column=class_column).var(axis=0)
  return pca


def transform_vectors_with_inter_class_pca(df, class_column='shape', n_components=50):
  """
    Performs PCA on mean vactors of classes and applies transformation to all hidden_reps in df
    :param df: dataframe containing hidden vectors + metadata
    :param class_column: column_name corresponding to class labels
    :param n_components: number of principal components
    :return: dataframe with transformed vectors
    """
  assert 'hidden_repr' in df.columns and class_column in df.columns
  df = df.copy()
  pca = inter_class_pca(df, class_column=class_column, n_components=n_components)
  transformed_vectors_as_matrix = pca.transform(df_col_to_matrix(df['hidden_repr']))
  df['hidden_repr'] = np.split(transformed_vectors_as_matrix, transformed_vectors_as_matrix.shape[0])
  return df


def find_closest_vectors(df, query_idx, class_column='shape', n_closest_matches=5):
    """
    finds the closest vector matches (cos_similarity) for a given query vector
    :param df: dataframe containing hidden vectors + metadata
    :param query_idx: index of the query vector in the df
    :param num_vectors: denotes how many of the closest vector matches shall be returned
    :return: list of tuples (i, cos_sim, label) correspondig to the num_vectors closest vector matches
    """
    assert 'hidden_repr' in df.columns
    assert query_idx in df.index
    query_row = df.iloc[query_idx]
    query_class = query_row[class_column]
    query_v_id = query_row["video_id"]
    remaining_df = df[df.index != query_idx]
    cos_distances = [(compute_cosine_similarity(v, query_row['hidden_repr']), l, int(v_id)) for _, v, l, v_id in
                     zip(remaining_df.index, remaining_df['hidden_repr'], remaining_df[class_column], remaining_df['video_id'])]
    sorted_distances = sorted(cos_distances, key=lambda tup: tup[0], reverse=True)
    sorted_distances = [tup for tup in sorted_distances if tup[2] != query_v_id]
    print([l for _, l, _ in sorted_distances[:n_closest_matches]].count(query_class), query_v_id)
    return sorted_distances[:n_closest_matches]


def closest_vector_analysis(df, class_column='shape', n_closest_matches=5):
    #construct pairwise similarity matrix
    assert 'hidden_repr' in df.columns and class_column in df.columns

    for i in df.index:
        label = df.iloc[i][class_column]
        closest_vectors = find_closest_vectors(df, i, class_column=class_column, n_closest_matches=n_closest_matches)
        print(label)
        pprint(closest_vectors)

    #sim = Parallel(n_jobs=NUM_CORES)(
    #    delayed(compute_cosine_similarity)(v1, v2) for v1, v2 in itertools.product(vectors1, vectors2))


def dnq_metric(sim_matr):
  sim_matr_rad = np.arccos(sim_matr.as_matrix())
  diagonal = np.diagonal(sim_matr_rad)
  diag_mean = np.mean(diagonal)
  np.fill_diagonal(sim_matr_rad, np.zeros(sim_matr_rad.shape[0]))
  non_diag_mean = np.sum(sim_matr_rad)/float(sim_matr_rad.shape[0]*(sim_matr_rad.shape[0]-1))
  return (1-diag_mean/non_diag_mean)


def general_result_analysis(df, class_column="category"):
  similarity_matrix(df, class_column, vector_type='no_pca', file_name="sim_matrix_font_large_cleaned_grouped_", plot_options=((100, 100), 5, 10))

  transformed_df = transform_vectors_with_inter_class_pca(df, class_column, n_components=200)
  similarity_matrix(transformed_df, class_column, vector_type='pca', file_name="sim_matrix_font_large_cleaned_grouped_", plot_options=((100, 100), 5, 10))

  #classifier_analysis(df, class_column)


def lr_analysis_train_test_separate(train_df, test_df, class_column="category", PCA=False, n_pca_components=500):

    if PCA:
      pca = inter_class_pca(test_df, class_column='category', n_components=n_pca_components)
      X_train = pca.transform(df_col_to_matrix(train_df['hidden_repr']))
      X_test = pca.transform(df_col_to_matrix(test_df['hidden_repr']))
    else:
      X_train = df_col_to_matrix(train_df['hidden_repr'])
      X_test = df_col_to_matrix(test_df['hidden_repr'])

    y_train = np.asarray(list(train_df[class_column]))
    y_test = np.asarray(list(test_df[class_column]))

    # train logistic regression modelpredict_proba
    lr = sklearn.linear_model.LogisticRegression()
    lr = lr.fit(X_train, y_train)

    # after hyperparameter search with cv, do final test with remaining data and store the plot
    print('Top 5 Accuracy:', top_n_accuracy(lr, X_test, y_test))
    print('Accuracy', lr.score(X_test, y_test))


def top_n_accuracy(trained_classifier, X_test, y_test, n=5):
    log_prob_matrix = trained_classifier.predict_proba(X_test)
    classes_dict = dict([(label, i) for i, label in enumerate(trained_classifier.classes_)])
    in_top_n_array = []
    for i in range(log_prob_matrix.shape[0]):
      label_index = classes_dict[y_test[i]]
      top_n_label_indices = np.argsort(log_prob_matrix[i,:])[-n:]
      in_top_n_array.append(label_index in top_n_label_indices)

    top_n_acc = np.mean(in_top_n_array)
    return top_n_acc


def io_calls():
  #df = pd.read_pickle(FLAGS.pickle_file_train)
  #dataframe_cleaned = io_handler.replace_char_from_dataframe(FLAGS.pickle_file_train, "category", "”", "")
  #dataframe_cleaned = io_handler.remove_rows_from_dataframe(FLAGS.pickle_file_test, "test")
  #filename = 'metadata_and_hidden_rep_df_07-13-17_09-47-43_cleaned' + '.pickle'
  #io_handler.store_dataframe(dataframe_cleaned, FLAGS.pickle_dir_main, filename)

  #dataframe_grouped = io_handler.insert_general_classes_to_20bn_dataframe(FLAGS.pickle_file_train, FLAGS.mapping_csv)
  #filename = 'metadata_and_hidden_rep_df_07-13-17_09-47-43_cleaned_grouped' + '.pickle'
  #io_handler.store_dataframe(dataframe_grouped, FLAGS.pickle_dir_main, filename)
  return None


def main():
    #df = pd.read_pickle(FLAGS.pickle_file_train)

    #general_result_analysis(df, class_column="class")

    #transformed_df = transform_vectors_with_inter_class_pca(df, class_column="category", n_components=600)
    #classifier_analysis(transformed_df, "category")

    #transformed_df = transform_vectors_with_inter_class_pca(df, class_column="category", n_components=20)
    #closest_vector_analysis(transformed_df, class_column="category")

    #test_df = pd.read_pickle(FLAGS.pickle_file_test)
    #train_df = pd.read_pickle(FLAGS.pickle_file_train)

    #knn_fit_and_score(train_df, test_df, class_column="category", CV=True)

    #lr_analysis_train_test_separate(train_df, test_df, class_column="category")

    #io_calls()

    #classifier_analysis_train_test(FLAGS.pickle_file_train, class_column="category")
    classifier_analysis_train_test_different_splits(FLAGS.pickle_file_train, class_column="class")

    #sim_matr_no_pca = pd.read_pickle('/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise/valid_run/sim_matrix_cos_no_pca.pickle')
    #sim_matr_pca = pd.read_pickle(
    #  '/common/homes/students/rothfuss/Documents/training/06-09-17_16-10_1000fc_noise/valid_run/sim_matrix_cos_pca.pickle')
if __name__ == "__main__":
    main()


