import tensorflow as tf
from tensorflow.python.platform import flags
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from data_postp.similarity_computations import transform_vectors_with_inter_class_pca

FLAGS = tf.python.platform.flags.FLAGS

METADATA_PICKLE_FILE = '/common/homes/students/rothfuss/Documents/selected_trainings/4_actNet_gdl/valid_run/metadata_and_hidden_rep_df_08-07-17_00-21-11_valid.pickle'

flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('training_epochs', 20000, 'training_epochs')
flags.DEFINE_integer('batch_size', 200, 'training_epochs')
flags.DEFINE_string('df_path', METADATA_PICKLE_FILE, 'training_epochs')
flags.DEFINE_string('label_column', 'category', 'name of column in df that contains the labels for the classification')
flags.DEFINE_float('keep_prob', 0.5, 'keep probability dropout')



NONLINEARITY = tf.nn.tanh #tf.nn.elu # tf.nn.relu

''' --- PREPARE DATA --- '''

def prepare_data():
  """
  prepare the data so that X and Y is available as ndarray
  X: ndarray of hidden_repr instances - shape (n_samples, num_dims_hidden_repr)
  Y: ndarray of one-hot encoded labels corresponding to the hidden_reps - - shape (n_samples, num_classes)
  """
  
  df = pd.read_pickle(FLAGS.df_path)
  #df = transform_vectors_with_inter_class_pca(df, class_column=FLAGS.label_column, n_components=300)
  assert 'hidden_repr' in df.columns and FLAGS.label_column in df.columns, "columns for hidden_representation and label must be in df.columns"
  X = np.stack([h.flatten() for h in df['hidden_repr']])
  n_classes = len(set(df[FLAGS.label_column]))
  category_dict = dict([(category, i) for i, category in enumerate(list(set(df['category'])))])
  category_dict_reversed = dict([(i, category) for i, category in enumerate(list(set(df['category'])))])
  Y = tf.one_hot([category_dict[category] for category in df['category']], n_classes)
  Y = tf.Session().run(Y)
  assert X.shape[0] == Y.shape[0] == len(df.index)
  return X, Y

def get_batch(X, Y, batch_size):
  assert X.shape[0] == Y.shape[0]
  r = np.random.randint(X.shape[0], size=batch_size)
  return X[r,:], Y[r,:]
  
X, Y = prepare_data()
#train - test spplit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
n_input, n_classes = X.shape[1], Y.shape[1]

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

n_hidden_1 = 200 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features

# Create model
def multilayer_perceptron(x, weights, biases, keep_prob):
    # Hidden layer with nonlinear activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = NONLINEARITY(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with nonlinear activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = NONLINEARITY(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

# Define Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(FLAGS.training_epochs):
        avg_cost, avg_acc = 0, 0
        total_batch = int(X_train.shape[0]/FLAGS.batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(X_train, Y_train, FLAGS.batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, a = sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: FLAGS.keep_prob})
            # Compute average loss and averavge accuracy
            avg_cost += c / total_batch
            avg_acc += a / total_batch
        # Display logs per epoch step
        if epoch % 100 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "acc=", "{:.9f}".format(avg_acc))
            print("Test Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: Y_test, keep_prob: 1}))
    print("Optimization Finished!")

    # Test model

    # Calculate accuracy
    print("Accuracy:", sess.run([accuracy], feed_dict={x: X_test, y: Y_test, keep_prob: 1}))
