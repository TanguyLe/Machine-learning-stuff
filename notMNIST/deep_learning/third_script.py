# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

from paths import notMNIST_pickle_file
from constants import image_size, letters

with open(notMNIST_pickle_file, 'rb') as f:
    save = pickle.load(f)
    training_dataset = save['training_dataset']
    training_labels = save['training_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    # print('Training set', training_dataset.shape, training_labels.shape)
    # print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)

    num_labels = len(letters)

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels


    training_dataset, training_labels = reformat(training_dataset, training_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    # print('Training set', training_dataset.shape, training_labels.shape)
    # print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
num_nodes = 1024
batch_size = 512

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_training_dataset = tf.constant(training_dataset)
    tf_training_labels = tf.constant(training_labels)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]), name="weights1")
    biases1 = tf.Variable(tf.zeros([num_nodes]), name="biases1")
    weights2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]), name="weights2")
    biases2 = tf.Variable(tf.zeros([num_labels]), name="biases2")

    # Training computation.
    def get_logits(tf_dataset):
        first_logits = tf.matmul(tf_dataset, weights1) + biases1
        relu_layer1 = tf.nn.relu(first_logits)
        second_logits = tf.matmul(relu_layer1, weights2) + biases2
        return second_logits


    logits = get_logits(tf_training_dataset)

    # Predictions for the training, validation, and test data.
    training_prediction = tf.nn.softmax(logits)

    valid_logits = get_logits(tf_valid_dataset)
    valid_prediction = tf.nn.softmax(valid_logits)

    test_logits = get_logits(tf_test_dataset)
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 6001


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver({"weights1": weights1, "biases1": biases1, "weights2": weights2, "biases2": biases2})
    saver.restore(session, "./tmp/model-2-12000")

    predictions = session.run([training_prediction])

    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
