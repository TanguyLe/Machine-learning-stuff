from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
from six.moves import cPickle as pickle
from paths import notMNIST_pickle_file

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

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


training_dataset, training_labels = reformat(training_dataset, training_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

num_nodes_1 = 2048
num_nodes_2 = int(num_nodes_1 * 0.5)
num_nodes_3 = int(num_nodes_1 * np.power(0.5, 2))
num_nodes_4 = int(num_nodes_1 * np.power(0.5, 3))
num_nodes_5 = int(num_nodes_1 * np.power(0.5, 4))
num_nodes_6 = int(num_nodes_1 * np.power(0.5, 5))
num_nodes_7 = int(num_nodes_1 * np.power(0.5, 6))
num_nodes_8 = int(num_nodes_1 * np.power(0.5, 7))
batch_size = 128
beta = 0.001

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_training_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_training_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_nodes_1], stddev=math.sqrt(2.0 / (image_size * image_size))))
    biases1 = tf.Variable(tf.zeros([num_nodes_1]))
    weights2 = tf.Variable(tf.truncated_normal([num_nodes_1, num_nodes_2], stddev=math.sqrt(2.0 / num_nodes_1)))
    biases2 = tf.Variable(tf.zeros([num_nodes_2]))
    weights3 = tf.Variable(tf.truncated_normal([num_nodes_2, num_nodes_3], stddev=math.sqrt(2.0 / num_nodes_2)))
    biases3 = tf.Variable(tf.zeros([num_nodes_3]))
    weights4 = tf.Variable(tf.truncated_normal([num_nodes_3, num_nodes_4], stddev=math.sqrt(2.0 / num_nodes_3)))
    biases4 = tf.Variable(tf.zeros([num_nodes_4]))
    weights5 = tf.Variable(tf.truncated_normal([num_nodes_4, num_nodes_5], stddev=math.sqrt(2.0 / num_nodes_4)))
    biases5 = tf.Variable(tf.zeros([num_nodes_5]))
    weights6 = tf.Variable(tf.truncated_normal([num_nodes_5, num_nodes_6], stddev=math.sqrt(2.0 / num_nodes_5)))
    biases6 = tf.Variable(tf.zeros([num_nodes_6]))
    weights7 = tf.Variable(tf.truncated_normal([num_nodes_6, num_nodes_7], stddev=math.sqrt(2.0 / num_nodes_6)))
    biases7 = tf.Variable(tf.zeros([num_nodes_7]))
    weights8 = tf.Variable(tf.truncated_normal([num_nodes_7, num_nodes_8], stddev=math.sqrt(2.0 / num_nodes_7)))
    biases8 = tf.Variable(tf.zeros([num_nodes_8]))
    weights9 = tf.Variable(tf.truncated_normal([num_nodes_8, num_labels], stddev=math.sqrt(2.0 / num_nodes_8)))
    biases9 = tf.Variable(tf.zeros([num_labels]))

    global_step = tf.Variable(0)

    # Training computation.
    def get_dropout_result(tf_dataset, w, b):
        first_logits = tf.matmul(tf_dataset, w) + b
        relu_layer = tf.nn.relu(first_logits)
        relu_layer_dropout = tf.nn.dropout(relu_layer, 0.5)

        return relu_layer_dropout


    def get_logits_full(tf_dataset):
        logits_1 = get_dropout_result(tf_dataset, weights1, biases1)
        logits_2 = get_dropout_result(logits_1, weights2, biases2)
        logits_3 = get_dropout_result(logits_2, weights3, biases3)
        logits_4 = get_dropout_result(logits_3, weights4, biases4)
        logits_5 = get_dropout_result(logits_4, weights5, biases5)
        logits_6 = get_dropout_result(logits_5, weights6, biases6)
        logits_7 = get_dropout_result(logits_6, weights7, biases7)
        logits_8 = get_dropout_result(logits_7, weights8, biases8)
        logits_9 = tf.matmul(logits_8, weights9) + biases9

        return logits_9


    logits = get_logits_full(tf_training_dataset)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_training_labels, logits=logits))

    regularizers = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3) + tf.nn.l2_loss(
        weights4) + tf.nn.l2_loss(weights5) + tf.nn.l2_loss(weights6) + tf.nn.l2_loss(weights7) + tf.nn.l2_loss(
        weights8) + tf.nn.l2_loss(weights9)

    loss = tf.reduce_mean(loss + beta * regularizers)

    learning_rate = tf.train.exponential_decay(0.6, global_step=global_step, decay_steps=1000000, decay_rate=0.98,
                                               staircase=True)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    training_prediction = tf.nn.softmax(logits)

    valid_logits = get_logits_full(tf_valid_dataset)
    valid_prediction = tf.nn.softmax(valid_logits)

    test_logits = get_logits_full(tf_test_dataset)
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 15000

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):

        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (training_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = training_dataset[offset:(offset + batch_size), :]
        batch_labels = training_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_training_dataset: batch_data, tf_training_labels: batch_labels}
        _, l, predictions, r, w1, w5 = session.run(
            [optimizer, loss, training_prediction, regularizers, weights1, weights5], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
