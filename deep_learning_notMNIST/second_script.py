# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

base_path = "C:\\Users\\leflo\\Documents\\projects\\machine_learning_stuff\\data\\notMNIST\\"

pickle_file = base_path + 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    # print('Training set', train_dataset.shape, train_labels.shape)
    # print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)

    image_size = 28
    num_labels = 10


    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels


    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    # print('Training set', train_dataset.shape, train_labels.shape)
    # print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
num_nodes = 1024
batch_size = 256

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]), name="weights1")
    biases1 = tf.Variable(tf.zeros([num_nodes]), name="biases1")
    weights2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]), name="weights2")
    biases2 = tf.Variable(tf.zeros([num_labels]), name="biases2")

    # weights2 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes]))
    # biases2 = tf.Variable(tf.zeros([num_nodes]))
    # weights3 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    # biases3 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    def get_logits(tf_dataset):
        first_logits = tf.matmul(tf_dataset, weights1) + biases1
        relu_layer1 = tf.nn.relu(first_logits)
        second_logits = tf.matmul(relu_layer1, weights2) + biases2
        # relu_layer2 = tf.nn.relu(second_logits)
        return second_logits


    logits = get_logits(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    valid_logits = get_logits(tf_valid_dataset)
    valid_prediction = tf.nn.softmax(valid_logits)

    test_logits = get_logits(tf_test_dataset)
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 12001


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    saver = tf.train.Saver()
    for step in range(num_steps):

        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
        if step % 2000 == 0:
            saver.save(session, './tmp/model-2', global_step=step)

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
