from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from paths import notMNIST_pickle_file

with open(notMNIST_pickle_file, 'rb') as f:
    save = pickle.load(f)
    training_dataset = save['training_dataset']
    training_labels = save['training_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save

image_size = 28
num_labels = 10
num_channels = 1  # grayscale


def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(training_dataset, training_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
beta = 0.01

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer15_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer15_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer25_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer25_biases = tf.Variable(tf.zeros([depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    global_step = tf.Variable(0)

    # Model.
    def model(data):
        conv_1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
        conv_15 = tf.nn.conv2d(hidden_1, layer15_weights, [1, 1, 1, 1], padding='SAME')
        hidden_15 = tf.nn.relu(conv_15 + layer15_biases)
        pool_1 = tf.nn.max_pool(value=hidden_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv_2 = tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
        conv_25 = tf.nn.conv2d(hidden_2, layer25_weights, [1, 1, 1, 1], padding='SAME')
        hidden_25 = tf.nn.relu(conv_25 + layer25_biases)
        pool_2 = tf.nn.max_pool(value=hidden_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        shape = pool_2.get_shape().as_list()
        reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    learning_rate = tf.train.exponential_decay(0.1, global_step=global_step, decay_steps=100000, decay_rate=0.96,
                                               staircase=True)
    regularizers = tf.nn.l2_loss(layer1_biases) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer3_weights)
    regularizers += tf.nn.l2_loss(layer4_weights)

    loss = tf.reduce_mean(loss + beta * regularizers)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        if step % 50 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            if l > 85:
                lt = 0.01
                print("changing learning rate")
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
