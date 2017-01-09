import tensorflow as tf
import pickle
from sklearn.utils import shuffle


def load_data(train_path, test_path):
    training_file = train_path
    testing_file = test_path
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return train, test


def weights(weight_name, shape=[]):
    """
    Return TensorFlow weights
    :param weight_name: name of weight
    :param shape: Number of features
    :return: TensorFlow weights
    """
    # Xavier Initialization - Linear only
    return tf.get_variable(weight_name, shape, initializer=tf.contrib.layers.xavier_initializer())


def biases(name, n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    b = tf.Variable(tf.zeros(n_labels),name=name)
    return b


def conv_layer(data, size, bias, stride=1):
    """
    Create a new Convolution Layer
    """
    layer = tf.nn.conv2d(input=data, filter=size, strides=[1, stride, stride, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)

    return layer


def max_pool_layer(data, sub_sampling_rate=2):
    """
    Sub-sampling data from Convolution Layer
    """
    k = sub_sampling_rate
    pool_layer = tf.nn.max_pool(value=data,
                                ksize=[1, k, k, 1],
                                strides=[1, k, k, 1],
                                padding='VALID')
    return pool_layer


def fc_layer(data, weight, bias):
    """
    Fully Connected Layer
    """
    fully_connected_layer = tf.matmul(data, weight)
    fully_connected_layer = tf.add(fully_connected_layer, bias)
    fully_connected_layer = tf.nn.relu(fully_connected_layer)

    return fully_connected_layer


def evaluate_validation_set(x_data, y_data, batch_size, features, labels, accuracy_operation, loss, summary_op):
    num_of_examples = len(x_data)
    total_accuracy = 0
    total_loss = 0.0
    sess = tf.get_default_session()
    x_data, y_data = shuffle(x_data, y_data)

    for off_set in range(0, num_of_examples, batch_size):
        end = off_set + batch_size
        batch_x, batch_y = x_data[off_set: end], y_data[off_set: end]
        accuracy, _loss, log = sess.run([accuracy_operation, loss, summary_op],
                                        feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += _loss
    return total_accuracy / num_of_examples, total_loss, log


# def batches(batch_size, features, labels):
#     """
#     Extract original data into mini-batches
#     :param batch_size: size of each batch ( default = 64)
#     :param features: number of features (input data)
#     :param labels:  number of labels (output data)
#     :return: array-like of mini batches
#     """
#     assert len(features) == len(labels)
#
#     f_size = np.ceil(len(features)/batch_size)
#     l_size = np.ceil(len(labels)/batch_size)
#
#     feature_batch = np.array_split(features, f_size)
#     label_batch   = np.array_split(labels, l_size)
#
#     output = [[feature_batch[i], label_batch[i]] for i in range(len(feature_batch))]
#
#     return output
#
#
# def variable_summaries(var):
#     """
#     Attach a lot of summaries to a Tensor (for TensorBoard visualization).
#     :param var: A tensor variable (weight, biases)
#     """
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#     tf.scalar_summary('mean', mean)
#     with tf.name_scope('stddev'):
#         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.scalar_summary('stddev', stddev)
#     tf.scalar_summary('max', tf.reduce_max(var))
#     tf.scalar_summary('min', tf.reduce_min(var))
#     tf.histogram_summary('histogram', var)
#
#
