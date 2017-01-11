import tensorflow as tf
from tensorflow.contrib.layers import flatten


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
                                padding='SAME')
    return pool_layer


def fc_layer(data, weight, bias):
    """
    Fully Connected Layer
    """
    fully_connected_layer = tf.matmul(data, weight)
    fully_connected_layer = tf.add(fully_connected_layer, bias)
    fully_connected_layer = tf.nn.relu(fully_connected_layer)

    return fully_connected_layer


def evaluate(x, y_data, features, labels, accuracy_operation, loss_op, BATCH_SIZE):
    num_of_examples = len(x)
    total_accuracy = 0
    val_loss = 0.0
    session = tf.get_default_session()

    for offset in range(0, num_of_examples, BATCH_SIZE):
        batch_features, batch_labels = x[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, _loss = session.run([accuracy_operation, loss_op], feed_dict={features: batch_features,
                                                                                labels: batch_labels})
        total_accuracy += (accuracy * len(batch_features))
        val_loss += _loss
    return total_accuracy / num_of_examples, val_loss/num_of_examples


def traffic_sign_net(x, keep_prob=0.5):
    """
    Inspired by VGG-16 CNN Architecture. I dropped few layers so that I could run on my computer quicker.
    :param x:
    :param keep_prob:
    :return:
    """
    # image size is 32x32x3 ... change fc5 if needed
    # use drop out significantly improves the result
    # max pool affects the accuracy
    tf.variable_scope(tf.get_variable_scope())
    w = {
        'conv1_0': weights('conv1_0', [1, 1, 3, 3]),
        'conv1_1': weights('conv1_1', [3, 3, 3, 16]),
        'conv1_2': weights('conv1_2', [3, 3, 16, 16]),

        'conv2_1': weights('conv2_1', [3, 3, 16, 32]),
        'conv2_2': weights('conv2_2', [3, 3, 32, 32]),

        'conv3_1': weights('conv3_1', [3, 3, 32, 64]),
        'conv3_2': weights('conv3_2', [3, 3, 64, 64]),

        'conv4_1': weights('conv4_1', [3, 3, 64, 128]),
        'conv4_2': weights('conv4_2', [3, 3, 128, 128]),

        'fc1': weights('fc1', [2048, 1024]),
        'fc2': weights('fc2', [1024, 1024]),
        'logit': weights('logits', [1024, 43])  # 43 since there is 43 traffic signs in Germany Data set
    }
    b = {
        'conv1_0': biases('conv1_0', 3),
        'conv1_1': biases('conv1_1', 16),
        'conv1_2': biases('conv1_2', 16),

        'conv2_1': biases('conv2_1', 32),
        'conv2_2': biases('conv2_2', 32),

        'conv3_1': biases('conv3_1', 64),
        'conv3_2': biases('conv3_2', 64),

        'conv4_1': biases('conv4_1', 128),
        'conv4_2': biases('conv4_2', 128),

        'fc1': biases('fc1', 1024),
        'fc2': biases('fc2', 1024),
        'fc3': biases('fc3', 1024),

        'logit': biases('logits', 43)
    }
    # Inspired by VGG-Net Architecture
    conv1_0 = conv_layer(x, w['conv1_0'], b['conv1_0'])
    conv1_1 = conv_layer(conv1_0, w['conv1_1'], b['conv1_1'])
    conv1_2 = conv_layer(conv1_1, w['conv1_2'], b['conv1_2'])
    pool_1 = tf.nn.dropout(conv1_2, keep_prob+0.3)

    conv2_1 = conv_layer(pool_1, w['conv2_1'], b['conv2_1'])
    conv2_2 = conv_layer(conv2_1, w['conv2_2'], b['conv2_2'])
    pool_2 = max_pool_layer(conv2_2)
    pool_2 = tf.nn.dropout(pool_2, keep_prob+0.2)

    conv3_1 = conv_layer(pool_2, w['conv3_1'], b['conv3_1'])
    conv3_2 = conv_layer(conv3_1, w['conv3_2'], b['conv3_2'])
    pool_3 = max_pool_layer(conv3_2)
    pool_3 = tf.nn.dropout(pool_3, keep_prob+0.2)

    conv4_1 = conv_layer(pool_3, w['conv4_1'], b['conv4_1'])
    conv4_2 = conv_layer(conv4_1, w['conv4_2'], b['conv4_2'])
    pool_4 = max_pool_layer(conv4_2)
    pool_4 = tf.nn.dropout(pool_4, keep_prob+0.2)

    flatten_layer = flatten(pool_4)

    fc1 = tf.matmul(flatten_layer, w['fc1'])
    fc1 = tf.add(fc1, b['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.add(tf.matmul(fc1, w['fc2']), b['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    logit = tf.add(tf.matmul(fc2, w['logit']), b['logit'])

    return logit


def vgg_net(x, keep_prob=0.5):
    # image size is 32x32x3 ... change fc5 if needed
    tf.variable_scope(tf.get_variable_scope())
    w = {
        'conv1_1': weights('conv1_1', [3, 3, 3, 64]),
        'conv1_2': weights('conv1_2', [3, 3, 64, 64]),

        'conv2_1': weights('conv2_1', [3, 3, 64, 128]),
        'conv2_2': weights('conv2_2', [3, 3, 128, 128]),

        'conv3_1': weights('conv3_1', [3, 3, 128, 256]),
        'conv3_2': weights('conv3_2', [3, 3, 256, 256]),
        'conv3_3': weights('conv3_3', [3, 3, 256, 256]),

        'conv4_1': weights('conv4_1', [3, 3, 256, 512]),
        'conv4_2': weights('conv4_2', [3, 3, 512, 512]),
        'conv4_3': weights('conv4_3', [3, 3, 512, 512]),

        'conv5_1': weights('conv5_1', [3, 3, 512, 512]),
        'conv5_2': weights('conv5_2', [3, 3, 512, 512]),
        'conv5_3': weights('conv5_3', [3, 3, 512, 512]),

        'fc1': weights('fc1', [512, 4096]),
        'fc2': weights('fc2', [4096, 4096]),
        'fc3': weights('fc3', [4096, 1000]),
        'logit': weights('logits', [1000, 43])
    }
    b = {
        'conv1_1': biases('conv1_1', 64),
        'conv1_2': biases('conv1_2', 64),

        'conv2_1': biases('conv2_1', 128),
        'conv2_2': biases('conv2_2', 128),

        'conv3_1': biases('conv3_1', 256),
        'conv3_2': biases('conv3_2', 256),
        'conv3_3': biases('conv3_3', 256),

        'conv4_1': biases('conv4_1', 512),
        'conv4_2': biases('conv4_2', 512),
        'conv4_3': biases('conv4_3', 512),

        'conv5_1': biases('conv5_1', 512),
        'conv5_2': biases('conv5_2', 512),
        'conv5_3': biases('conv5_3', 512),

        'fc1': biases('fc1', 4096),
        'fc2': biases('fc2', 4096),
        'fc3': biases('fc3', 1000),

        'logit': biases('logits', 43)
    }
    # Inspired by VGG-Net Architecture
    conv1_1 = conv_layer(x, w['conv1_1'], b['conv1_1'])
    conv1_2 = conv_layer(conv1_1, w['conv1_2'], b['conv1_2'])
    pool_1 = max_pool_layer(conv1_2)

    conv2_1 = conv_layer(pool_1, w['conv2_1'], b['conv2_1'])
    conv2_2 = conv_layer(conv2_1, w['conv2_2'], b['conv2_2'])
    pool_2 = max_pool_layer(conv2_2)
    pool_2 = tf.nn.dropout(pool_2, keep_prob)

    conv3_1 = conv_layer(pool_2, w['conv3_1'], b['conv3_1'])
    conv3_2 = conv_layer(conv3_1, w['conv3_2'], b['conv3_2'])
    conv3_3 = conv_layer(conv3_2, w['conv3_3'], b['conv3_3'])
    pool_3 = max_pool_layer(conv3_3)
    pool_3 = tf.nn.dropout(pool_3, keep_prob)

    conv4_1 = conv_layer(pool_3, w['conv4_1'], b['conv4_1'])
    conv4_2 = conv_layer(conv4_1, w['conv4_2'], b['conv4_2'])
    conv4_3 = conv_layer(conv4_2, w['conv4_3'], b['conv4_3'])
    pool_4 = max_pool_layer(conv4_3)
    pool_4 = tf.nn.dropout(pool_4, keep_prob)

    conv5_1 = conv_layer(pool_4, w['conv5_1'], b['conv5_1'])
    conv5_2 = conv_layer(conv5_1, w['conv5_2'], b['conv5_2'])
    conv5_3 = conv_layer(conv5_2, w['conv5_3'], b['conv5_3'])
    pool_5 = max_pool_layer(conv5_3)
    pool_5 = tf.nn.dropout(pool_5, keep_prob)

    flatten_layer = flatten(pool_5)

    fc1 = tf.matmul(flatten_layer, w['fc1'])
    fc1 = tf.add(fc1, b['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.add(tf.matmul(fc1, w['fc2']), b['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    fc3 = tf.add(tf.matmul(fc2, w['fc3']), b['fc3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob)

    logit = tf.add(tf.matmul(fc3, w['logit']), b['logit'])

    return logit


def LeNet(x):

    tf.variable_scope(tf.get_variable_scope())
    w = {
               'layer_1': weights('layer_1', [5, 5, 3, 6]),
               'layer_2': weights('layer_2', [5, 5, 6, 16]),
               'fc_1': weights('fc_1', [1024, 120]),
               'fc_2': weights('fc_2', [120, 84]),
               'out': weights('out', [84, 43])
              }
    biases = {
               'layer_1': tf.Variable(tf.zeros(6), name='bias_layer_1'),
               'layer_2': tf.Variable(tf.zeros(16), name='bias_layer_2'),
               'fc_1': tf.Variable(tf.zeros(120), name='bias_fc1'),
               'fc_2': tf.Variable(tf.zeros(84), name='bias_fc1'),
               'out': tf.Variable(tf.zeros(43), name='bias_logits')
    }

    layer_1 = conv_layer(x, w['layer_1'], biases['layer_1'])
    layer_1 = max_pool_layer(layer_1, 2)

    layer_2 = conv_layer(layer_1, w['layer_2'], biases['layer_2'])
    layer_2 = max_pool_layer(layer_2, 2)

    flatten_layer = flatten(layer_2)

    fc1 = tf.add(tf.matmul(flatten_layer, w['fc_1']), biases['fc_1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    fc2 = tf.add(tf.matmul(fc1, w['fc_2']), biases['fc_2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    logit = tf.add(tf.matmul(fc2, w['out']), biases['out'])

    return logit


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
