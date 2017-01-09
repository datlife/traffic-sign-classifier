from net_builder import *
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


def vgg_net(x, keep_prob=0.5):
    shape = 32
    tf.variable_scope(tf.get_variable_scope())
    w = {
        'layer_1': weights('layer_1', [3, 3, 3, 64]), 'layer_2': weights('layer_2', [3, 3, 64, 64]),
        'layer_3': weights('layer_3', [3, 3, 64, 128]), 'layer_4': weights('layer_4', [3, 3, 128, 128]),

        'layer_5': weights('layer_5', [3, 3, 128, 256]),
        'layer_6': weights('layer_6', [3, 3, 256, 256]), 'layer_7': weights('layer_7', [3, 3, 256, 256]),

        'layer_8': weights('layer_8', [3, 3, 256, 512]),
        'layer_9': weights('layer_9', [3, 3, 512, 512]), 'layer_10': weights('layer_10', [3, 3, 512, 512]),

        'fc1': weights('fc1', [(shape/16)*(shape/16)*512, 4096]),
        'fc3': weights('fc3', [4096, 1000]),
        'logit': weights('logits', [1000, 43])
    }
    b = {
        'layer_1': biases('layer_1', 64), 'layer_2': biases('layer_2', 64),
        'layer_3': biases('layer_3', 128), 'layer_4': biases('layer_4', 128),

        'layer_5': biases('layer_5', 256), 'layer_6': biases('layer_6', 256), 'layer_7': biases('layer_7', 256),
        'layer_8': biases('layer_8', 512), 'layer_9': biases('layer_9', 512), 'layer_10': biases('layer_10', 512),
        'fc1': biases('fc1', 4096),
        'fc3': biases('fc3', 1000),
        'logit': biases('logits', 43)
    }

    # Based on VGG-Net Architecture
    conv3_64 = conv_layer(x, w['layer_1'], b['layer_1'])
    conv3_64_2 = conv_layer(conv3_64, w['layer_2'], b['layer_2'])
    pool_1 = max_pool_layer(conv3_64_2)

    conv3_128 = conv_layer(pool_1, w['layer_3'], b['layer_3'])
    conv3_128_2 = conv_layer(conv3_128, w['layer_4'], b['layer_4'])
    pool_2 = max_pool_layer(conv3_128_2)
    pool_2 = tf.nn.dropout(pool_2, keep_prob)

    conv3_256 = conv_layer(pool_2, w['layer_5'], b['layer_5'])
    conv3_256_2 = conv_layer(conv3_256, w['layer_6'], b['layer_6'])
    conv3_256_3 = conv_layer(conv3_256_2, w['layer_7'], b['layer_7'])
    pool_3 = max_pool_layer(conv3_256_3)
    pool_3 = tf.nn.dropout(pool_3, keep_prob)

    conv3_512 = conv_layer(pool_3, w['layer_8'], b['layer_8'])
    conv3_512_2 = conv_layer(conv3_512, w['layer_9'], b['layer_9'])
    conv3_512_3 = conv_layer(conv3_512_2, w['layer_10'], b['layer_10'])
    pool_4 = max_pool_layer(conv3_512_3)
    pool_4 = tf.nn.dropout(pool_4, keep_prob)

    flatten_layer = flatten(pool_4)

    fc1 = tf.matmul(flatten_layer, w['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc3 = tf.add(tf.matmul(fc1, w['fc3']), b['fc3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob)

    logit = tf.add(tf.matmul(fc3, w['logit']), b['logit'])

    return logit


def vgg_net2(x, keep_prob=0.5):
    shape = 32
    tf.variable_scope(tf.get_variable_scope())
    w = {
        'layer_1': weights('layer_1', [3, 3, 3, 64]), 'layer_2': weights('layer_2', [3, 3, 64, 64]),
        'layer_3': weights('layer_3', [3, 3, 64, 128]), 'layer_4': weights('layer_4', [3, 3, 128, 128]),

        'layer_5': weights('layer_5', [3, 3, 128, 256]),
        'layer_6': weights('layer_6', [3, 3, 256, 256]), 'layer_7': weights('layer_7', [3, 3, 256, 256]),

        'layer_8': weights('layer_8', [3, 3, 256, 512]),
        'layer_9': weights('layer_9', [3, 3, 512, 512]), 'layer_10': weights('layer_10', [3, 3, 512, 512]),


        'fc1': weights('fc1', [(shape/16)*(shape/16)*512, 4096]),
        'fc2': weights('fc2', [4096, 4096]),
        'fc3': weights('fc3', [4096, 1000]),
        'logit': weights('logits', [1000, 43])

    }
    b = {
        'layer_1': biases('layer_1', 64), 'layer_2': biases('layer_2', 64),
        'layer_3': biases('layer_3', 128), 'layer_4': biases('layer_4', 128),

        'layer_5': biases('layer_5', 256), 'layer_6': biases('layer_6', 256), 'layer_7': biases('layer_7', 256),
        'layer_8': biases('layer_8', 512), 'layer_9': biases('layer_9', 512), 'layer_10': biases('layer_10', 512),
        'fc1': biases('fc1', 4096),
        'fc2': biases('fc2', 4096),
        'fc3': biases('fc3', 1000),
        'logit': biases('logits', 43)
    }

    # Based on VGG-Net Architecture
    conv3_64 = conv_layer(x, w['layer_1'], b['layer_1'])
    conv3_64_2 = conv_layer(conv3_64, w['layer_2'], b['layer_2'])
    pool_1 = max_pool_layer(conv3_64_2)

    conv3_128 = conv_layer(pool_1, w['layer_3'], b['layer_3'])
    conv3_128_2 = conv_layer(conv3_128, w['layer_4'], b['layer_4'])
    pool_2 = max_pool_layer(conv3_128_2)

    conv3_256 = conv_layer(pool_2, w['layer_5'], b['layer_5'])
    conv3_256_2 = conv_layer(conv3_256, w['layer_6'], b['layer_6'])
    conv3_256_3 = conv_layer(conv3_256_2, w['layer_7'], b['layer_7'])
    pool_3 = max_pool_layer(conv3_256_3)

    conv3_512 = conv_layer(pool_3, w['layer_8'], b['layer_8'])
    conv3_512_2 = conv_layer(conv3_512, w['layer_9'], b['layer_9'])
    conv3_512_3 = conv_layer(conv3_512_2, w['layer_10'], b['layer_10'])
    pool_4 = max_pool_layer(conv3_512_3)

    flatten_layer = flatten(pool_4)

    fc1 = tf.add(tf.matmul(flatten_layer, w['fc1']), b['fc1'])
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

train_path = '../data/train.p'
test_path = '../data/test.p'

train, test = load_data(train_path, test_path)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.0002
KEEP_PROP = 0.5
save_loc = './saved_models/vgg.chkpt'


# Remove the previous weights and bias
tf.reset_default_graph()

# Features and Labels
features = tf.placeholder(tf.float32, (None, 32, 32, 3))  # Gray scale, Default image size is 32x32x3
labels = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(labels, len(set(y_train)))

logits = vgg_net(features, KEEP_PROP)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
# Train Model
with tf.Session() as sess:

    print("Start Testing...")
    try:
        # loader = tf.train.import_meta_graph('./vgg_net/vgg.chkpt.meta')
        # loader.restore(sess, save_loc)
        saver.restore(sess, save_loc)
        print("Restored Model Successfully.")
    except Exception as e:
        print(e)
        pass

    num_samples = len(X_test)
    # sess.run(tf.local_variables_initializer())
    for i in range(EPOCHS):
        X_test, y_test = shuffle(X_test, y_test)

        print("EPOCH {} : Testing on {} samples".format(i + 1, num_samples))
        acc = 0.0
        for offset in range(0, num_samples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_test[offset:end], y_test[offset:end]
            # Pre-process data
            batch_x = batch_x - np.mean(batch_x)
            batch_x = batch_x/np.std(batch_x, axis=0)

            _acc = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})

            acc += _acc*len(batch_x)

        print("Test Accuracy = {:.4f}".format(acc / num_samples))

        print("Finished Testing. Model is not saved")
