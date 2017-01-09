from net_builder import *
from image_processor import *
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


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

    # Layer 1: "Convolution" Layer:  Input 32x32x1 --> Output : 28x28x6 ---- 6 filter size = 5x5, stride = 1, pad = 0
    #            Activation        - ReLU
    #            Pooling           - MaxPooling --- 14x14x6
    layer_1 = conv_layer(x, w['layer_1'], biases['layer_1'])
    layer_1 = max_pool_layer(layer_1, 2)

    # Layer 2: "Convolution" Layer:  Input 14x14x6 --> Output : 10x10x16 ---- 6 filter size = 5x5, stride = 1, pad = 0
    #            Activation       -  ReLU
    #            Pooling          -  MedianPooling -- 5x5x16
    layer_2 = conv_layer(layer_1, w['layer_2'], biases['layer_2'])
    layer_2 = max_pool_layer(layer_2, 2)

    # Flatten Output : 5x5x16 --> 400
    flatten_layer = flatten(layer_2)

    # Layer 3: "Fully Connected" Layer: (Hidden Layer) Input: 400:  Output: 1x120
    #            Activation       -  ReLU
    fc1 = tf.add(tf.matmul(flatten_layer, w['fc_1']), biases['fc_1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
    # Layer 4: "Fully Connected" Layer: (Hidden Layer) Input: 120:  Output 84 outputs
    fc2 = tf.add(tf.matmul(fc1, w['fc_2']), biases['fc_2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    # Layer 5 : Final Layer:                           Input 84  :  Output 10 output
    logit = tf.add(tf.matmul(fc2, w['out']), biases['out'])

    return logit

# Hyper-parameters
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
KEEP_PROP = 0.5
save_loc = './saved_models/vgg.chkpt'

# ///////////////////////////// IMPORT DATA SET ////////////////////////////////////////
train_path = '../data/train.p'
test_path = '../data/test.p'
train, test = load_data(train_path, test_path)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# //////////////////// PRE-PROCESS DATA /////////////////////////////////////////////////
# Remove the previous weights and bias
tf.reset_default_graph()

# Features and Labels
features = tf.placeholder(tf.float32, (None, 32, 32, 3))  # Gray scale, Default image size is 32x32x3
labels = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(labels, len(set(y_train)))

logits = vgg_net(features, KEEP_PROP)
# logits = LeNet(features)

# Soft-Max
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss = tf.reduce_mean(cross_entropy)

# ////////////////////////// OPTIMIZATION /////////////////////////////
epoch_step = tf.Variable(0, name='epoch')
exp_lr = tf.train.exponential_decay(LEARNING_RATE, epoch_step, BATCH_SIZE, 0.95, name='expo_rate')
optimizer = tf.train.AdamOptimizer(exp_lr, name='adam_optimizer')
training_ops = optimizer.minimize(loss, global_step=epoch_step)
# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# ///////////////////////////VISUALIZATION/////////////////////////////////////
# TensorBoard - Debugging
# scalar_summary: values over time
# histogram_summary: value distribution from one particular layer.
recorder = tf.summary.FileWriter('./logs/', graph=tf.get_default_graph())


def evaluate(x, y_data, loss_op):
    num_of_examples = len(x)
    total_accuracy = 0
    val_loss = 0.0
    session = tf.get_default_session()

    for offset in range(0, num_of_examples, BATCH_SIZE):
        batch_features, batch_labels = x[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, _loss = session.run([accuracy_operation, loss_op], feed_dict={features: batch_features,
                                                                                labels: batch_labels})

        total_accuracy += (accuracy * len(batch_x))
        val_loss += _loss
    return total_accuracy / num_of_examples, val_loss/num_of_examples

# Train Model
with tf.Session() as sess:
    print("Start training...")
    try:
        saver.restore(sess, save_loc)
        print("Restored Model Successfully.")
    except Exception as e:
        print(e)
        print("No model found...Start building a new one")
        sess.run(tf.global_variables_initializer())

    num_examples = len(X_train)
    # sess.run(tf.local_variables_initializer())
    for i in range(EPOCHS):
        # Separate Training and Validation Set
        train_samples = np.ceil(int(num_examples * 0.8)).astype('uint32')
        X_train, y_train = shuffle(X_train, y_train)

        # Validation set
        x_val = X_train[train_samples:]
        y_val = y_train[train_samples:]

        print("EPOCH {} ".format(i + 1))
        for offset in range(0, train_samples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            batch_x = batch_x - np.mean(batch_x)
            batch_x = batch_x/np.std(batch_x, axis=0)

            _, lr = sess.run([training_ops, exp_lr], feed_dict={features: batch_x, labels: batch_y})

        validation_accuracy, validation_loss = evaluate(x_val, y_val, loss)
        print("LR: {:<7.8f} Validation loss: {:<6.5f} Validation Accuracy = {:.3f}".format(lr,
                                                                                           validation_loss,
                                                                                           validation_accuracy))

        print()
        # recorder.add_summary(log, i)
    saver.save(sess, save_loc)
    print("Train Model saved")
