import time
import numpy as np
from utils.data_processor import load_data
from utils.net_builder import *
from sklearn.utils import shuffle


class TrafficNet(object):
    
    def __init__(self):
        self.learn_rate = 0.001
        self.batch_size = 128
        self.keep_prob = 0.5

        # Features and Labels
        self.features = tf.placeholder(tf.float32, (None, 32, 32, 3))  # Gray scale, Default image size is 32x32x3
        self.labels = tf.placeholder(tf.int32, None)

        self.w = {
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
        self.b = {
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
        self.conv_0 = conv_layer(self.features, self.w['conv1_0'], self.b['conv1_0'])
        self.conv_1 = conv_layer(self.conv_0, self.w['conv1_1'], self.b['conv1_1'])
        self.conv_2 = conv_layer(self.conv_1, self.w['conv1_2'], self.b['conv1_2'])
        self.pool_1 = tf.nn.dropout(self.conv_2, self.keep_prob + 0.3)

        self.conv_1 = conv_layer(self.pool_1, self.w['conv2_1'], self.b['conv2_1'])
        self.conv_2 = conv_layer(self.conv_1, self.w['conv2_2'], self.b['conv2_2'])
        self.pool_2 = max_pool_layer(self.conv_2)
        self.pool_2 = tf.nn.dropout(self.pool_2, self.keep_prob + 0.2)

        self.conv_1 = conv_layer(self.pool_2, self.w['conv3_1'], self.b['conv3_1'])
        self.conv_2 = conv_layer(self.conv_1, self.w['conv3_2'], self.b['conv3_2'])
        self.pool_3 = max_pool_layer(self.conv_2)
        self.pool_3 = tf.nn.dropout(self.pool_3, self.keep_prob + 0.2)

        self.conv_1 = conv_layer(self.pool_3, self.w['conv4_1'], self.b['conv4_1'])
        self.conv_2 = conv_layer(self.conv_1, self.w['conv4_2'], self.b['conv4_2'])
        self.pool_4 = max_pool_layer(self.conv_2)
        self.pool_4 = tf.nn.dropout(self.pool_4, self.keep_prob + 0.2)

        self.flatten_layer = flatten(self.pool_4)

        self.fc1 = tf.matmul(self.flatten_layer, self.w['fc1'])
        self.fc1 = tf.add(self.fc1, self.b['fc1'])
        self.fc1 = tf.nn.relu(self.fc1)
        self.fc1 = tf.nn.dropout(self.fc1, self.keep_prob)

        self.fc2 = tf.add(tf.matmul(self.fc1, self.w['fc2']), self.b['fc2'])
        self.fc2 = tf.nn.relu(self.fc2)
        self.fc2 = tf.nn.dropout(self.fc2, self.keep_prob)

        self.logits = tf.add(tf.matmul(self.fc2, self.w['logit']), self.b['logit'])

    def train(self, train_file='../data/train.p', save_loc='../saved_models/vgg.chkpt',
              epochs=10, learn_rate=0.001, batch_size=128, keep_prob=0.5):

        # Update Learning Rate and Batch Size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.keep_prob = keep_prob

        train = load_data(train_file)
        x_train, y_train = train['features'], train['labels']
        one_hot_y = tf.one_hot(self.labels, len(set(y_train)))

        # Soft-Max
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y)
        loss = tf.reduce_mean(cross_entropy)
        # ////////////////////////// OPTIMIZATION /////////////////////////////
        epoch_step = tf.Variable(0, name='epoch')
        exp_lr = tf.train.exponential_decay(self.learn_rate, epoch_step, self.batch_size, 0.95, name='expo_rate')
        optimizer = tf.train.AdamOptimizer(exp_lr, name='adam_optimizer')
        training_ops = optimizer.minimize(loss, global_step=epoch_step)

        # Model Evaluation
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        # Train Model
        with tf.Session() as sess:
            print("Start training...")
            start_time = time.clock()
            try:
                saver.restore(sess, save_loc)
                print("Restored Model Successfully.")
            except Exception as e:
                print(e)
                print("No model found...Start building a new one")
                sess.run(tf.initialize_all_variables())

            num_examples = len(x_train)
            # sess.run(tf.local_variables_initializer())
            for i in range(epochs):
                # Separate Training and Validation Set
                train_samples = np.ceil(int(num_examples * 0.8)).astype('uint32')
                x_train, y_train = shuffle(x_train, y_train)

                # Validation set
                x_val = x_train[train_samples:]
                y_val = y_train[train_samples:]

                print("EPOCH {} ".format(i + 1))
                for offset in range(0, train_samples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = x_train[offset:end], y_train[offset:end]

                    _, lr = sess.run([training_ops, exp_lr], feed_dict={self.features: batch_x,
                                                                        self.labels: batch_y})

                validation_accuracy, validation_loss = self.evaluate(x_val, y_val, accuracy_operation, loss)

                print("LR: {:<7.8f} Validation loss: {:<6.5f} Validation Accuracy = {:.3f}".format(lr,
                                                                                                   validation_loss,
                                                                                                   validation_accuracy))
                print()
                if validation_accuracy > 0.995:
                    print("Reached accuracy requirement (99.5%). Training completed.")
                    break

            saver.save(sess, save_loc)
            print("Train Model saved")

            # Calculate runtime and print out results
            train_time = time.clock() - start_time
            m, s = divmod(train_time, 60)
            h, m = divmod(m, 60)
            print("\nOptimization Finished!! Training time: %02dh:%02dm:%02ds"
                  % (h, m, s))

    def test(self, test_file='../data/test.p', model='../saved_models/vgg.chkpt', batch_size=128):
        test_data = load_data(test_file)
        x_test, y_test = test_data['features'], test_data['labels']
        # Model Evaluation
        one_hot_y = tf.one_hot(self.labels, len(set(y_test)))
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        current_graph = tf.get_default_graph()
        # Train Model
        with tf.Session(graph=current_graph) as sess:
            print("Start Testing...")
            saver.restore(sess, tf.train.latest_checkpoint(model))
            print("Restored Model Successfully.")
            num_samples = len(x_test)
            x_test, y_test = shuffle(x_test, y_test)
            print("Testing on {} samples".format(num_samples))
            total_acc = 0.0
            for offset in range(0, num_samples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = x_test[offset:end], y_test[offset:end]

                _acc = sess.run(accuracy_operation, feed_dict={self.features: batch_x, self.labels: batch_y})
                total_acc += _acc * len(batch_x)
            print("Test Accuracy = {:.4f}".format(total_acc/num_samples))
            print("\nFinished Testing. Model is not saved")

    def evaluate(self, features, labels, acc_op, loss_op):
        num_of_examples = len(features)
        total_accuracy = 0
        val_loss = 0.0
        session = tf.get_default_session()

        for offset in range(0, num_of_examples, self.batch_size):
            batch_features = features[offset:offset + self.batch_size]
            batch_labels = labels[offset:offset + self.batch_size]

            accuracy, _loss = session.run([acc_op, loss_op],
                                          feed_dict={self.features: batch_features, self.labels: batch_labels})

            total_accuracy += (accuracy * len(batch_features))
            val_loss += _loss
        return total_accuracy / num_of_examples, val_loss/num_of_examples
