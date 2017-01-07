import tensorflow as tf
import pickle
from sklearn.utils import shuffle

# Hyper Parameters
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


class TrainNet:
    """
    Training a Convolution Neural Nets
    """
    def __init__(self):
        self.training_file = None
        self.testing_file = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        # Features and Labels
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, None)
        self.one_hot_y = tf.one_hot(self.y, 10)

        # Loss Optimizer
        self.logits = None
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.one_hot_y)
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.training_op = self.optimizer.minimize(self.loss)

        # Model Evaluation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

    def load_data(self, train_path, test_path):
        self.training_file = train_path
        self.testing_file = test_path
        with open(self.training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(self.testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_test, self.y_test = test['features'], test['labels']

    def load_network(self, conv_net):


    def evaluate(X_data, y_data):
        num_exps = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for off_set in range(0, num_exps, BATCH_SIZE):
            b_x, b_y = X_data[off_set:off_set+BATCH_SIZE], y_data[off_set:off_set+BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, feed_dict={x: b_x, y: b_y})
            total_accuracy += (accuracy*len(b_x))

        return total_accuracy/num_examples


# Train Model
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_op, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'VGG-Net')
    print("Model saved")
