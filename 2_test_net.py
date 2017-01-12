from utils.data_processor import load_data
from utils.net_builder import *
from sklearn.utils import shuffle


BATCH_SIZE = 4096
KEEP_PROP = 0.6
save_loc = './saved_models/vgg.chkpt'

# Remove the previous weights and bias
tf.reset_default_graph()

test_path = '../data/test.p'

test = load_data(test_path)
X_test, y_test = test['features'], test['labels']

# Remove the previous weights and bias
tf.reset_default_graph()

# Features and Labels
features = tf.placeholder(tf.float32, (None, 32, 32, 3))  # Gray scale, Default image size is 32x32x3
labels = tf.placeholder(tf.int32, None)

logits = traffic_sign_net(features, KEEP_PROP)
one_hot_y = tf.one_hot(labels, len(set(y_test)))

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
# Train Model
with tf.Session() as sess:
    print("Start Testing...")
    try:
        saver.restore(sess, save_loc)
        print("Restored Model Successfully.")
    except Exception as e:
        print(e)
        pass
    num_samples = len(X_test)
    X_test, y_test = shuffle(X_test, y_test)
    print("Testing on {} samples".format(num_samples))
    acc = 0.0
    for offset in range(0, num_samples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_test[offset:end], y_test[offset:end]
        _acc = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        acc += _acc*len(batch_x)

    print("Test Accuracy = {:.4f}".format(acc/num_samples))

    print("\nFinished Testing. Model is not saved")
