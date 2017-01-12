import numpy as np
import time
from utils.data_processor import load_data, augment_data, save_data
from utils.net_builder import *
from sklearn.utils import shuffle

# drop out rate : https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# Hyper-parameters

EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
KEEP_PROP = 0.6
save_loc = './saved_models/vgg.chkpt'


# ///////////////////////////// IMPORT DATA SET ////////////////////////////////////////
train_path = '../data/train.p'
train = load_data(train_path)
X_train, y_train = train['features'], train['labels']


# X_train, y_train = augment_data(X_train, y_train)
# train_data = {'features': X_train, 'labels': y_train}
# save_data(train_data)

# print('augmented data is done')
# ////////////////////////////// PRE-PROCESS DATA //////////////////////////////////////
# Remove the previous weights and bias
tf.reset_default_graph()

# Features and Labels
features = tf.placeholder(tf.float32, (None, 32, 32, 3))  # Gray scale, Default image size is 32x32x3
labels = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(labels, len(set(y_train)))

logits = traffic_sign_net(features, KEEP_PROP)

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
    start_time = time.clock()
    try:
        saver.restore(sess, save_loc)
        print("Restored Model Successfully.")
    except Exception as e:
        print(e)
        print("No model found...Start building a new one")
        sess.run(tf.initialize_all_variables())

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

            _, lr = sess.run([training_ops, exp_lr], feed_dict={features: batch_x, labels: batch_y})

        validation_accuracy, validation_loss = evaluate(x_val, y_val, loss)
        print("LR: {:<7.8f} Validation loss: {:<6.5f} Validation Accuracy = {:.3f}".format(lr,
                                                                                           validation_loss,
                                                                                           validation_accuracy))

        print()
        if validation_accuracy > 0.993:
            print("Training completed.")
            break

    saver.save(sess, save_loc)
    print("Train Model saved")

    # Calculate runtime and print out results
    train_time = time.clock() - start_time
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)
    print("\nOptimization Finished!! Training time: %02dh:%02dm:%02ds"
          % (h, m, s))
