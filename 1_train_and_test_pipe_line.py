import tensorflow as tf
import time
from traffic_net import TrafficNet

"""
########################################
# Generate new data set. Only run once
#######################################
import utils.data_processor as util
train_path = '../data/train.p'
train = util.load_data(train_path)
X_train, y_train = train['features'], train['labels']
X_train, y_train = util.augment_data(X_train, y_train)
train_data = {'features': X_train, 'labels': y_train}
util.save_data(train_data, '../train.p')

"""
# Remove the previous weights and bias for new session
tf.reset_default_graph()
conv_net = TrafficNet()

conv_net.train(train_file='../train.p',
               save_loc='../saved_models/traffic-net.chkpt',
               epochs=20,
               learn_rate=0.001,
               batch_size=256,
               keep_prob=0.6)
time.sleep(5)
conv_net.test(test_file='../data/test.p', model='../saved_models/', batch_size=256)
