import tensorflow as tf
import utils.data_processor as util
from traffic_net import TrafficNet


TRAIN_PATH = '../train1.p'
TEST_PATH  = '../data/test.p'

train = util.load_data(TRAIN_PATH)
X_train, y_train = train['features'], train['labels']


# Pre-process data
# features = X_train/255                                       # scale pixel values to [0, 1]
# labels  = (y_train - len(set(y_train))) / len(set(y_train))  # scale label values to [-1, 1]

# Remove the previous weights and bias for new session
tf.reset_default_graph()
conv_net = TrafficNet()

# conv_net.train(features, y_train,
#                save_loc='./model/vgg.chkpt',
#                epochs=10,
#                learn_rate=0.001,
#                batch_size=128,
#                keep_prob=0.5,
#                acc_threshold=0.98)
# time.sleep(5)

test = util.load_data(TEST_PATH)
X_test, y_test = test['features'], test['labels']
X_test = X_test/255

conv_net.test(X_test, y_test, model='./model/vgg.chkpt', batch_size=256)
# conv_net.score(test_data='../data/test.p', plot=True, normalize=True)
