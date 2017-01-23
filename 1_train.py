import tensorflow as tf
import utils.data_processor as util
from TrafficNet import TrafficNet

# Load Data
TRAIN_PATH = '../train_gamma.p'
train = util.load_data(TRAIN_PATH)
X_train, y_train = train['features'], train['labels']

# Pre-process data
features = X_train/255                                       # scale pixel values to [0, 1]
# labels  = (y_train - len(set(y_train))) / len(set(y_train))  # scale label values to [-1, 1]

# Remove the previous weights and bias for new session
tf.reset_default_graph()
conv_net = TrafficNet()

# Train CNN
conv_net.train(features, y_train,
               save_loc='./model/vgg.chkpt',
               epochs=2,
               learn_rate=0.001,
               batch_size=128,
               keep_prob=0.5,
               acc_threshold=0.996)
