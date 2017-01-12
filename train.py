import tensorflow as tf
from traffic_net import TrafficNet

# Remove the previous weights and bias
tf.reset_default_graph()

conv_net = TrafficNet()
conv_net.train(train_file='../data/train.p',
               save_loc='../saved_models/vgg.chkpt',
               epochs=5,
               learn_rate=0.001,
               batch_size=128,
               keep_prob=0.6)
