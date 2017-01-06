import tensorflow as tf

class ConvNet:
    """
    Convolution Neural Network Object


    """
    def __init__(self):
        self.data_input = None
        self.model_saver = tf.train.Saver()

        self.convolution_layers = {}
        self.max_pooling_layers = {}
        self.fully_connected_layers = {}

    def add_conv_layer(self, name, data, size=[], bias, stride=1):
        """
        Create a new Convolution Layer
        :param name:
        :param data:
        :param size:
        :param stride:
        :return:
        """
        layer = tf.nn.conv2d(input=data,filter=size,strides=[1, stride, stride, 1], padding='SAME')
        layer = tf.nn.bias_add(layer, bias)
        layer = tf.nn.relu(layer)

        self.convolution_layers += {name: layer}
        return layer

    def add_max_pool_layer(self, name, data, sub_sampling_rate=2):
        """
        Sub-sampling data from Convolution Layer
        :param name: name for this max pool layer. stored in max_pooling_layer
        :param sub_sampling_rate: how fast this max pool "squeeze" the data size
        :return: subsample data
        """
        k = sub_sampling_rate
        max_pool_layer = tf.nn.max_pool(value=data,
                                        ksize=[1, k, k , 1],
                                        strides=[1, k, k, 1],
                                        padding='SAME')
        self.max_pooling_layers += {name: max_pool_layer}
        return max_pool_layer

    def add_fully_connected_layer(self, name, data, weight, bias):
        """

        :param name:
        :param data:
        :param weight:
        :param bias:
        :return:
        """
        fully_connected_layer = tf.matmul(data, weight)
        fully_connected_layer = tf.add(fully_connected_layer, bias)
        fully_connected_layer = tf.nn.relu(fully_connected_layer)

        self.fully_connected_layers += {name: fully_connected_layer}

        return fully_connected_layer

    def predict(self, image):

    def train(self):

    def process_data(self, data):

