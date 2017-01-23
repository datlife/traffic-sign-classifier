import tensorflow as tf
import numpy as np
import cv2
import utils.data_processor as util
from TrafficNet import TrafficNet


TEST_PATH  = '../data/test.p'
test = util.load_data(TEST_PATH)
X_test, y_test = test['features'], test['labels']
X_test = X_test/255

tf.reset_default_graph()
conv_net = TrafficNet()
# # Test Image
# conv_net.test(X_test, y_test, model='./model/vgg.chkpt', batch_size=256)
#
# # Display Confusion Matrix
# conv_net.score(features=X_test, labels=y_test,
#                model='./model/vgg.chkpt', plot=True, normalize=True)
#

# Visualize softmax
# Map ClassID to traffic sign names
import csv
signs = []
with open('signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

import os
path = './difficult_images/'
hard_imgs = []
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hard_imgs.append(img)

# Display shape of test images and pre-process
hard_imgs = np.array(hard_imgs)
hard_imgs = hard_imgs / 255
conv_net.visualize_softmax(hard_imgs, signs, saved_model='./model/vgg.chkpt', top_k=5)