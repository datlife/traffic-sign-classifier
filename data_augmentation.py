
# #######################################
# Generate new data set.
# ######################################
# Create a bar chart of frequencies
import utils.data_processor as util
import utils.image_processor as img_processor
import numpy as np
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
plt.interactive(False)


OUTPUT_FILE_PATH = '../train1.p'
SCALE_FACTOR = 3
train_path = '../data/train.p'

train = util.load_data(train_path)
X_train, y_train = train['features'], train['labels']

item, count = np.unique(y_train, return_counts=True)
freq = np.array((item, count)).T

print('Before Data Augmentation: %d samples' % (y_train.shape[0]))
plt.figure(1)
plt.bar(item, count, alpha=0.2)
plt.title('Before Data Augmentation: Unequally Distributed Data')

X_train, y_train = util.augment_data(X_train, y_train, scale=SCALE_FACTOR)
train_data = {'features': X_train, 'labels': y_train}
util.save_data(train_data, OUTPUT_FILE_PATH)

item2, count2 = np.unique(y_train, return_counts=True)
freq2 = np.array((item2, count2)).T

print('After Data Augmentation: %d samples' % (y_train.shape[0]))
plt.figure(2)
plt.bar(item2, count2, alpha=0.2)
plt.title('After Data Augmentation: More Equally Distributed Data')
print(X_train.shape)
print(y_train.shape)
plt.show()


