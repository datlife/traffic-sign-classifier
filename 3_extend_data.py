
# #######################################
# Generate new data set.
# ######################################
# Create a bar chart of frequencies
import utils.data_processor as util
import numpy as np
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
plt.interactive(False)

# Parameters
OUTPUT_FILE_PATH = '../train_gamma.p'
SCALE_FACTOR = 3.5
train_path = '../data/train.p'

train = util.load_data(train_path)
X_train, y_train = train['features'], train['labels']


extended_data, extended_labels = util.augment_data(X_train, y_train, scale=SCALE_FACTOR)
train_data = {'features': extended_data, 'labels': extended_labels}
util.save_data(train_data, OUTPUT_FILE_PATH)

item, count = np.unique(y_train, return_counts=True)
freq = np.array((item, count)).T
item2, count2 = np.unique(extended_labels, return_counts=True)
freq2 = np.array((item2, count2)).T


print('Before Data Augmentation: %d samples' % (y_train.shape[0]))
plt.figure(1)
plt.bar(item, count, alpha=0.2)
plt.title('Before Data Augmentation: Unequally Distributed Data')

print('After Data Augmentation: %d samples' % (extended_labels.shape[0]))
plt.figure(2)
plt.bar(item2, count2, alpha=0.2)
plt.title('After Data Augmentation: More Equally Distributed Data')

plt.show()


