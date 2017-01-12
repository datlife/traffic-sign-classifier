import numpy as np
import pickle
from utils.image_processor import random_transform


def load_data(train_path):
    training_file = train_path
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    return train


def save_data(file):
    with open('./train.p', 'wb') as f:
        pickle.dump(file, f)


def augment_data(X_train, y_train):

    total_traffic_signs = len(set(y_train))
    # Calculate how many images in one traffic sign
    ts, imgs_per_sign = np.unique(y_train, return_counts=True)
    avg_img_per_sign = np.ceil(np.mean(imgs_per_sign)).astype('uint32')

    # Based on that average, we can estimate how many images a traffic sign need to have
    # First, separate each traffic sign training set into different arrays
    separated_data = []
    for traffic_sign in range(total_traffic_signs):
        images_in_this_sign = X_train[y_train == traffic_sign, ...]
        separated_data.append(images_in_this_sign)

    # Second, for each data set, I generate new images randomly based on current total images
    expanded_data = np.array(np.zeros((1, 32, 32, 3)))
    expanded_labels = np.array([0])

    for sign, sign_images in enumerate(separated_data):
        # Determine if current traffic sign need more data
        need_more = bool(imgs_per_sign[sign] < avg_img_per_sign)

        if need_more is True:
            scale_factor = (avg_img_per_sign / imgs_per_sign[sign]).astype('uint32')
            new_images = []

            # Generate new images  <---- Could apply list comprehension here
            for img in sign_images:
                for _ in range(scale_factor):
                    new_images.append(random_transform(img))

            # Add old images and new images into 1 array
            new_images = np.concatenate((sign_images, new_images), axis=0)
            new_labels = np.full(len(new_images), sign, dtype='uint8')
            # Insert new_images to current data set
            expanded_data = np.concatenate((expanded_data, new_images), axis=0)
            expanded_labels = np.concatenate((expanded_labels, new_labels), axis=0)
        else:
            expanded_data = np.concatenate((expanded_data, sign_images), axis=0)
            expanded_labels = np.concatenate((expanded_labels, np.full(imgs_per_sign[sign], sign, dtype=u'uint8')))

    return expanded_data, expanded_labels

