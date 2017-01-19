import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils.image_processor import random_transform
from sklearn.metrics import confusion_matrix


def load_data(train_path):
    training_file = train_path
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    return train


def save_data(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def augment_data(X_train, y_train, scale=2):
    total_traffic_signs = len(set(y_train))
    # Calculate how many images in one traffic sign
    ts, imgs_per_sign = np.unique(y_train, return_counts=True)
    avg_per_sign = np.ceil(np.mean(imgs_per_sign)).astype('uint32')

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
        scale_factor = (scale*(avg_per_sign / imgs_per_sign[sign])).astype('uint32')
        print(sign, " ", avg_per_sign / imgs_per_sign[sign], " ", scale_factor)

        new_images = []
        # Generate new images  <---- Could apply list comprehension here
        for img in sign_images:
            for _ in range(scale_factor):
                new_images.append(random_transform(img))

        # Add old images and new images into 1 array
        if len(new_images) > 0:
            sign_images = np.concatenate((sign_images, new_images), axis=0)
        new_labels = np.full(len(sign_images), sign, dtype='uint8')
        # Insert new_images to current data set
        expanded_data = np.concatenate((expanded_data, sign_images), axis=0)
        expanded_labels = np.concatenate((expanded_labels, new_labels), axis=0)

    return expanded_data[1:], expanded_labels[1:]


def plt_confusion_matrix(labels, pred, normalize=False, title='Confusion matrix'):
    """
    Given one-hot encoded labels and preds, displays a confusion matrix.

    Arguments:
        `labels`:
            The ground truth one-hot encoded labels.
        `pred`:
            The one-hot encoded labels predicted by a model.
        `normalize`:
            If True, divides every column of the confusion matrix
            by its sum. This is helpful when, for instance, there are 1000
            'A' labels and 5 'B' labels. Normalizing this set would
            make the color coding more meaningful and informative.
    """
    labels = [label.argmax() for label in labels]
    pred = [label.argmax() for label in pred]

    classes = np.arange(len(set(labels)))

    cm = confusion_matrix(labels, pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def visualize_softmax_probabilities(model, image, num_probs=5, correct_prediction=None):
    import pandas as pd

    if image.ndim == 3:
        image = image[np.newaxis, ...]
    if image.shape[3] > 1:
        image = pre_process_gray(image)

    pred = model.predict(image)[0]
    top_k = np.argsort(pred)[:-(num_probs + 1):-1]

    pred = pd.DataFrame(data=np.array((top_k, pred[top_k])).T, columns=['ClassId', 'Probability'])
    names = pd.read_csv('./signnames.csv', header=0)
    pred = pd.merge(pred, names, how='inner', on='ClassId')

    fig = plt.figure(figsize=(11, 7))
    grid = plt.subplot(1, 2, 1)

    if correct_prediction is not None:
        name = names.at[correct_prediction, 'SignName']
        grid.set_title("Correct Prediction: " + name)

    plt.imshow(image[0, :, :, 0], cmap='gray')

    grid = plt.subplot(1, 2, 2)
    plt.barh(np.arange(num_probs)[::-1], np.array(pred['Probability']), align='center')
    plt.xlim([0, 1.0])
    plt.yticks(np.arange(num_probs)[::-1], np.array(pred['SignName']))

    plt.tight_layout()