import cv2
import numpy as np

# http://www.cs.toronto.edu/~adeandrade/assets/bpfcnnatorii.pdf
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
# https://github.com/jpthalman/CarND/blob/master/Projects/P2-TrafficSigns/Traffic_Signs_Recognition.ipynb


def random_transform(img):
    # There are total of 6 transformation
    # I will create an boolean array of 6 elements [ 0 or 1]
    a = np.random.randint(0, 2, [1, 5]).astype('bool')[0]
    # if a[0] == 1:
    #     img = translate(img)
    # if a[1] == 1:
    #     img = rotate(img)
    if a[2] == 1:
        img = shear(img)
    if a[3] == 1:
        img = blur(img)
    if a[4] == 1:
        img = gamma(img)
    return img


def translate(img):
    x = img.shape[0]
    y = img.shape[1]

    x_shift = np.random.uniform(-0.3 * x, 0.3 * x)
    y_shift = np.random.uniform(-0.3 * y, 0.3 * y)

    shift_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shift_img = cv2.warpAffine(img, shift_matrix, (x, y))

    return shift_img


def rotate(img):
    row, col, channel = img.shape

    angle = np.random.uniform(-60, 60)
    rotation_point = (row / 2, col / 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)

    rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))
    return rotated_img


def shear(img):
    x, y, channel = img.shape

    shear = np.random.randint(5,15)
    pts1 = np.array([[5, 5], [20, 5], [5, 20]]).astype('float32')
    pt1 = 5 + shear * np.random.uniform() - shear / 2
    pt2 = 20 + shear * np.random.uniform() - shear / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, M, (y, x))
    return result


def blur(img):
    r_int = np.random.randint(0, 2)
    odd_size = 2 * r_int + 1
    return cv2.GaussianBlur(img, (odd_size, odd_size), 0)


def gamma(img):
    gamma = np.random.uniform(0.3, 1.5)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_img = cv2.LUT(img, table)
    return new_img


def mean_image(images):
    # Find the mean image
    chann_index_swap = np.swapaxes(images, 0, 3)
    mean_img= [[[sum(pixel)/len(pixel) for pixel in col] for col in row] for row in chann_index_swap]
    mean_img = np.swapaxes(mean_img, 0, 2)
    mean_img = np.swapaxes(mean_img, 0, 1)
    return mean_img


def pre_process(img, mean_img):
    # Normalize data
    normalized_img = img/255.
    normalized_img -= mean_img
    return normalized_img

# def gcn(image):

# def brighten(img, level):
#
# def darken(img, level):
