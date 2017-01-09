import cv2
import numpy as np

# http://www.cs.toronto.edu/~adeandrade/assets/bpfcnnatorii.pdf
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
# https://github.com/jpthalman/CarND/blob/master/Projects/P2-TrafficSigns/Traffic_Signs_Recognition.ipynb
# def yuv_normalize(img):
#     img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
#     img = cv.normalize()
#

def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img


def translate(img):
    x = img.shape[0]
    y = img.shape[1]

    x_shift = np.random.uniform(-0.4 * x, 0.4 * x)
    y_shift = np.random.uniform(-0.4 * y, 0.4 * y)
    print(x_shift, y_shift)
    shift_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shift_img = cv2.warpAffine(img, shift_matrix, (x, y))

    return shift_img


def rotate(img):
    row, col, channel = img.shape

    angle = np.random.uniform(-30, 30)
    rotation_point = (row / 2, col / 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)

    rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))
    return rotated_img


def affine_transform(img):
    x, y, channel = img.shape

    pt1 = np.float32([[5, 5], [20, 5], [5, 30]])
    pt2 = np.float32([[2, 20], [20, 5], [10, 32]])

    matrix = cv2.getAffineTransform(pt1, pt2)
    result = cv2.warpAffine(img, matrix, (y, x))
    return result


# def gcn(image):

# def brighten(img, level):
#
# def darken(img, level):
#
# def blur(img, level):
