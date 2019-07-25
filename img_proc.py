import cv2 as cv
import numpy as np
from inception_blocks_v2 import *

"""
Load image into array
Extract face region using OpenCV haarcascade
Need to use the opencv xml file
"""

CV_PROFILE = 'haarcascade_frontalface_default.xml'
FACE_CLASS = cv.CascadeClassifier(CV_PROFILE)
INPUT_C, INPUT_X, INPUT_Y = 3, 96, 96
# face_model = load_model(model_shape=(3, 96, 96))


def read_from_file(file_path):
    """
    Take in
    :param file_path: file path to the image
    :return:  numpy ndarray
    """
    img_array = cv.imread(file_path)
    return img_array


def extract_face_region(cv_img):
    """
    Extract face pixel from current image
    :param cv_img: raw numpy.array from imread
    :return: List[ (np.ndarry , (coor) )] , list of ( extract_face_region, coord) pairs
    """
    detected_face = None
    x = y = w = h = None
    detected_faces = []

    region = FACE_CLASS.detectMultiScale(cv_img, 1.3, 5)

    # Following is drawing rectangle on img
    for (x, y, w, h) in region:
        cv.rectangle(cv_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        detected_face = cv_img[y: y+h, x: x+w]
        # detected_face.shape  rectangle shape

        detected_faces.append((detected_face, (x, y, w, h) ))
    return detected_faces


def img_array_to_encoding(detected_face, model):
    """
    Takes in one set of detected region
    Resize into model input shape
    """
    img = detected_face[..., ::-1]
    squeezed_img = cv.resize(img, (INPUT_X, INPUT_Y))
    squeezed_img = np.around(np.transpose(squeezed_img, (2,0,1))/255.0, decimals=12)
    embedding = model.predict_on_batch([[squeezed_img]])
    return embedding


def encoding_img(img_path, model):
    img_array = read_from_file(img_path)
    extracted, coord = extract_face_region(img_array)[0]
    if coord:
        img_emb = img_array_to_encoding(extracted, model)
        return img_emb
    else:
        return None
