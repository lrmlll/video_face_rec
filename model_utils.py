import os
import numpy as np
from numpy import genfromtxt
from inception_blocks_v2 import *
from inception_blocks_v2 import triplet_loss
from tensorflow.keras.models import load_model


def load_weights(weight_path=None):
    if not weight_path:
        weight_path = './weights'
    file_names = filter(lambda f: not f.startswith('.'), os.listdir(weight_path))
    paths = {}
    weights_dict = {}

    for n in file_names:
        paths[n.replace('.csv', '')] = weight_path + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(weight_path + '/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(weight_path + '/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]
    return weights_dict


def load_weights_from_facenet(fr_model, weights_path):
    # Load weights from csv files (which was exported from Openface torch model)
    weights = WEIGHTS
    weights_dict = load_weights(weights_path)

    # Set layer weights of the model
    for name in weights:
        if fr_model.get_layer(name) is not None:
            fr_model.get_layer(name).set_weights(weights_dict[name])
        # elif model.get_layer(name) is not None:
        #     model.get_layer(name).set_weights(weights_dict[name])


def load_facenet_model(model_shape=None, weights_path=None):
    if not model_shape:
        model_shape = (3, 96, 96)
    face_rec_model = faceRecoModel(input_shape=model_shape)
    face_rec_model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])

    # Load weights csv into face_rec_model
    load_weights_from_facenet(face_rec_model, weights_path)
    return face_rec_model


def load_facenet_model_h5(model_path):
    model = load_model(model_path, custom_objects={'triplet_loss': triplet_loss})
    return model
