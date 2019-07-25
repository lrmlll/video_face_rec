import numpy as np
import cv2 as cv
from sklearn.metrics.pairwise import cosine_similarity

"""
Load pretrained model here?
Compare output
Look into faceDB
"""


def pairwise_cosine_similarity(test_representation, source_representation, threshold = 0.4):
    # a = np.matmul(np.transpose(source_representation), test_representation)
    # b = np.sum(np.multiply(source_representation, source_representation))
    # c = np.sum(np.multiply(test_representation, test_representation))
    # # return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    # similarity_score = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    similarity_score = cosine_similarity(test_representation, source_representation).squeez()
    if similarity_score < threshold:
        validated = True
    else:
        validated = False
    return similarity_score, validated


def linear_similarity(test_representation, source_representation, threshold):
    # encoding = img_to_encoding(image_path, model)
    # dist = np.linalg.norm(encoding - database[identity])
    if not threshold:
        threshold = 0.6
    dist = np.linalg.norm(test_representation - source_representation)
    if dist < threshold:
        validated = True
    else:
        validated = False

    return dist, validated


def verify(to_test_embedding, anchor_embedding, method= 'linear', threshold =None):
    """
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    # Use model to generate embedding
    if method == 'cosine':
        score, validated = pairwise_cosine_similarity(to_test_embedding, anchor_embedding, threshold)
    # elif method == 'linear' :
    else:
        score, validated = linear_similarity(to_test_embedding, anchor_embedding, threshold)

    return score, validated


def verify_from_db(person_embeding ,face_db, threshold=None):
    # This one takes in verify function to do pairwise validation with DB images, for now...
    cur_best = 10
    most_likely = None
    found = False
    for ppl_anchor_label, person_anchor_emb in face_db.items():
        dist, validated = verify(person_embeding, person_anchor_emb, 'linear', threshold)
        print( "Compoare with {0}, distance/similarity is {1}".format(ppl_anchor_label, dist))
        if validated and dist < cur_best  :
            found = True
            cur_best = dist
            most_likely = ppl_anchor_label

    return found, most_likely, cur_best
