import cv2 as cv
import numpy as np
import glob
from time import time
from img_proc import extract_face_region, encoding_img, img_array_to_encoding
from model_utils import load_facenet_model_h5
from face_validation import verify_from_db


def load_face_db(reg_dir, reg_db, model):
    files = glob.glob(reg_dir+'/*.jpg')
    for face_reg in files:
        reg_tag = face_reg.split(".")[0].replace("faceDB/", "")
        reg_name = reg_tag.replace("_baseline", "")

        anchor_img_emb = encoding_img(face_reg, model)
        if isinstance(anchor_img_emb, np.ndarray):
            reg_db[reg_name] = anchor_img_emb
    return reg_db


if __name__ == "__main__":
    INPUT_C, INPUT_X, INPUT_Y = 3, 96, 96
    FACE_REG_DIR = 'faceDB'
    FACE_REG = dict()
    COLOR = (255,0,0)
    EPSILON = 0.70

    # # Load model
    # face_model = load_facenet_model(model_shape=(3, 96, 96))
    face_model = load_facenet_model_h5('face_net.h5')
    print("Model loaded")

    faceDB = load_face_db(FACE_REG_DIR, FACE_REG, face_model)
    print("FaceDB loaded")

    # Start frame caption
    cap = cv.VideoCapture(0)
    cur_valid = False
    in_frame = []

    # Keep running
    while (True):
        ret, img = cap.read()

        # Setting refresh interval for validation
        if int(time()) % 15 == 0:
            cur_valid = False

        # Extract face image here
        faces = extract_face_region(img)

        # re-authenticate after resetting

        # Need to handle multiple faces
        for face, coord in faces:
            x, y, w, h = coord
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if not cur_valid:
                found = 0
                # Plot rectangle for detected
                # cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                person_embeding = img_array_to_encoding(face, face_model)

                # Validate image/face
                validated, identity_label, dist = verify_from_db(person_embeding, faceDB, EPSILON)

                if validated:
                    cur_valid = True
                    found = 1
                    cv.putText(img, "{0}, similarity dist {1} ".format(identity_label, str(round(dist,2))), (int(x + w + 15), int(y - 12)), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
                    in_frame.append((identity_label, dist, coord))

            else:
                cv.putText(img, "{0}, similarity dist {1} ".format(identity_label, str(round(dist,2))),
                           (int(x + w + 15), int(y - 12)), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)

        cv.imshow('img', img)

        if cv.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
