import cv2 as cv
import numpy as np
from numpy.linalg import norm

# Face detector
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def preprocess_face(path):
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv.resize(face, (128, 128))

    return face


def extract_face_features(img):
    return img.flatten() / 255.0


def face_similarity(f1, f2):
    if norm(f1) == 0 or norm(f2) == 0:
        return 0
    return np.dot(f1, f2) / (norm(f1) * norm(f2))


def face_match(path1, path2):
    f1 = preprocess_face(path1)
    f2 = preprocess_face(path2)

    if f1 is None or f2 is None:
        return 0

    feat1 = extract_face_features(f1)
    feat2 = extract_face_features(f2)

    return face_similarity(feat1, feat2)
