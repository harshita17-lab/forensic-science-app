import cv2 as cv
import numpy as np
from model import build_cnn, extract_features
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Load CNN model
cnn_model = build_cnn()

# PREPROCESSING


def preprocess(path):
    img = cv.imread(path, 0)
    img = cv.resize(img, (128, 128))

    # CLAHE enhancement
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img

# GABOR FILTER


def gabor_filter(img):
    kernel = cv.getGaborKernel((21, 21), 5, 0, 10, 0.5)
    return cv.filter2D(img, cv.CV_8UC3, kernel)

# ORB MATCHING + VISUALIZATION


def orb_match_visual(img1, img2):
    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Draw top matches
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None)
    cv.imwrite("static/match.jpg", img_matches)

    return len(matches)

# COSINE SIMILARITY


def cosine_similarity(f1, f2):
    return np.dot(f1, f2) / (norm(f1) * norm(f2))

# SAVE GRAPH


def save_visualization(orb_score, cnn_score):
    labels = ['ORB Matches', 'CNN Similarity']
    values = [orb_score, cnn_score * 100]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Fingerprint Comparison Metrics")
    plt.ylabel("Score")
    plt.savefig("static/result.png")
    plt.close()

# MAIN FUNCTION


def fingerprint_match(path1, path2):
    # Preprocess
    img1 = preprocess(path1)
    img2 = preprocess(path2)

    # Enhance
    img1 = gabor_filter(img1)
    img2 = gabor_filter(img2)

    # ORB Matching
    orb_score = orb_match_visual(img1, img2)

    # CNN Features
    f1 = extract_features(cnn_model, img1)
    f2 = extract_features(cnn_model, img2)

    cnn_score = cosine_similarity(f1, f2)

    # Save graph
    save_visualization(orb_score, cnn_score)

    # Decision (STRICT)
    is_match = (cnn_score > 0.8 and orb_score > 50)

    # Return HTML result
    return f"""
    <b>CNN Similarity:</b> {cnn_score:.2f}<br>
    <b>ORB Matches:</b> {orb_score}<br><br>
    <b>Result:</b> {'✅ Same Fingerprint' if is_match else '❌ Different Fingerprint'}
    """
