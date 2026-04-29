from face_utils import face_match
import cv2 as cv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from model import build_cnn, extract_features

# Load CNN model
cnn_model = build_cnn()

# ---------------- PREPROCESSING ----------------


def preprocess(path):
    img = cv.imread(path, 0)
    img = cv.resize(img, (128, 128))

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img

# ---------------- GABOR FILTER ----------------


def gabor_filter(img):
    kernel = cv.getGaborKernel((21, 21), 5, 0, 10, 0.5)
    return cv.filter2D(img, cv.CV_8UC3, kernel)

# ---------------- ORB MATCHING ----------------


def orb_match_visual(img1, img2):
    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None)
    cv.imwrite("static/match.jpg", img_matches)

    return len(matches)

# ---------------- COSINE SIMILARITY ----------------


def cosine_similarity(f1, f2):
    if norm(f1) == 0 or norm(f2) == 0:
        return 0
    return np.dot(f1, f2) / (norm(f1) * norm(f2))

# ---------------- QUALITY METRICS ----------------


def blur_score(img):
    return cv.Laplacian(img, cv.CV_64F).var()


def contrast_score(img):
    return img.std()

# ---------------- VISUALIZATION ----------------


def save_visualization(orb_score, cnn_score):
    labels = ['ORB Matches', 'CNN Similarity']
    values = [orb_score, cnn_score * 100]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Fingerprint Comparison Metrics")
    plt.ylabel("Score")
    plt.savefig("static/result.png")
    plt.close()

# ---------------- MAIN FINGERPRINT FUNCTION ----------------


def fingerprint_match(path1, path2):
    # Preprocess
    img1 = preprocess(path1)
    img2 = preprocess(path2)

    # Enhance
    img1 = gabor_filter(img1)
    img2 = gabor_filter(img2)

    # ORB
    orb_score = orb_match_visual(img1, img2)

    # CNN features
    f1 = extract_features(cnn_model, img1)
    f2 = extract_features(cnn_model, img2)

    cnn_score = cosine_similarity(f1, f2)

    # -------- QUALITY --------
    blur1 = blur_score(img1)
    blur2 = blur_score(img2)

    contrast1 = contrast_score(img1)
    contrast2 = contrast_score(img2)

    blur_norm = min(1.0, (blur1 + blur2) / 2 / 500)
    contrast_norm = min(1.0, (contrast1 + contrast2) / 2 / 100)

    quality_score = (blur_norm + contrast_norm) / 2

    # -------- NORMALIZE ORB --------
    orb_norm = orb_score / (orb_score + 50)

    # -------- ADAPTIVE FUSION --------
    w_cnn = 0.5 + 0.3 * quality_score
    w_orb = 1.0 - w_cnn

    final_score = (w_cnn * cnn_score) + (w_orb * orb_norm)

    # Decision
    is_match = final_score > 0.6

    # Save visualization
    save_visualization(orb_score, cnn_score)

    # -------- DEBUG LOG (important for research) --------
    print(f"CNN: {cnn_score:.2f}, ORB: {orb_score}, Quality: {quality_score:.2f}, Final: {final_score:.2f}")

    # -------- SAVE RESULTS (for FAR/FRR later) --------
    with open("results.txt", "a") as f:
        f.write(
            f"{cnn_score},{orb_score},{quality_score},{final_score},{is_match}\n")

    # -------- HTML OUTPUT --------
    html_output = f"""
    <b>CNN Similarity:</b> {cnn_score:.2f}<br>
    <b>ORB Matches:</b> {orb_score}<br>
    <b>Quality Score:</b> {quality_score:.2f}<br>
    <b>Final Score:</b> {final_score:.2f}<br><br>
    <b>Result:</b> {'✅ Same Fingerprint' if is_match else '❌ Different Fingerprint'}
    """

    # IMPORTANT: return BOTH score and HTML
    return final_score, html_output


# ---------------- MULTIMODAL FUSION ----------------


def multimodal_match(fp1, fp2, face1, face2):

    # Fingerprint
    fp_score, fp_result = fingerprint_match(fp1, fp2)

    # Face
    face_score = face_match(face1, face2)

    # -------- SIMPLE FUSION --------
    final_score = 0.6 * fp_score + 0.4 * face_score

    return final_score, fp_result, face_score
