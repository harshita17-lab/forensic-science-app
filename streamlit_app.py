import streamlit as st
import os
from utils import fingerprint_match, multimodal_match
from face_utils import face_match

st.title("🔍 Multi-Modal Biometric System")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- FINGERPRINT ----------------
st.header("Fingerprint Recognition")

fp1 = st.file_uploader("Upload Fingerprint 1", type=["jpg", "png"])
fp2 = st.file_uploader("Upload Fingerprint 2", type=["jpg", "png"])

# ---------------- FACE ----------------
st.header("Face Recognition")

face1 = st.file_uploader("Upload Face 1", type=["jpg", "png"], key="face1")
face2 = st.file_uploader("Upload Face 2", type=["jpg", "png"], key="face2")

# ---------------- MODE ----------------
mode = st.selectbox(
    "Select Mode",
    ["Fingerprint Only", "Face Only", "Multimodal"]
)

# ---------------- BUTTON ----------------
if st.button("Compare"):

    # -------- Save files first --------
    path_fp1 = path_fp2 = None
    path_face1 = path_face2 = None

    if fp1:
        path_fp1 = os.path.join(UPLOAD_FOLDER, fp1.name)
        with open(path_fp1, "wb") as f:
            f.write(fp1.getbuffer())

    if fp2:
        path_fp2 = os.path.join(UPLOAD_FOLDER, fp2.name)
        with open(path_fp2, "wb") as f:
            f.write(fp2.getbuffer())

    if face1:
        path_face1 = os.path.join(UPLOAD_FOLDER, face1.name)
        with open(path_face1, "wb") as f:
            f.write(face1.getbuffer())

    if face2:
        path_face2 = os.path.join(UPLOAD_FOLDER, face2.name)
        with open(path_face2, "wb") as f:
            f.write(face2.getbuffer())

    # -------- PROCESS BASED ON MODE --------

    if mode == "Fingerprint Only":
        if path_fp1 and path_fp2:
            score, result = fingerprint_match(path_fp1, path_fp2)
            st.markdown(result, unsafe_allow_html=True)
        else:
            st.warning("Upload both fingerprint images")

    elif mode == "Face Only":
        if path_face1 and path_face2:
            face_score = face_match(path_face1, path_face2)
            st.write(f"Face Similarity: {face_score:.2f}")
        else:
            st.warning("Upload both face images")

    elif mode == "Multimodal":
        if path_fp1 and path_fp2 and path_face1 and path_face2:
            final_score, fp_result, face_score = multimodal_match(
                path_fp1, path_fp2, path_face1, path_face2
            )

            st.markdown(fp_result, unsafe_allow_html=True)
            st.write(f"Face Similarity: {face_score:.2f}")
            st.write(f"Final Combined Score: {final_score:.2f}")
        else:
            st.warning("Upload all required images")
