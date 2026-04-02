import streamlit as st
from utils import fingerprint_match
import os

st.title("🔍 Fingerprint Recognition System")

file1 = st.file_uploader("Upload Fingerprint 1", type=["jpg", "png"])
file2 = st.file_uploader("Upload Fingerprint 2", type=["jpg", "png"])

if file1 and file2:
    path1 = os.path.join("static/uploads", file1.name)
    path2 = os.path.join("static/uploads", file2.name)

    with open(path1, "wb") as f:
        f.write(file1.getbuffer())

    with open(path2, "wb") as f:
        f.write(file2.getbuffer())

    if st.button("Compare"):
        result = fingerprint_match(path1, path2)
        st.markdown(result, unsafe_allow_html=True)

        st.image("static/match.jpg", caption="ORB Matching")
        st.image("static/result.png", caption="Comparison Graph")
