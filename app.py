from flask import Flask, render_template, request
import os
from utils import fingerprint_match, multimodal_match
from face_utils import face_match

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    face_result = None
    final_score = None

    if request.method == "POST":

        mode = request.form.get("mode")

        # -------- FILES --------
        file1 = request.files.get("file1")   # fingerprint 1
        file2 = request.files.get("file2")   # fingerprint 2
        face1 = request.files.get("face1")
        face2 = request.files.get("face2")

        path_fp1 = path_fp2 = None
        path_face1 = path_face2 = None

        # -------- SAVE FILES --------
        if file1:
            path_fp1 = os.path.join(
                app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(path_fp1)

        if file2:
            path_fp2 = os.path.join(
                app.config['UPLOAD_FOLDER'], file2.filename)
            file2.save(path_fp2)

        if face1:
            path_face1 = os.path.join(
                app.config['UPLOAD_FOLDER'], face1.filename)
            face1.save(path_face1)

        if face2:
            path_face2 = os.path.join(
                app.config['UPLOAD_FOLDER'], face2.filename)
            face2.save(path_face2)

        # -------- MODES --------

        if mode == "fingerprint":
            if path_fp1 and path_fp2:
                score, result = fingerprint_match(path_fp1, path_fp2)

        elif mode == "face":
            if path_face1 and path_face2:
                face_score = face_match(path_face1, path_face2)
                face_result = f"Face Similarity: {face_score:.2f}"

        elif mode == "multimodal":
            if path_fp1 and path_fp2 and path_face1 and path_face2:
                final_score, result, face_score = multimodal_match(
                    path_fp1, path_fp2, path_face1, path_face2
                )
                face_result = f"Face Similarity: {face_score:.2f}"

    return render_template(
        "index.html",
        result=result,
        face_result=face_result,
        final_score=final_score
    )


if __name__ == "__main__":
    app.run(debug=True)
