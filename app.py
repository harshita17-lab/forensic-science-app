from flask import Flask, render_template, request
import os
from utils import fingerprint_match

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file1 = request.files["file1"]
        file2 = request.files["file2"]

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)

        file1.save(path1)
        file2.save(path2)

        result = fingerprint_match(path1, path2)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
