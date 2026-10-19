"""
FelineHealth-AI Flask API.

    GET  /api/health    -> is the server up and are the models loaded?
    POST /api/predict   -> form-data field "image" (jpg/png/webp, max 10 MB)

Run (from the FelineHealth-Ai project root, with venv active):
    python backend[application].py
"""
import io

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from inference import FelineHealthService

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
MAX_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
CORS(app)  # allow the React dev server (another port) to call the API

# load all models ONCE when the server starts (not on every request)
try:
    service = FelineHealthService()
    load_error = None
    print("Models loaded:", ", ".join(service.class_names))
except Exception as e:                      # server still starts, /api/health shows why
    service, load_error = None, str(e)
    print("WARNING: models not loaded:", e)


def error(message, code):
    return jsonify({"status": "error", "message": message}), code


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Flask backend!"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if service else "error",
                    "models_loaded": service is not None,
                    "classes": service.class_names if service else [],
                    "error": load_error})


@app.route("/api/predict", methods=["POST"])
def predict():
    if service is None:
        return error(f"Models are not loaded: {load_error}", 503)
    if "image" not in request.files:
        return error("No file uploaded. Send the photo in a form field called 'image'.", 400)
    f = request.files["image"]
    if not f.filename:
        return error("No file selected.", 400)
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXT:
        return error("Unsupported file type. Please upload a JPG, PNG or WEBP photo.", 400)
    data = f.read()
    try:
        Image.open(io.BytesIO(data)).verify()               # is it really an image?
    except (UnidentifiedImageError, OSError):
        return error("The file could not be read as an image.", 400)
    try:
        return jsonify(service.analyse(data))
    except Exception as e:                                   # never crash the server
        app.logger.exception("Prediction failed")
        return error(f"Prediction failed: {e}", 500)


@app.errorhandler(413)
def too_large(_):
    return error(f"File too large. Maximum size is {MAX_MB} MB.", 413)


if __name__ == "__main__":
    app.run(debug=True, port=5000)