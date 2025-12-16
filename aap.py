import os, json, zipfile, urllib.request
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(APP_DIR, "model_artifacts")
ZIP_PATH = os.path.join(ART_DIR, "model_package.zip")
EXTRACT_DIR = os.path.join(ART_DIR, "unzipped")

MODEL_ZIP_URL = os.environ.get("MODEL_ZIP_URL", "").strip()

app = Flask(__name__)
CORS(app)

_infer = None
_labels = None

def _ensure_model():
    global _infer, _labels
    if _infer is not None and _labels is not None:
        return

    os.makedirs(ART_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        if not MODEL_ZIP_URL:
            raise RuntimeError("MODEL_ZIP_URL env var is missing")
        urllib.request.urlretrieve(MODEL_ZIP_URL, ZIP_PATH)

    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

    saved_model_dir = os.path.join(EXTRACT_DIR, "kaggle", "working", "model_saved")
    labels_path = os.path.join(EXTRACT_DIR, "kaggle", "working", "class_names.json")

    with open(labels_path, "r", encoding="utf-8") as f:
        _labels = json.load(f)

    loaded = tf.saved_model.load(saved_model_dir)
    _infer = loaded.signatures["serve"]

def preprocess(image_file):
    img = Image.open(image_file).convert("RGB").resize((256, 256))
    x = np.array(img).astype("float32")
    x = np.expand_dims(x, 0)
    return x

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/predict")
def predict():
    _ensure_model()
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400

    x = preprocess(request.files["image"])
    out = _infer(tf.constant(x))
    logits = out[list(out.keys())[0]].numpy()[0]
    probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(probs))

    return jsonify({
        "class_id": idx,
        "class_name": _labels[idx] if idx < len(_labels) else str(idx),
        "confidence": float(probs[idx])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
