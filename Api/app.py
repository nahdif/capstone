import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# URLs untuk tiga model
MODEL_URLS = {
    "skin": {
        "json": "https://storage.googleapis.com/your-bucket-name/skin_model.json",
        "weights": "https://storage.googleapis.com/your-bucket-name/skin_model_weights.h5"
    },
    "diabetes": {
        "json": "https://storage.googleapis.com/your-bucket-name/diabetes_model.json",
        "weights": "https://storage.googleapis.com/your-bucket-name/diabetes_model_weights.h5"
    },
    "heart": {
        "json": "https://storage.googleapis.com/your-bucket-name/heart_model.json",
        "weights": "https://storage.googleapis.com/your-bucket-name/heart_model_weights.h5"
    }
}

models = {}

def download_and_load_model(model_name):
    json_path = f"{model_name}_model.json"
    weights_path = f"{model_name}_model_weights.h5"
    if not os.path.exists(json_path):
        response = requests.get(MODEL_URLS[model_name]["json"])
        with open(json_path, "wb") as f:
            f.write(response.content)
    if not os.path.exists(weights_path):
        response = requests.get(MODEL_URLS[model_name]["weights"])
        with open(weights_path, "wb") as f:
            f.write(response.content)
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# Muat model
for model_name in MODEL_URLS.keys():
    models[model_name] = download_and_load_model(model_name)

def preprocess_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or "disease_type" not in request.form:
        return jsonify({"error": "File and disease_type are required"}), 400

    file = request.files["file"]
    disease_type = request.form["disease_type"]

    if disease_type not in models:
        return jsonify({"error": "Invalid disease_type"}), 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        image = preprocess_image(file_path)
        prediction = models[disease_type].predict(image)
        labels = ["Class 1", "Class 2", "Class 3"]  # Sesuaikan label
        result = {"prediction": labels[np.argmax(prediction)], "confidence": float(np.max(prediction))}
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
