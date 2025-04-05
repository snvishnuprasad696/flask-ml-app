from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import io
import gdown

app = Flask(__name__)

# Load your trained model
model_path = "trained_model_new.h5"
if not os.path.exists(model_path):
    file_id = "1Cp3vOFt3WJySK-_-7jI7OREplMi3TWZz"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
# Load the model
model = tf.keras.models.load_model(model_path)

# HARDCODED CLASS NAMES (NOT RECOMMENDED)
class_labels = ['biological', 'clothes', 'concrete', 'e-waste', 'explosive', 'glass', 'medical-waste', 'metal', 'paper', 'plastic', 'rubber', 'wood']  # Replace with your actual class names!

@app.route("/")
def home():
    return "Flask Model Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0].tolist()

        predicted_class_index = prediction.index(max(prediction))
        predicted_class_label = class_labels[predicted_class_index]

        return jsonify({"prediction": prediction, "predicted_class": predicted_class_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

