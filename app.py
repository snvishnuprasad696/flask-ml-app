from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import io
import gdown

app = Flask(__name__)

# Path to save the TFLite model
model_path = "model.tflite"

# If the model is not already downloaded, download it from Google Drive
if not os.path.exists(model_path):
    file_id = "171lFA7aHe2E33CtWyfFm-OUQfc9u32U1"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        # Preprocess image
        img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Set the tensor to the input tensor of the model
        input_data = np.array(img_array, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the prediction results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0].tolist()

        # Get the class with the highest prediction score
        predicted_class_index = prediction.index(max(prediction))
        predicted_class_label = class_labels[predicted_class_index]

        return jsonify({"prediction": prediction, "predicted_class": predicted_class_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
