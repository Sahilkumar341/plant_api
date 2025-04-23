from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import gdown

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# --------------------------
# Step 1: Download model if not already
MODEL_PATH = "model_compressed.h5"

FILE_ID = "1p-BH4TKdSe9Azq0jifqYbwkX4vLK1LZU"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# --------------------------
# Step 2: Load model after downloading
model = tf.keras.models.load_model(MODEL_PATH)

# --------------------------
# Step 3: Preprocess image
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print("Image preprocessing error:", e)
        raise

# --------------------------
# Step 4: Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        if image_file.content_type not in ['image/jpeg', 'image/png']:
            return jsonify({'error': 'Invalid image format'}), 400

        image_tensor = preprocess_image(image_file.read())
        prediction = model.predict(image_tensor)

        plant_id = int(np.argmax(prediction[0]))
        disease_id = int(np.argmax(prediction[1]))
        severity_id = int(np.argmax(prediction[2]))

        return [plant_id, disease_id, severity_id]

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Error processing image'}), 500

# --------------------------
# Step 5: Start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
