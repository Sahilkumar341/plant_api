# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import io
# import os
# import gdown

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Improved CORS setup

# # --------------------------
# # Step 1: Download model if not already
# MODEL_PATH = "model_compressed.h5"
# FILE_ID = "1p-BH4TKdSe9Azq0jifqYbwkX4vLK1LZU"

# if not os.path.exists(MODEL_PATH):
#     print("Downloading model from Google Drive...")
#     gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# # --------------------------
# # Step 2: Load model
# model = tf.keras.models.load_model(MODEL_PATH)

# # --------------------------
# # Step 3: Preprocess image
# def preprocess_image(image_bytes):
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image = image.resize((64, 64))
#         image_array = np.array(image) / 255.0
#         return np.expand_dims(image_array, axis=0)
#     except Exception as e:
#         print("Image preprocessing error:", e)
#         raise

# # --------------------------
# # Step 4: Prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image uploaded'}), 400

#         image_file = request.files['image']
#         if image_file.content_type not in ['image/jpeg', 'image/png']:
#             return jsonify({'error': 'Invalid image format'}), 400

#         image_tensor = preprocess_image(image_file.read())
#         prediction = model.predict(image_tensor)

#         # Extract each prediction safely
#         try:
#             plant_index = int(np.argmax(prediction[0]))
#             disease_index = int(np.argmax(prediction[1]))
#             severity_index = int(np.argmax(prediction[2]))
#         except Exception as e:
#             print("Error extracting prediction indices:", e)
#             return jsonify({'error': 'Failed to extract predictions'}), 500

#         print("Predictions:", prediction)
#         print(f"Plant: {plant_index}, Disease: {disease_index}, Severity: {severity_index}")

#         return jsonify({
#             "plant": plant_index,
#             "disease": disease_index,
#             "severity": severity_index
#         })

#     except Exception as e:
#         print("Prediction error:", e)
#         return jsonify({'error': 'Error processing image'}), 500

# # --------------------------
# # Step 5: Run server
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import gdown

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------
# Step 1: Download TFLite model from Google Drive if not exists
TFLITE_MODEL_PATH = "model_quant.tflite"
DRIVE_FILE_ID = "1wDfAr6oFC8dmI1KG2I3-1l4SOKkcFvAg"
  # Change this to your actual file ID

if not os.path.exists(TFLITE_MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", TFLITE_MODEL_PATH, quiet=False)

# ----------------------------
# Step 2: Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Step 3: Preprocess image
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
        return image_array
    except Exception as e:
        print("Image preprocessing error:", e)
        raise

# ----------------------------
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

        interpreter.set_tensor(input_details[0]['index'], image_tensor)
        interpreter.invoke()

        plant_pred = interpreter.get_tensor(output_details[0]['index'])
        disease_pred = interpreter.get_tensor(output_details[1]['index'])
        severity_pred = interpreter.get_tensor(output_details[2]['index'])

        plant_index = int(np.argmax(plant_pred))
        disease_index = int(np.argmax(disease_pred))
        severity_index = int(np.argmax(severity_pred))

        print("Prediction:", {
            "plant": plant_index,
            "disease": disease_index,
            "severity": severity_index
        })

        return jsonify({
            "plant": plant_index,
            "disease": disease_index,
            "severity": severity_index
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Error processing image'}), 500

# ----------------------------
# Step 5: Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=int(os.environ.get("PORT", 5000)))

