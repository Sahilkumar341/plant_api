# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import io
# import os
# import gdown

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # ----------------------------
# # Step 1: Download TFLite model from Google Drive if not exists
# TFLITE_MODEL_PATH = "model_quant_updated.tflite"
# DRIVE_FILE_ID = "1m9BQV3BHsL1fkzCljOtkr7Y4C08XsUAP"
#   # Change this to your actual file ID
# # https://drive.google.com/file/d/1m9BQV3BHsL1fkzCljOtkr7Y4C08XsUAP/view?usp=drive_link
# if not os.path.exists(TFLITE_MODEL_PATH):
#     print("Downloading model from Google Drive...")
#     gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", TFLITE_MODEL_PATH, quiet=False)

# # ----------------------------
# # Step 2: Load TFLite model
# interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # ----------------------------
# # Step 3: Preprocess image
# def preprocess_image(image_bytes):
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image = image.resize((64, 64))
#         image_array = np.array(image) / 255.0
#         image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
#         return image_array
#     except Exception as e:
#         print("Image preprocessing error:", e)
#         raise

# # ----------------------------
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

#         interpreter.set_tensor(input_details[0]['index'], image_tensor)
#         interpreter.invoke()

#         plant_pred = interpreter.get_tensor(output_details[0]['index'])
#         disease_pred = interpreter.get_tensor(output_details[1]['index'])
#         severity_pred = interpreter.get_tensor(output_details[2]['index'])

#         plant_index = int(np.argmax(plant_pred))
#         disease_index = int(np.argmax(disease_pred))
#         severity_index = int(np.argmax(severity_pred))

#         print("Prediction:", {
#             "plant": plant_index,
#             "disease": disease_index,
#             "severity": severity_index
#         })

#         return jsonify({
#             "plant": plant_index,
#             "disease": disease_index,
#             "severity": severity_index
#         })

#     except Exception as e:
#         print("Prediction error:", e)
#         return jsonify({'error': 'Error processing image'}), 500

# # ----------------------------
# # Step 5: Run server
# if __name__ == '__main__':
#     app.run(host='0.0.0.0',  port=int(os.environ.get("PORT", 5000)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
import cv2
import io
import os
import gdown

# -----------------------------
# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------
# Step 1: Download TFLite model
TFLITE_MODEL_PATH = "model_quant_updated.tflite"
TFLITE_FILE_ID = "1m9BQV3BHsL1fkzCljOtkr7Y4C08XsUAP"
if not os.path.exists(TFLITE_MODEL_PATH):
    print("Downloading TFLite model...")
    gdown.download(f"https://drive.google.com/uc?id={TFLITE_FILE_ID}", TFLITE_MODEL_PATH, quiet=False)

# -----------------------------
# Step 2: Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Step 3: Download and Load U²-Net (U2NETP) model
from u2net import U2NETP  # Make sure you have u2net.py in the same folder or installed as module

U2NET_MODEL_PATH = "u2netp.pth"
U2NET_FILE_ID = "1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"  # U2NETP model from official repo

if not os.path.exists(U2NET_MODEL_PATH):
    print("Downloading U2NETP model...")
    gdown.download(f"https://drive.google.com/uc?id={U2NET_FILE_ID}", U2NET_MODEL_PATH, quiet=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_u2net = U2NETP(3, 1)
model_u2net.load_state_dict(torch.load(U2NET_MODEL_PATH, map_location=device))
model_u2net.to(device)
model_u2net.eval()

transform_u2net = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Step 4: Background removal
def remove_background_single_image(pil_img):
    original_size = pil_img.size
    input_tensor = transform_u2net(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = model_u2net(input_tensor)
        mask = d1[0][0].cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = cv2.resize(mask, original_size)

    image_np = np.array(pil_img)
    result = image_np * mask[..., None]
    result = result.astype(np.uint8)
    return Image.fromarray(result).convert("RGB")

# -----------------------------
# Step 5: Preprocess image
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = remove_background_single_image(image)  # Apply U²-Net
        image = image.resize((64, 64))  # Match model input
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
        return image_array
    except Exception as e:
        print("Image preprocessing error:", e)
        raise

# -----------------------------
# Step 6: Prediction route
@app.route('/')
def home():
    return 'Server is running!'

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

# -----------------------------
# Step 7: Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
