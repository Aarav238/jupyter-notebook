from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(name)
CORS(app)

model = load_model("resnet_model.h5")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    img = cv2.resize(img, (100, 100))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)[0][0]
    result = "Malignant" if prediction > 0.5 else "Benign"

    return jsonify({
        'prediction': result,
        'confidence': f"{prediction * 100:.2f}%"
    })

if name == 'main':
    app.run(debug=True)