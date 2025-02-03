from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('ai_image_detector.h5')

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img_resized, axis=0)

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint to classify uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_path = f'./{file.filename}'
    file.save(image_path)

    try:
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        result = "AI-generated" if prediction > 0.5 else "Real"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
