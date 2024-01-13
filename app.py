from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = tf.keras.models.load_model('updated_model.h5')

# Class names
label_name = ["CN - Cognitively Normal", "AD - Alzheimer Disease", "EMCI - Early Mild Cognitive Impairment",
              "MCI - Mild Cognitive Impairment", "LMCI - Late Mild Cognitive Impairment"]

def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image.reshape(1, 150, 150, 3)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image_file = request.files['file']
    image = Image.open(image_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    result = {
        "class_name": label_name[predicted_class],
        "confidence": float(prediction[0][predicted_class])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
