from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, template_folder='template')

# Load the trained model
model = load_model('dog_cat_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        try:
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            print(f"Prediction value: {prediction[0][0]}")
            if prediction[0][0] > 0.5:
                result = 'Dog'
            else:
                result = 'Cat'
            return jsonify({'prediction': result})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
