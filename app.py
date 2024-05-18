from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('leaf_classifier_model.h5')

def preprocess_image(image):
    # Resize image to match model input size
    image = image.resize((255, 255))
    # Convert to RGB (if the image is in RGBA format)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Convert to numpy array
    image_array = np.asarray(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Expand dimensions to match model input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image file
    file = request.files['image']
    # Read image using PIL
    image = Image.open(file)
    # Preprocess image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    # Convert prediction to string label
    if prediction[0] > 0.5:
        result = 'Withered Leaf'
    else:
        result = 'Fresh Leaf'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)