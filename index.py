import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import tensorflow as tf
from flask_cors import CORS

# Path to your model
model_path = '/Users/shweta/cropfitt/crop_disease_model.tflite'

# Load TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model: {e}")
    interpreter = None

# Load class indices (assuming you have this file)
class_indices_file = 'class_indices.json'

if not os.path.exists(class_indices_file):
    print(f"Class indices file '{class_indices_file}' not found.")
    class_indices = {}
else:
    with open(class_indices_file, 'r') as f:
        class_indices = json.load(f)

# Reverse class indices to get labels
class_labels = {v: k for k, v in class_indices.items()}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the index route as an API response
@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to the Crop Disease Detection API!",
        "endpoints": {
            "/predict": "POST method - Upload an image to get predictions"
        }
    })

# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, or JPEG are allowed.'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))  # Resize the image based on model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image to [0, 1]

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Run the interpreter
        interpreter.invoke()

        # Get the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)
        predicted_label = class_labels.get(predicted_class[0], 'Unknown Class')

        # Define the symptoms data
        symptoms_data = {
            "Pepper__bell___Bacterial_spot": [
                "Dark, water-soaked spots on leaves.",
                "Yellowing of leaf margins.",
                "Small, dark, sunken lesions on fruits.",
                "Defoliation and reduced yield.",
                "Wilting and stunted growth."
            ],
            "Pepper__bell___healthy": [
                "No visible symptoms.",
                "Bright green leaves.",
                "Healthy fruit development.",
                "Sturdy stems.",
                "Normal growth patterns."
            ],
            "Potato___Early_blight": [
                "Dark brown spots on older leaves.",
                "Yellowing of leaf margins.",
                "Defoliation.",
                "Reduced tuber quality.",
                "Stunted plant growth."
            ],
            "Potato___Late_blight": [
                "Water-soaked spots on leaves.",
                "White mold on the undersides of leaves.",
                "Rapid wilting of plants.",
                "Dark, greasy spots on tubers.",
                "Rotting of tubers in storage."
            ],
            "Potato___healthy": [
                "No visible symptoms.",
                "Healthy, green leaves.",
                "Robust tuber development.",
                "Strong stems.",
                "Normal growth patterns."
            ],
            "Tomato_Bacterial_spot": [
                "Dark, water-soaked lesions on leaves.",
                "Yellowing of leaf edges.",
                "Small, dark spots on fruits.",
                "Leaf curling and wilting.",
                "Reduced fruit quality."
            ],
            "Tomato_Early_blight": [
                "Dark, concentric rings on leaves.",
                "Yellowing and dropping of lower leaves.",
                "Reduced fruit yield.",
                "Dark lesions on stems.",
                "Stunted plant growth."
            ],
            "Tomato_Late_blight": [
                "Large, irregularly shaped water-soaked spots.",
                "White mold on the underside of leaves.",
                "Brown lesions on stems.",
                "Rapid wilting of plants.",
                "Tubers rot in the ground."
            ],
            "Tomato_Leaf_Mold": [
                "Yellowing of leaves.",
                "Fuzzy greenish-gray mold on the underside.",
                "Leaf curling.",
                "Defoliation.",
                "Reduced fruit quality."
            ],
            "Tomato_Septoria_leaf_spot": [
                "Small, round spots with dark borders.",
                "Yellowing leaves.",
                "Defoliation.",
                "Reduced yield.",
                "Dark, sunken spots on stems."
            ],
            "Tomato_Spider_mites_Two_spotted_spider_mite": [
                "Fine webbing on leaves.",
                "Yellowing and stippling of leaves.",
                "Leaf drop.",
                "Stunted growth.",
                "Brown, crispy leaves."
            ],
            "Tomato__Target_Spot": [
                "Dark, concentric ring spots on leaves.",
                "Leaf drop.",
                "Reduced yield.",
                "Spots may appear on fruits.",
                "Stunted growth."
            ],
            "Tomato__Tomato_YellowLeaf__Curl_Virus": [
                "Yellowing of leaves.",
                "Curling and distortion of leaves.",
                "Stunted growth.",
                "Reduced fruit set.",
                "Plant wilting."
            ],
            "Tomato__Tomato_mosaic_virus": [
                "Mosaic patterns on leaves.",
                "Stunted growth.",
                "Leaf curling.",
                "Deformed fruits.",
                "Reduced yield."
            ],
            "Tomato_healthy": [
                "No visible symptoms.",
                "Healthy green leaves.",
                "Robust fruit development.",
                "Strong stems.",
                "Normal growth patterns."
            ]
        }


        # Get symptoms for the predicted label
        symptoms = symptoms_data.get(predicted_label, ["No symptoms available"])

        # Return the prediction and symptoms as JSON
        return jsonify({
            'prediction': predicted_label,
            'symptoms': symptoms
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    from waitress import serve
    print("Server is live at http://127.0.0.1:5001")
    serve(app, host="0.0.0.0", port=5001)
