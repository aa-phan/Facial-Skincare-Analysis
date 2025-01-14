from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from inferance import run_model  # Directly import the function

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET'])
def home():
    return "Flask app running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    # Save the uploaded image to the tmp folder
    filename = secure_filename(f"{uuid.uuid4()}_{image_file.filename}")
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    try:
        # Directly call the run_model function
        prediction = run_model(image_path)
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': 'Inference failed', 'details': str(e)}), 500
    finally:
        # Clean up the temporary image
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
