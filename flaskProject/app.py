from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io
import sys

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

app = Flask(__name__)

model = MobileNetV2(weights='imagenet')

def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        prepared_image = prepare_image(image)

        predictions = model.predict(prepared_image)
        results = decode_predictions(predictions, top=3)[0]

        return jsonify({"predictions": [{"label": result[1], "probability": float(result[2])} for result in results]})

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
