from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask import Flask
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "https://bui-tung-hung.github.io/DPL_API/"]}})
app.config['CORS_HEADERS'] = 'Content-Type'



# Load mô hình đã lưu
model = tf.keras.models.load_model('./handwriting_recognition_model.h5')

def preprocess_image(image):
    image = image.resize((28, 28))  # Resize ảnh về kích thước 28x28
    image = image.convert('L')  # Chuyển đổi ảnh sang grayscale
    image = np.array(image).astype("float32") / 255.0  # Chuyển đổi thành numpy array và normalize
    image = image.flatten()  # Flatten từ (28, 28) thành (784,)
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension (1, 784)
    return image

@cross_origin()
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(img)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print(predicted_class)
    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
