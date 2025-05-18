from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
import cv2
import json

app = Flask(__name__)


alphabet_model = tf.keras.models.load_model('mobilenet_model.h5')

# Load gesture model
gesture_model = tf.keras.models.load_model('isl_model_3dcnn.h5')

# Load class index mapping for gestures
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

# Fallback class names (for alphabet model)
class_names = getattr(alphabet_model, 'class_names', None)
if class_names:
    class_names = {v: k for k, v in class_names.items()}
else:
    class_names = sorted([chr(i) for i in range(65, 91)] + [str(i) for i in range(10)])


#Utilities
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = img.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


#Routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/alphabet')
def alphabet_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_alphabet():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    processed = preprocess_image(image_data)
    prediction = alphabet_model.predict(processed)[0]
    pred_label = class_names[np.argmax(prediction)]
    return jsonify({'prediction': pred_label})

@app.route('/gesture')
def gesture_page():
    return render_template('video.html')

@app.route('/predict_gesture', methods=['POST'])
def predict_gesture():
    data = request.get_json()
    frames = data['frames']

    video_frames = []
    for frame_data in frames:
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        video_frames.append(img)

    if len(video_frames) != 16:
        return jsonify({'prediction': 'Error: 16 frames required'})

    video_input = np.array(video_frames).reshape(1, 16, 64, 64, 3)
    preds = gesture_model.predict(video_input)
    pred_index = int(np.argmax(preds))
    pred_label = idx_to_class[pred_index]

    return jsonify({'prediction': pred_label})

@app.route('/text-to-gesture', methods=['GET', 'POST'])
def text_to_gesture():
    gesture_classes = ['hello', 'thankyou', 'yes', 'no', 'good', 'welcome',
                       'house', 'morning', 'work', 'nice', 'bye']

    video_list = []
    missing_words = []
    input_text = ""

    if request.method == 'POST':
        input_text = request.form['text_input'].lower().strip()
        words = input_text.split()

        for word in words:
            if word in gesture_classes:
                video_list.append(f'/static/videos/{word}.mp4')
            else:
                missing_words.append(word)

    error = None
    if missing_words:
        error = f"No gesture video found for: {', '.join(missing_words)}"

    return render_template('to_gest.html',
                           video_list=video_list,
                           input_text=input_text,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
