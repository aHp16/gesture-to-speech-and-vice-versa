# Gesture to Speech: Enabling Communication for the Speech-Impaired

A deep learning-based system that translates hand gestures into audible speech, aiming to bridge communication gaps for individuals with speech impairments.

## Features

- Real-time gesture recognition using webcam input
- Gesture to Speech: Recognizes hand gestures in real-time and converts them into spoken words using text-to-speech synthesis
- Text to Gesture: Converts typed text into corresponding sign language gestures (gesture-based)
- Support for both static (alphabets and numbers) and dynamic gestures
- Intuitive web interface using Flask, HTML/CSS, and JavaScript
- Custom-trained MobileNetV2 CNN and 3D CNN models
- Works offline after initial setup

## Technologies Used

- Python, Flask
- OpenCV for gesture input capture
- TensorFlow/Keras for model training and inference
- Text-to-Speech (text-to-speech synthesis)
- HTML/CSS/JavaScript for frontend
- Apache 2.0 Licensed

## 📁 Project Structure

Gesture-to-Speech/

├── app.py

├── models/

│ ├── mobilenet_model.h5

│ └── isl_3dcnn_model.h5

├── static/

│ ├── videos/

│ └── style.css

├── templates/

│ ├── index.html

│ ├── video.html

│ ├── main.html

│ └── to_gest.html

├── dataset/

│ ├── AlphNum/

│ └── gestures/

├── class_indices.json

└── README.md

## Note on Datasets & Model Files
Due to GitHub's file size limitations and storage constraints, the following assets are not included in this repository:

dataset/ folders (containing image and video gesture data)

static/ videos (the videos displayed for the feature text-to-gesture)

models/isl_3dcnn_model.h5 (the trained 3D CNN model for dynamic gestures)

If you're interested in accessing these resources for academic or research purposes, feel free to reach out — I'd be happy to share them. (You can reach me via the email listed on my GitHub profile)
