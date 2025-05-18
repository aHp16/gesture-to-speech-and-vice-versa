# Gesture to Speech: Enabling Communication for the Speech-Impaired

A deep learning-based system that translates hand gestures into audible speech, aiming to bridge communication gaps for individuals with speech impairments.

## Features

- Real-time gesture recognition using webcam input
- Speech output using text-to-speech synthesis
- Support for both alphabets and dynamic gestures
- Intuitive web interface using Flask, HTML/CSS, and JavaScript
- Custom-trained CNN and 3D CNN models
- Works offline after initial setup

## Technologies Used

- Python, Flask
- OpenCV for gesture input capture
- TensorFlow/Keras for model training and inference
- Text-to-Speech (pyttsx3 or gTTS)
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
