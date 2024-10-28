# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:17:44 2024

@author: hifzh
"""

import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mode_openess_v1.h5')


sess_options = ort.SessionOptions()
sess = ort.InferenceSession('model_mouth_openness.onnx', sess_options)
# Define a function to preprocess the image
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    cv2.imshow('resized', resized)
    normalized = (resized  / 127.5) - 1.0
    #normalized = resized  / 255
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    reshaped_32 = reshaped.astype(np.float32)
    return reshaped_32

# Start the webcam
stream_url = "http://192.168.151.48/"
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame resolution: {frame_width} x {frame_height}")

cascade_path = "C:/Users/hifzh/final_year/mouth-openness-detection/haarcascade_frontalface_default.xml"
face_casc = cv2.CascadeClassifier(cascade_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original', frame)
    faces = face_casc.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    
    
    for (x, y, w, h) in faces:
        half_y = (y + h // 2) + 10  # Start the y coordinate at the vertical midpoint
        half_h = (h // 2) + 10     # Set the height from the midpoint to the bottom of the face
        
        face_lower_half = frame[half_y:half_y + half_h, x:x + w]
        
        
        processed_frame = preprocess_image(face_lower_half)
        
        input_name = sess.get_inputs()[0].name
        feed = {input_name: processed_frame}

        outputs = sess.run(None, feed)
        
        print(outputs)
        label = "Open" if outputs[0][0] > 0.5 else "Closed"

        cv2.putText(frame, f'Mouth: {label}  Prediction: {outputs[0][0].item():.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mouth Openness Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
