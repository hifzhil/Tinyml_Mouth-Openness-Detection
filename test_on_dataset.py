import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mode_openess_v1.h5')

# Define the path to the dataset directory
dataset_dir = 'C:/Users/hifzh/computer-vision/Mouth-Openess-Detection/dataset/train/Yawn'

# Define a function to preprocess the image
def preprocess_image(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = (resized  / 127.5) - 1.0
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

# Function to recursively process images in the directory and count "open" predictions
def process_directory(directory):
    open_count = 0
    total_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                processed_frame = preprocess_image(file_path)
                prediction = model.predict(processed_frame)
                label = "Open" if prediction[0] > 0.5 else "Closed"
                print(f'File: {file_path}, Label: {label}, Confidence: {prediction[0][0]:.2f}')
                if label == "Open":
                    open_count += 1
                total_count += 1
    return open_count, total_count

# Run the process_directory function on the dataset directory and get the counts
open_count, total_count = process_directory(dataset_dir)

# Print the counts
print(f'Total images processed: {total_count}')
print(f'Number of "Open" predictions: {open_count}')
print(f'Number of "Closed" predictions: {total_count - open_count}')
