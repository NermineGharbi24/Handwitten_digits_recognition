# inference.py
# Script to use the trained model for inference on new images

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import sys
import os

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for digit recognition
    
    Args:
        image_path: path to the image file
        
    Returns:
        preprocessed image
    """
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Read image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Invert if needed (assuming white digit on black background)
    if np.mean(img) > 127:
        img = 255 - img
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    
    return img

def predict_digit(model, image):
    """
    Predict the digit in the given image
    
    Args:
        model: loaded model
        image: preprocessed image
        
    Returns:
        predicted digit (0-9)
    """
    # Make prediction
    prediction = model.predict(image)
    digit = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    return digit, confidence

def display_prediction(image, digit, confidence):
    """
    Display the image with the prediction
    
    Args:
        image: preprocessed image
        digit: predicted digit
        confidence: prediction confidence
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    # Check if model exists
    model_path = 'digit_recognition_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = load_and_preprocess_image(image_path)
    
    if image is None:
        return
    
    # Make prediction
    print("Making prediction...")
    digit, confidence = predict_digit(model, image)
    
    # Print result
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the image with prediction
    display_prediction(image, digit, confidence)

if __name__ == "__main__":
    main()
