# Digit Recognition with Transfer Learning

This project demonstrates how to use transfer learning to recognize handwritten digits from the MNIST dataset. It uses a convolutional neural network (CNN) as a base model and fine-tunes it for digit classification.

## Project Overview

The project implements a deep learning model that can recognize handwritten digits (0-9) with high accuracy. It showcases:

- Transfer learning techniques
- Convolutional Neural Networks (CNNs)
- TensorFlow/Keras implementation
- Model evaluation and visualization
- Saving and loading models for inference

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Project Structure

```
digit-recognition/
├── digit_recognition.py      # Main script with model implementation
├── model_checkpoints/        # Directory for model checkpoints
├── digit_recognition_model.h5  # Saved model file
├── training_curves.png       # Visualization of training metrics
├── confusion_matrix.png      # Confusion matrix visualization
├── prediction_examples.png   # Example predictions
└── README.md                 # This file
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/digit-recognition.git
   cd digit-recognition
   ```

2. Run the main script:
   ```bash
   python digit_recognition.py
   ```

3. The script will:
   - Load and preprocess the MNIST dataset
   - Create and train the model
   - Evaluate model performance
   - Generate visualizations
   - Save the trained model

## Model Architecture

The model uses a transfer learning approach:

1. **Base Model**: A pre-trained CNN with three convolutional layers, each followed by max pooling
2. **Transfer Layer**: The base model is frozen, and a new classification head is added
3. **Classification Head**: Dense layer with dropout for regularization and a softmax output layer

## Results

The model achieves over 99% accuracy on the MNIST test dataset. See the generated visualizations for detailed performance metrics:

- `training_curves.png`: Shows the model's accuracy and loss during training
- `confusion_matrix.png`: Displays classification performance across all digits
- `prediction_examples.png`: Shows example predictions on test images

## Using the Model for Predictions

You can use the saved model to make predictions on new images:

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model('digit_recognition_model.h5')

# Function to predict digit
def predict_digit(image):
    """
    Make a prediction on a single image
    
    Args:
        image: preprocessed image of shape (28, 28, 1)
        
    Returns:
        predicted digit (0-9)
    """
    if image.shape != (28, 28, 1):
        image = image.reshape(1, 28, 28, 1)
    else:
        image = image.reshape(1, 28, 28, 1)
    
    image = image.astype('float32') / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]
```

## Future Improvements

- Implement data augmentation for better generalization
- Try different pre-trained architectures
- Extend to recognize handwritten letters
- Deploy the model as a web service

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MNIST dataset creators
- TensorFlow and Keras documentation
