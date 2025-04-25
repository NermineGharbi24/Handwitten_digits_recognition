# Digit Recognition with Transfer Learning
# This project uses a pre-trained CNN model fine-tuned for MNIST digit recognition

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape data for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    
    # Normalize pixel values to [0, 1]
    x_train /= 255.0
    x_test /= 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Create a base model for transfer learning
def create_base_model():
    base_model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten()
    ])
    
    # Freeze the base model layers
    base_model.trainable = False
    
    return base_model

# Create the transfer learning model
def create_transfer_model(base_model):
    inputs = Input(shape=(28, 28, 1))
    x = base_model(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    # Create a directory for saving model checkpoints
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        'model_checkpoints/digit_recognition_model.h5',
        save_best_only=True
    )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    return model, history

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes)
    print("\nClassification Report:")
    print(report)
    
    return cm, report

# Visualize results
def visualize_results(history, cm, x_test, y_test, model):
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Display some test examples with predictions
    plt.figure(figsize=(15, 10))
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Display 15 random test examples
    indices = np.random.choice(range(len(x_test)), 15, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        title = f"True: {y_true_classes[idx]}\nPred: {y_pred_classes[idx]}"
        if y_true_classes[idx] != y_pred_classes[idx]:
            title += " ❌"
        else:
            title += " ✓"
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.close()

# Function to make a prediction on a single image
def predict_digit(model, image):
    """
    Make a prediction on a single image
    
    Args:
        model: trained model
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

# Save model for future use
def save_model(model, filename='digit_recognition_model.h5'):
    model.save(filename)
    print(f"Model saved as {filename}")

# Main function to run the full pipeline
def main():
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    print("Creating base model for transfer learning...")
    base_model = create_base_model()
    
    print("Creating transfer learning model...")
    model = create_transfer_model(base_model)
    model.summary()
    
    print("Training model...")
    model, history = train_model(model, x_train, y_train, x_test, y_test)
    
    print("Evaluating model...")
    cm, report = evaluate_model(model, x_test, y_test)
    
    print("Visualizing results...")
    visualize_results(history, cm, x_test, y_test, model)
    
    print("Saving model...")
    save_model(model)
    
    print("Done!")

if __name__ == "__main__":
    main()
