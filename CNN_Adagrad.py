import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the path to the dataset directory
dataset_path = "D:\\Research Paper Codes\\EuroSAT dataset\\EuroSAT"

# Define the list of land use classes
land_use_classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                    "Industrial", "Pasture", "PermanentCrop", "Residential",
                    "River", "SeaLake"]

# Load and preprocess the dataset
def load_dataset():
    images = []
    labels = []
    for land_use in land_use_classes:
        class_path = os.path.join(dataset_path, land_use)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = imread(image_path)
            image = resize(image, (64, 64), mode='reflect', anti_aliasing=True)
            images.append(image)
            labels.append(land_use_classes.index(land_use))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load the dataset
images, labels = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation for the CNN model
datagen = ImageDataGenerator(
    rotation_range=-+20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Verify GPU availability
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if not physical_devices:
    print("GPU is not available. Make sure you have installed the necessary CUDA libraries.")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the CNN model with hyperparameter tuning
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(land_use_classes), activation='softmax')
])

# Compile the model with Adagrad optimizer
adagrad_optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)  # You can adjust the learning rate as needed
model.compile(optimizer=adagrad_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Create a log directory for TensorBoard
log_dir = "logs/cnn_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with data augmentation, TensorBoard callback, and MetricsCallback
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[tensorboard_callback])

# Function to calculate precision, recall, and F1-score at each epoch
def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
_, cnn_accuracy = model.evaluate(X_test, y_test)
print("CNN Accuracy:", cnn_accuracy)

# Calculate precision, recall, and F1-score
precision, recall, f1 = get_metrics(y_test, y_pred_classes)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Lists to store precision, recall, and F1-score at each epoch
precision_list = []
recall_list = []
f1_list = []

# Callback to calculate precision, recall, and F1-score at each epoch
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        precision, recall, f1 = get_metrics(y_test, y_pred)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Create a log directory for TensorBoard
log_dir = "logs/cnn_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Save the CNN model
model.save("cnn_model.h5")

# Plot the epoch-wise accuracy, loss, precision, recall, and F1-score
plt.figure(figsize=(18, 12))

# Accuracy
plt.subplot(2, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Model Accuracy')
plt.legend()

# Loss
plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Model Loss')
plt.legend()

# Precision
plt.subplot(2, 3, 3)
plt.plot(precision_list, label='Precision', color='green')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend()

# Recall
plt.subplot(2, 3, 4)
plt.plot(recall_list, label='Recall', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend()

# F1-score
plt.subplot(2, 3, 5)
plt.plot(f1_list, label='F1-Score', color='red')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('F1-Score')
plt.legend()

plt.tight_layout()
plt.show()
