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
from keras import mixed_precision

# Define the path to the dataset directory
dataset_path = "D:\\Research Paper Codes\\EuroSAT Dataset\\EuroSAT"

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

# Split the dataset into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

# Data augmentation for the RNN model
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# Reshape data for RNN
X_train = X_train.reshape(-1, 64, 64, 3)  # Reshape to (samples, time steps, features)
X_val = X_val.reshape(-1, 64, 64, 3)
X_test = X_test.reshape(-1, 64, 64, 3)

# Define the RNN model with hyperparameter tuning
inputs = tf.keras.layers.Input(shape=(64, 64, 3))
x = tf.keras.layers.Reshape((64, 64 * 3,))(inputs)
x = tf.keras.layers.LSTM(64)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(land_use_classes), activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Create a log directory for TensorBoard
log_dir = "logs/rnn_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Function to calculate precision, recall, and F1-score at each epoch
def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

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

metrics_callback = MetricsCallback()

# Enable mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Train the model with data augmentation, TensorBoard callback, and MetricsCallback
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard_callback, metrics_callback])

# Create and compile your model as usual
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Calculate accuracy
_, rnn_accuracy = model.evaluate(X_test, y_test)
print("RNN Accuracy(Using Adam Optimizer):", rnn_accuracy)

# Calculate precision, recall, and F1-score
precision, recall, f1 = get_metrics(y_test, y_pred_classes)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)

# Create a log directory for TensorBoard
log_dir = "logs/rnn_model_adam/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Save the RNN model
model.save("rnn_model_adam.h5")

# Plot the epoch-wise accuracy, loss, precision, recall, and F1-score
plt.figure(figsize=(18, 12))

# Accuracy
plt.subplot(2, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('RNN Model Accuracy')
plt.legend()

# Loss
plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN Model Loss')
plt

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