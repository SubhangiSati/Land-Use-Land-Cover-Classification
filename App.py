import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Define the classes
CLASSES = ["annual_crop", "forest", "herbaceous_vegetation", "highway",
           "industrial", "pasture", "permanent_crop", "residential",
           "river", "sea_lake"]

# Load the CNN model
model = load_model('cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((64, 64))  # Assuming input size is 224x224
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict class
def predict_class(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class, confidence

# Streamlit app
def main():
    st.title('Land Use Classification')
    st.write('Upload an image for land class prediction')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        predicted_class, confidence = predict_class(image)
        st.write('Prediction:', predicted_class)
        st.write('Confidence:', confidence)

if __name__ == '__main__':
    main()
