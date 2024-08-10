import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# Constants
labels = ['Bird', 'Drone']
image_size = 224

# Load Model
model = load_model('BirdvsDrone_vgg16.h5')

# Title
st.title('Birds vs Drones Classification')

# Sidebar
st.sidebar.title('Navigation')
app_mode = st.sidebar.selectbox('Choose the app mode', ['Introduction', 'Predict'])

if app_mode == 'Introduction':
    st.markdown("""
    <div style='text-align: justify;'>
    The Bird vs Drone Classification Web App is designed to address the growing challenge of distinguishing between birds and drones, a critical issue in Unmanned Aerial Systems (UASs) technology. As drones become increasingly common, their interaction with natural environments, especially with birds, has drawn significant attention from researchers. One major concern is the risk of bird strikes, which are among the top causes of drone crashes.

    This web app utilizes advanced machine learning algorithms to classify flying objects in birds and drones from uploaded images. The technology is particularly relevant for protecting natural resources and supporting green initiatives, given the impact of drones on wildlife and airspace safety.
    </div>
    """, unsafe_allow_html=True)

elif app_mode == 'Predict':
    st.write('### Predict a new image')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img = np.array(image)
        if img.shape[-1] == 4:  # Remove alpha channel if present
            img = img[:, :, :3]
        img = cv2.resize(img, (image_size, image_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the class
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_class = labels[predicted_class_index]

        # Debugging: Display the prediction probabilities
        st.write(f"Prediction Probabilities: {predictions}")
        st.write(f"Predicted Class Index: {predicted_class_index}")
        st.write(f"Predicted Class: {predicted_class}")

        st.write(f"### Prediction: {predicted_class}")
st.markdown("""
    <br>
    <center>
        <strong>Developed by Muneeb Ul Hassan</strong>
        <a href="https://www.linkedin.com/in/muneeb-ul-hassan-machine-learning-expert/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:20px;height:20px;margin-left:8px;">
        </a>
    </center>
    """, unsafe_allow_html=True)