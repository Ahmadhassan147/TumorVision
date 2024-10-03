import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Load pre-trained model
model = load_model('braintumormodel.h5')

# Title of the app
st.title('TumorVision: Brain Tumor Classification')

# Upload image function
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded MRI Image.', use_column_width=True)
    
    # Read the image and preprocess it
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))  # Resize to match model input
    img_array = np.array(img)
    
    # Model prediction
    if st.button('Scan Brain'):
        img_array = img_array.reshape(1, 150, 150, 3)  # Reshape for model
        prediction = model.predict(img_array)
        
        labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        result = labels[np.argmax(prediction)]
        
        st.write(f"The model predicts: **{result}**")

        # Plot the prediction probabilities
        st.subheader("Prediction Probabilities")
        st.bar_chart(prediction[0])
