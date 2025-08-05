import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Set Streamlit page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

# Load SVM model
with open("CatandDogs.pkl", "rb") as f:
    CatandDogs = pickle.load(f)

# Load MobileNetV2 for feature extraction
cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Custom header
st.markdown(
    """
    <h1 style='text-align: center;'>🐾 Cat vs Dog Classifier</h1>
    <p style='text-align: center; font-size:18px;'>Upload an image of a <b>cat</b> or <b>dog</b> to find out what it is!</p>
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analyzing image..."):
            # Read and preprocess image
            img = Image.open(uploaded_file).resize((160, 160))
            img_array = keras_image.img_to_array(img)
            img_array = preprocess_input(img_array)
            features = cnn_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            features_flat = features.flatten().reshape(1, -1)

            # Predict
            prediction = CatandDogs.predict(features_flat)[0]
            confidence = CatandDogs.predict_proba(features_flat)[0][prediction]

            # Results
            label = "🐱 Cat" if prediction == 0 else "🐶 Dog"
            color = "green" if prediction == 1 else "blue"
            st.markdown(
                f"<h3 style='color:{color};'>Prediction: {label}</h3>"
                f"<p style='font-size:18px;'>Confidence: <b>{confidence * 100:.2f}%</b></p>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.info("Tip: Use clear images with the animal centered for best results!")
