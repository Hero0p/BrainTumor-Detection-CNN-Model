import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Title
st.title("Brain Tumor Detection from MRI Scan")
st.write("Upload an MRI image of the brain to predict if a tumor is present.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_detection_model.h5")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "Positive" if prediction >= 0.5 else "Negative"
    st.markdown(f"### Prediction: **{result}**")
    st.markdown(f"(Confidence: {prediction:.2f})")
