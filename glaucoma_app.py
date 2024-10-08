import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Function to preprocess and predict the uploaded image
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100, 100), Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, caption='Uploaded Fundus Image', use_column_width=True)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load the pre-trained model
model = tf.keras.models.load_model('my_model2.h5')

# Add a header image (logo)
logo = Image.open('logo.png')  # Replace 'logo.jpg' with your logo file path
st.image(logo, width=200)  # Make the logo bigger

# Custom CSS to make the header bigger and full width
st.markdown("""
    <style>
    .header {
        font-size: 60px;
        font-weight: bold;
        color: #2F4F4F;
        text-align: center;
        margin-bottom: 0px;
    }
    .subheader {
        font-size: 20px;
        text-align: center;
        color: #4F4F4F;
    }
    .line {
        border: 2px solid #2F4F4F;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Page title and description
st.markdown("<div class='header'> Glaucoma Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>This web app uses a deep learning model to detect glaucoma through fundus images of the eye.</div>", unsafe_allow_html=True)
st.markdown("<div class='line'></div>", unsafe_allow_html=True)

# File uploader for fundus image
file = st.file_uploader("Please upload a fundus image (JPG format only)", type=["jpg"])

if file is None:
    st.warning("You haven't uploaded a JPG image file. Please upload one to proceed.")
else:
    # Open and process the image
    imageI = Image.open(file)
    prediction = import_and_predict(imageI, model)
    
    # Show the result
    pred = prediction[0][0]
    if pred > 0.5:
        st.success("## **Prediction:** Your eye is Healthy. Great!! 🎉")
        st.balloons()
    else:
        st.error("## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.")

# Add footer or extra information if needed
st.write("""
    <hr style='border:2px solid #2F4F4F'>

    """, unsafe_allow_html=True)
