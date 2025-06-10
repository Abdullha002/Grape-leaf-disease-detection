import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf



CLASS_NAMES = ['Black_rot', 'Esca_Black_Measles', 'Healthy', 'Leaf_blight' ]

st.title("Grape Leaf Disease Detection")
st.markdown("Upload an image of the grape leaf")

Grape_image = st.file_uploader("Choose an Image...", type = "jpg")
submit = st.button('Predict Diseases')

model = load_model("leaf_disease_coloured.h5")
if submit:
    if Grape_image is not None:
        file_bytes = np.asarray(bytearray(Grape_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels = "BGR")
        st.write(opencv_image.shape)

        opencv_image = cv2.resize(opencv_image, (256, 256))

        opencv_image.shape = (1, 256, 256, 3)

        y_pred = model.predict(opencv_image)
        result_index = np.argmax(y_pred)
        
        # Check if result_index is within the range of CLASS_NAMES
        if result_index < len(CLASS_NAMES):
            result = CLASS_NAMES[result_index]
            st.title("Predicted disease is {}.".format(result))
        else:
            st.error("Prediction index out of range.")
