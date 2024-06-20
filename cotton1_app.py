import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# TensorFlow configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Load the pre-trained model
MODEL_PATH = 'C:/Users/RAM BABU/Downloads/model_inception.h5'
model = load_model(MODEL_PATH)

# Function to make predictions using the loaded model
def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
    
        preds = model.predict(x)
        preds = np.argmax(preds, axis=1)
    
        classes = ["The leaf is diseased cotton leaf","The leaf is diseased cotton plant","The leaf is fresh cotton leaf","The leaf is fresh cotton plant"]
        if preds[0] <len(classes):
            return classes[preds[0]]
        else:
                return "there is no match"
    except Exception as e:
        return f"an error occurred: {e}"

# Streamlit app
st.title("Cotton Leaf Prediction")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image and make predictions
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded file to a temporary location
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Make predictons using the model
        print("Calling model_prediction function..")
        result = model_predict(temp_path, model)
        print("Result from model_predict:",result)
        st.success(result)
    else:
        st.error("please upload an image file.")
else: print("NO image uploaded.")
    
session.close()

       
