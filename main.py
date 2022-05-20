import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image,ImageEnhance
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("new_model.h5")
image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

st.title('Number Plate Detection  App')
st.text("***Road Safety***")


def main():
	"""Number Plate Detection App"""

activities = ["Detection","About"]
choice = st.sidebar.selectbox("Select Activty",activities)

if choice == 'Detection':
    st.subheader("Number Plate Detection")

        
if image_file is not None:
    our_image =  np.asarray(bytearray(image_file.read()), dtype=np.uint8)

    opencv_image = cv2.imdecode(our_image, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    st.image(opencv_image, channels="RGB")
				  
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]
            
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Nuumber Plate is {}".format(image_file[prediction]))
        

elif choice == 'About':
        
    st.subheader("About Number Plate Detection App")
	


# if _name_ == "_main_":
#     main()
