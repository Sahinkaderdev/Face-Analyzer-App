import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="AI Face Analyzer", page_icon="🧑‍🦱", layout="wide")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

st.markdown("<h1 style='text-align: center;'>🧑‍🦱 AI Face Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a photo to analyze Age, Gender, and Emotion!</p>", unsafe_allow_html=True)
st.divider()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📤 Upload Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with right_col:
    st.subheader("📊 Analysis Result")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing Face... Please wait."):
            try:
                img_array = np.array(image.convert('RGB'))
                opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                results = DeepFace.analyze(
                    img_path=opencv_image, 
                    actions=['age', 'gender', 'emotion'], 
                    enforce_detection=False
                )
                
                result = results[0] if isinstance(results, list) else results
                
                age = result.get('age', 'N/A')
                gender = result.get('dominant_gender', 'N/A')
                emotion = result.get('dominant_emotion', 'N/A')
                
                st.success("Analysis Completed Successfully! 🎉")
                
                res_c1, res_c2, res_c3 = st.columns(3)
                with res_c1:
                    st.metric(label="Estimated Age", value=f"~{int(age)} yrs" if isinstance(age, (int, float)) else str(age))
                with res_c2:
                    st.metric(label="Gender", value=str(gender).capitalize())
                with res_c3:
                    st.metric(label="Emotion", value=str(emotion).capitalize())
                    
            except Exception as e:
                st.error("Could not analyze the image properly. Please try uploading a clearer face image.")
                st.caption(f"Error details: {str(e)}")
    else:
        st.info("Please upload a photo from the left panel to get analysis results.")