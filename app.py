import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="AI Face Analyzer", page_icon="🧑‍🦱", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass

st.markdown("<div class='main-title'> AI Face Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload any photo, and our AI will analyze the Age, Gender, and Emotion!</div>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader(" Upload Section")
    uploaded_file = st.file_uploader("Upload your photo here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Photo", use_container_width=True)

with right_col:
    st.subheader(" Analysis Output")
    
    if uploaded_file is not None:
        with st.spinner("🔍 AI is analyzing the photo... Please wait."):
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            try:
                analysis = DeepFace.analyze(
                    img_path=opencv_image, 
                    actions=['age', 'gender', 'emotion'], 
                    enforce_detection=True,
                    detector_backend='retinaface'
                )
                
                result = analysis[0] if isinstance(analysis, list) else analysis
                
                age = result['age']
                gender = result['dominant_gender']
                emotion = result['dominant_emotion']
                
                st.success("Analysis Complete! ")
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(label="Age", value=f"~{int(age)} yrs")
                with c2:
                    st.metric(label="Gender", value=str(gender).capitalize())
                with c3:
                    st.metric(label="Mood / Emotion", value=str(emotion).capitalize())
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Oops! AI could not detect a clear face in the photo. Please upload a clearer image.")
                st.warning(f"Technical Details: {str(e)}")
    else:
        st.info("👈 Please upload a photo from the left section to see the results here.")