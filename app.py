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
    st.warning("style.css file missing! Please ensure style.css is in the same folder.")

st.markdown("<div class='main-title'> AI Face Analyzer </div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload any photo, and our AI will analyze Age, Gender, and Emotion with high accuracy!</div>", unsafe_allow_html=True)

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
        with st.spinner(" AI is detecting facial landmarks & analyzing features... Please wait."):
            
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            try:
                
                analysis = DeepFace.analyze(
                    img_path=opencv_image, 
                    actions=['age', 'gender', 'emotion'], 
                    enforce_detection=True,
                    detector_backend='retinaface',
                    align=True
                )
                
                result = analysis[0] if isinstance(analysis, list) else analysis
                
                
                raw_age = int(result['age'])
                age_min = max(0, raw_age - 2)
                age_max = raw_age + 2
                
                gender = result['dominant_gender']
                gender_confidence = result['gender'][gender]
                
                emotion = result['dominant_emotion']
                emotion_confidence = result['emotion'][emotion]
                
                st.success("Analysis Complete! ")
                
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(
                        label="Predicted Age", 
                        value=f"~{raw_age} yrs", 
                        delta=f"Range: {age_min}-{age_max} yrs",
                        delta_color="normal"
                    )
                with c2:
                    st.metric(
                        label="Gender", 
                        value=str(gender).capitalize(), 
                        delta=f"{gender_confidence:.1f}% Confidence",
                        delta_color="normal"
                    )
                with c3:
                    st.metric(
                        label="Mood / Emotion", 
                        value=str(emotion).capitalize(), 
                        delta=f"{emotion_confidence:.1f}% Match",
                        delta_color="normal"
                    )
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
                
                with st.expander(" View Detailed Emotion Breakdown"):
                    st.write("Distribution of facial expression probability:")
                    sorted_emotions = sorted(result['emotion'].items(), key=lambda x: x[1], reverse=True)
                    for emo_name, score in sorted_emotions:
                        st.write(f"**{emo_name.capitalize()}**: {score:.1f}%")
                        st.progress(min(int(score), 100))
                
            except Exception as e:
                st.error("Oops! AI could not detect a clear face in the photo. Please upload a well-lit image facing the camera.")
                st.caption(f"Technical Error Details: {str(e)}")
    else:
        st.info("👈 Please upload a photo from the left section to see the results here.")