import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

st.set_page_config(
    page_title="Face Metrics AI", 
    page_icon="🧑‍🦱", 
    layout="wide"
)

def inject_custom_css(css_file):
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Could not find '{css_file}'. Please check your directory.")

inject_custom_css("style.css")


st.markdown("""
    <div class="app-header">
        <h1 class="main-title"> AI Face Analyzer</h1>
        <p class="sub-title">Upload any photo, and our AI will analyze Age, Gender, and Emotion with high accuracy!</p>
    </div>
""", unsafe_allow_html=True)


upload_col, result_col = st.columns([1, 1], gap="large")

with upload_col:
    st.subheader(" Upload Section")
    uploaded_file = st.file_uploader("Choose a photo (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="Uploaded Input", use_container_width=True)

with result_col:
    st.subheader(" Analysis Output")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing facial features and emotion geometry..."):
            
            img_bytes = np.array(user_image)
            cv_img = cv2.cvtColor(img_bytes, cv2.COLOR_RGB2BGR)
            
            try:
                results = DeepFace.analyze(
                    img_path=cv_img,
                    actions=['age', 'gender', 'emotion'],
                    detector_backend='retinaface',
                    align=True,
                    enforce_detection=True
                )
                
 
                face_data = results[0] if isinstance(results, list) else results
                
                age = int(face_data['age'])
                gender = face_data['dominant_gender']
                gender_acc = face_data['gender'][gender]
                
                emotion = face_data['dominant_emotion']
                emotion_acc = face_data['emotion'][emotion]
                
                st.success("Analysis Complete!")
                
                st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
                
                col_age, col_gender, col_mood = st.columns(3)
                
                with col_age:
                    st.metric(
                        label="Predicted Age", 
                        value=f"~{age} yrs", 
                        delta=f"Est: {max(0, age-2)}-{age+2} yrs"
                    )
                
                with col_gender:
                    st.metric(
                        label="Gender", 
                        value=str(gender).capitalize(), 
                        delta=f"{gender_acc:.1f}% Confidence"
                    )
                
                with col_mood:
                    st.metric(
                        label="Primary Mood", 
                        value=str(emotion).capitalize(), 
                        delta=f"{emotion_acc:.1f}% Match"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander(" Emotion Distribution"):
                    emotions_sorted = sorted(face_data['emotion'].items(), key=lambda x: x[1], reverse=True)
                    for emo_name, score in emotions_sorted:
                        st.write(f"**{emo_name.capitalize()}**: {score:.1f}%")
                        st.progress(min(int(score), 100))

            except Exception as err:
                st.error("Could not detect a clear face. Please upload a clear, front-facing portrait.")
                st.caption(f"Debug Info: {str(err)}")

    else:
        st.info("👈 Upload an image on the left side to get real-time facial analytics.")