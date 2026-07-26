import os
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="AI Face Analyzer", page_icon="👤", layout="centered")

st.title("👤 AI Face Analyzer")
st.write("Upload a photo to analyze age, gender, emotion, and ethnicity.")

uploaded_file = st.file_uploader("Choose a clear portrait photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", use_column_width=True)

    temp_path = "temp_input_image.jpg"
    image.convert("RGB").save(temp_path)

    if st.button("🔍 Analyze Face", type="primary"):
        with st.spinner("Analyzing face features... Please wait."):
            try:
                results = DeepFace.analyze(
                    img_path=temp_path,
                    actions=['age', 'gender', 'race', 'emotion'],
                    detector_backend='mtcnn',
                    enforce_detection=False
                )

                st.success("Analysis Complete!")

                analysis = results[0] if isinstance(results, list) else results

                st.markdown("---")
                st.subheader("📊 Analysis Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Estimated Age", f"{analysis.get('age', 'N/A')} years")

                    gender_data = analysis.get('gender', {})
                    if isinstance(gender_data, dict):
                        dominant_gender = max(gender_data, key=gender_data.get)
                    else:
                        dominant_gender = str(gender_data)
                    st.metric("Gender", dominant_gender.capitalize())

                with col2:
                    st.metric("Dominant Emotion", str(analysis.get('dominant_emotion', 'N/A')).capitalize())
                    st.metric("Dominant Race/Ethnicity", str(analysis.get('dominant_race', 'N/A')).title())

                st.markdown("---")
                st.subheader("🎭 Emotion Breakdown")
                emotions = analysis.get('emotion', {})
                if emotions:
                    for emotion_name, confidence in emotions.items():
                        st.write(f"**{emotion_name.capitalize()}**: {confidence:.1f}%")
                        st.progress(min(float(confidence) / 100.0, 1.0))

            except Exception as e:
                st.error(f"Error during analysis: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)