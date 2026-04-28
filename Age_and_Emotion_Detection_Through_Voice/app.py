import streamlit as st
import librosa
import numpy as np
import os
import joblib


st.title("🎙️ Age & Emotion Detection Through Voice")



gender_model = joblib.load('gender_model.pkl')
age_model = joblib.load('age_model.pkl')
emotion_model = joblib.load('emotion_model.pkl')

age_encoder = joblib.load('age_encoder.pkl')
emotion_encoder = joblib.load('emotion_encoder.pkl')


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)

    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(y=y, sr=sr).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(y=y, sr=sr)
    )

    features = np.hstack([mfcc, chroma, mel])

    return features

def analyze_voice(audio_path):

    features = extract_features(audio_path).reshape(1, -1)

    # Gender prediction
    gender = gender_model.predict(features)[0]
    if gender == 0:
        return "❌ Upload male voice"

    # Age prediction
    age_pred = age_model.predict(features)[0]
    age_label = age_encoder.inverse_transform([age_pred])[0]

    if age_label not in ['sixties', 'seventies', 'eighties']:
        return f"✅ Male | Age Group: {age_label}"

    # Emotion prediction (senior only)
    emotion_pred = emotion_model.predict(features)[0]
    emotion = emotion_encoder.inverse_transform([emotion_pred])[0]

    return f"🧓 Male | Senior Citizen | Emotion: {emotion}"



st.write("Upload a voice note to analyze gender, age, and emotion")

uploaded_file = st.file_uploader(
    "Upload voice file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.read())
    st.audio("temp_audio")
    result = analyze_voice("temp_audio")
    st.success(result)

