import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import load_model
import tempfile

st.title("Senior Citizen Identification System")

# Load Model
model = load_model("age_gender_model.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

uploaded_file = st.file_uploader("Upload Mall Walking Video", type=["mp4", "avi"])

if uploaded_file is not None:
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    data_log = []
    
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128,128))
            face = face / 255.0
            face = np.reshape(face, (1,128,128,3))
            
            age_pred, gender_pred = model.predict(face)
            
            age = int(age_pred[0][0])
            gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
            senior = "Yes" if age > 60 else "No"
            
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            data_log.append([current_time, age, gender, senior])
            
            label = f"{age}, {gender}, Senior: {senior}"
            
            color = (0,255,0) if senior == "No" else (0,0,255)
            
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)
        
        stframe.image(frame, channels="BGR")
    
    cap.release()
    
    df = pd.DataFrame(data_log, columns=[
        "Time of Visit", "Age", "Gender", "Senior Citizen"
    ])

    
    df.to_csv("visit_log.csv", index=False)
    
    st.success("Detection Complete! CSV Saved as visit_log.csv")
    
    st.dataframe(df)
