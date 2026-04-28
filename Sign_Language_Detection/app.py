import streamlit as st
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import pytz
import tensorflow as tf


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()


def is_allowed_time():
    ist = pytz.timezone("Asia/Kolkata")
    current_hour = datetime.now(ist).hour
    return 18 <= current_hour <= 22

st.title("🤟 Sign Language Detection System")

ist = pytz.timezone("Asia/Kolkata")
st.info(f"🕒 Current IST Time: {datetime.now(ist).strftime('%I:%M %p')}")


if not is_allowed_time():
    st.error("⛔ System available only from 6 PM to 10 PM")
    st.stop()


option = st.selectbox(
    "Select Input Method",
    ("Upload Image", "Live Webcam")
)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=300)

        img = image.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)

        st.success(f"✅ Predicted Sign Label: {predicted_label}")

elif option == "Live Webcam":
    st.warning("Press 'Q' to stop webcam")

    run = st.checkbox("Start Webcam")
    if run:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (28, 28))
            img = img / 255.0
            img = img.reshape(1, 28, 28, 1)

            pred = model.predict(img)
            label = np.argmax(pred)

            cv2.putText(frame, f"Prediction: {label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()