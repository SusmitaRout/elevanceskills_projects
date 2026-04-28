import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from PIL import Image

# ------------------------------
# Load Models
# ------------------------------
nationality_model = load_model("nationality_model.keras")
age_model = load_model("age_model.keras")
emotion_model = load_model("emotion_model.keras")

nationality_labels = ["United States", "African", "Indian", "Other"]
emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ------------------------------
# Dress Color Detection Function
# ------------------------------
def detect_dress_color(image):

    img = (image * 255).astype("uint8")
    h, w, _ = img.shape

    lower_part = img[int(h*0.55):h, 0:w]
    lower_part = cv2.resize(lower_part, (100,100))

    hsv = cv2.cvtColor(lower_part, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape((-1,3))

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    h_val, s_val, v_val = dominant

        # Black
    if v_val < 50:
        return "Black"

# White
    elif s_val < 40 and v_val > 180:
        return "White"

# Pink (LOW saturation red)
    elif (0 <= h_val < 10 or 160 <= h_val <= 180) and s_val < 120 and v_val > 150:
        return "Pink"

# Red (HIGH saturation)
    elif (0 <= h_val < 10 or 160 <= h_val <= 180):
        return "Red"

# Orange
    elif 10 <= h_val < 25:
        return "Orange"

# Yellow
    elif 25 <= h_val < 35:
        return "Yellow"

# Green
    elif 35 <= h_val < 85:
        return "Green"

# Blue
    elif 85 <= h_val < 125:
        return "Blue"

# Purple
    elif 125 <= h_val < 160:
        return "Purple"

    else:
        return "Other"

# ------------------------------
# Prediction Function
# ------------------------------
def predict_attributes(image):

    results = {}

    # Resize for nationality & age
    img_nat = cv2.resize(image, (96,96)) / 255.0
    img_nat = np.expand_dims(img_nat, axis=0)

    nat_pred = np.argmax(nationality_model.predict(img_nat))
    nationality = nationality_labels[nat_pred]
    results["Nationality"] = nationality

    age = int(age_model.predict(img_nat)[0][0])

    # Emotion prediction
    emo_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    emo_img = cv2.resize(emo_img, (48,48))
    emo_img = emo_img.reshape(1,48,48,1) / 255.0

    emo_pred = np.argmax(emotion_model.predict(emo_img))
    emotion = emotion_names[emo_pred]
    results["Emotion"] = emotion

    dress_color = detect_dress_color(image/255.0)

    # Conditional Logic
    if nationality == "Indian":
        results["Age"] = age
        results["Dress Color"] = dress_color
        results["Emotion"] = emotion

    elif nationality == "United States":
        results["Age"] = age
        results["Emotion"] = emotion

    elif nationality == "African":
        results["Dress Color"] = dress_color
        results["Emotion"] = emotion
    else:
        results["Emotion"] = emotion
    return results

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🌍 Person Attribute Detection System")

st.write("Upload an image to detect nationality, emotion, age and dress color.")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):

        output = predict_attributes(image_np)

        st.subheader("🔍 Prediction Results")

        for key, value in output.items():
            st.write(f"**{key}:** {value}")
