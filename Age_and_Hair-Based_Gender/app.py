import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model



st.set_page_config(
    page_title="Age & Hair based Gender Detector",
    page_icon="🧑‍🤝‍🧑",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.result-text {
    font-size: 1.3rem;
    font-weight: 500;
    padding: 0.75rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}

.image-container {
    margin-bottom: 2rem;
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: rgba(237, 242, 247, 0.5);
}

.app-footer {
    text-align: center;
    margin-top: 2rem;
    opacity: 0.7;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton > button:hover {
    background-color: #1E40AF;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_age_gender_model():
    try:
        model = load_model("best_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(uploaded_image):
    img = uploaded_image.convert("L")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_hair_length_gui(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected PIL Image")

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "Short"

    x, y, w, h = faces[0]
    img_h, img_w = img.shape[:2]

    vertical_long = (y + int(1.6 * h)) < img_h

    left_space = x
    right_space = img_w - (x + w)
    horizontal_long = left_space > 0.15 * w or right_space > 0.15 * w

    cheek = gray[y+int(0.3*h):y+int(0.7*h), x:x+w]
    texture_long = np.var(cheek) > 150

    return "Long" if (vertical_long or horizontal_long or texture_long) else "Short"

def predict_age_gender_with_logic(model, image):
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)

    pred_age = int(np.round(predictions[1][0][0]))
    gender_prob = predictions[0][0][0]
    model_gender = "Female" if gender_prob > 0.5 else "Male"

    # ✅ GUI-safe hair detection
    hair = detect_hair_length_gui(image)

    if 20 <= pred_age <= 30:
        final_gender = "Female" if hair == "Long" else "Male"
    else:
        final_gender = model_gender

    return pred_age, model_gender, hair, final_gender


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def main():

    # -------------------------------
    # App Header
    # -------------------------------
    st.markdown(
        "<div class='main-header'>Age & Hair based Gender Detector</div>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # Load Model
    # -------------------------------
    with st.spinner("Loading model... Please wait"):
        model = load_age_gender_model()

    if model is None:
        st.error("Model could not be loaded. Please check model path.")
        return

    # -------------------------------
    # Upload Images Section
    # -------------------------------
    st.markdown(
        "<div class='sub-header'>Upload Images</div>",
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    # -------------------------------
    # Detect Button Logic
    # -------------------------------
    detect_clicked = st.button("Detect Age & Gender", key="detect_btn")

    if detect_clicked:
        if not uploaded_files:
            st.info("Please upload one or more images first.")
        else:
            with st.spinner("Analyzing images..."):
                for i, uploaded_file in enumerate(uploaded_files):

                    st.markdown("<hr>", unsafe_allow_html=True)

                    image = Image.open(uploaded_file)

                    col1, col2 = st.columns(2)

                    # ---- Display Image ----
                    with col1:
                        st.image(
                            image,
                            caption=f"Image {i + 1}: {uploaded_file.name}",
                            width=300
                        )

                    # ---- Prediction & Results ----
                    with col2:
                        age, model_gender, hair, final_gender = \
                            predict_age_gender_with_logic(model, image)

                        st.markdown(
                            "<div class='sub-header'>Results</div>",
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"""
                            <div class='result-text' style='background-color:#DBEAFE;'>
                                Predicted Age: <b>{age}</b> years
                            </div>

                            <div class='result-text' style='background-color:#E0E7FF;'>
                                Model Gender: <b>{model_gender}</b>
                            </div>

                            <div class='result-text' style='background-color:#DCFCE7;'>
                                Hair Length: <b>{hair}</b>
                            </div>

                            <div class='result-text' style='background-color:#FEF3C7;'>
                                <b>FINAL Gender Output: {final_gender}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
if __name__ == "__main__":
    main()