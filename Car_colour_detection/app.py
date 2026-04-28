import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


def detect_car_color(car_img):
    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)

    # STRICT blue range (no green overlap)
    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = car_img.shape[0] * car_img.shape[1]

    blue_ratio = blue_pixels / total_pixels

    if blue_ratio > 0.08:   
        return "Blue"
    else:
        return "Not Blue"

    

def main():
    st.title("🚦 Car Colour Detection & Traffic Counting System")
    st.write("Upload a traffic image to detect car colours and count cars and people.")

    uploaded_file = st.file_uploader(
        "Upload Traffic Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = model(image_bgr)[0]

        car_count = 0
        people_count = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "car":
                car_count += 1
                car_img = image_bgr[y1:y2, x1:x2]

                if car_img.size == 0:
                    continue

                car_color = detect_car_color(car_img)

                if car_color == "Blue":
                    box_color = (0, 0, 255)  # Red box
                else:
                    box_color = (255, 0, 0)  # Blue box

                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    image_bgr, car_color, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
                )

            elif label == "person":
                people_count += 1
               # cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display counts
        cv2.putText(
            image_bgr, f"Cars: {car_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
        )
        cv2.putText(
            image_bgr, f"People: {people_count}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        final_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        st.image(
            final_image,
            caption="Detection Result",
            width='stretch'
        )
    

if __name__ == "__main__":
    main()    