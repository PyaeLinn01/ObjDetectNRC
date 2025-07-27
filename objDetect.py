import streamlit as st
import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
import os

# Global model variable
YOLO_MODEL = None

# Class labels for detection
class_labels = {
    0: 'id',
    1: 'three'
}

# Function to get the YOLO model
def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        model_path = "v3.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure it is in the correct directory.")
            return None
        YOLO_MODEL = YOLO(model_path)
    return YOLO_MODEL

# Function to process image with YOLO model
def process_frame_with_results(cv_image):
    yolo_model = get_yolo_model()
    if yolo_model is None:
        return cv_image, []

    results = yolo_model(cv_image)
    detected_classes = set()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            confidence = box.conf[0]
            class_name = class_labels.get(cls, "Unknown")

            cvzone.putTextRect(cv_image, f'{class_name} ({confidence:.2f})', (x1, y1 - 10), scale=1, thickness=2)
            cvzone.cornerRect(cv_image, (x1, y1, x2 - x1, y2 - y1))

            detected_classes.add(class_name)
    
    return cv_image, list(detected_classes)

# Streamlit Application
st.title("NRC textbox detection")

uploaded_file = st.file_uploader("Upload an image to detect NRC textboxes", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # Load image and convert to RGB format
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display uploaded image
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        st.write("Processing image for textboxes...")
        processed_image, detected_classes = process_frame_with_results(image)

        # Convert processed image back to RGB for display
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, caption="Detection Results", use_container_width=True)

        if detected_classes:
            st.subheader("Detected NRC Textboxes:")
            for detected_class in detected_classes:
                st.write(f"Detected: {detected_class}")
        else:
            st.write("No NRC textboxes detected.")
    except Exception as e:
        st.error(f"An error occurred: {e}")