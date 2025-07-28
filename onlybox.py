# Requirements:
# streamlit, opencv-python, pytesseract, pillow

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import os
import cvzone
from ultralytics import YOLO

# Set Tesseract config for Myanmar only
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'
CUSTOM_CONFIG = r'--oem 3 --psm 6 -l mya'

# Class labels for detection
class_labels = {
    0: 'id',
    1: 'three'
}

def preprocess_image(image: Image.Image):
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Denoising with specified strength h=19
    img = cv2.fastNlMeansDenoising(img, None, h=19, templateWindowSize=7, searchWindowSize=21)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(19, 19))
    img = clahe.apply(img)
    
    # Adaptive thresholding with specified parameters
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 20)
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return img

def detect_boxes_and_ocr(image_cv):
    model_path = "v3.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the correct directory.")
        return []
    yolo_model = YOLO(model_path)
    results = yolo_model(image_cv)
    detected = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            confidence = box.conf[0]
            class_name = class_labels.get(cls, "Unknown")
            roi = image_cv[y1:y2, x1:x2]
            # Convert ROI to PIL Image for preprocessing
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            # Apply preprocessing
            roi_processed = preprocess_image(roi_pil)
            # OCR
            os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'
            CUSTOM_CONFIG = r'--oem 3 --psm 6 -l mya'
            text = pytesseract.image_to_string(roi_processed, config=CUSTOM_CONFIG)
            detected.append({
                'class': class_name,
                'confidence': float(confidence),
                'box': (x1, y1, x2, y2),
                'text': text.strip()
            })
    return detected

def main():
    st.title("Myanmar NRC Textbox Detection and OCR")
    st.write("Upload an NRC image. The app will detect textboxes, process each, and extract Myanmar text only.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Convert PIL Image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections = detect_boxes_and_ocr(image_cv)

        # Draw boxes and labels on image for display
        image_disp = image_cv.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class']
            confidence = det['confidence']
            cvzone.putTextRect(image_disp, f'{class_name} ({confidence:.2f})', (x1, y1 - 10), scale=1, thickness=2)
            cvzone.cornerRect(image_disp, (x1, y1, x2 - x1, y2 - y1))
        st.subheader("Detected NRC Textboxes")
        st.image(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Show OCR results for each detected box
        if detections:
            for i, det in enumerate(detections, 1):
                st.markdown(f"**Box {i}: {det['class']} ({det['confidence']:.2f})**")
                # Show the preprocessed image for this box
                x1, y1, x2, y2 = det['box']
                roi = image_cv[y1:y2, x1:x2]
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_processed = preprocess_image(roi_pil)
                st.image(roi_processed, caption=f"Preprocessed Box {i}", use_column_width=True, channels="GRAY")
                st.write(f"Extracted text: {det['text']}")
        else:
            st.write("No NRC textboxes detected.")

if __name__ == "__main__":
    main()
