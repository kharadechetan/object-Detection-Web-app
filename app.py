import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# Set the event loop policy for Windows (Fixes asyncio error)
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load the trained YOLO model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI
st.title("🔍 Object Detection Web App")
st.write("Upload an image to detect objects using the trained YOLOv10 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Perform inference
    results = model.predict(image_np)

    # Prepare detection results
    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(box.conf[0].item(), 2)
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            # Append detection results
            detected_objects.append([class_name, conf, (x1, y1, x2, y2)])

            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{class_name}: {conf}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed image
    st.image(image_np, caption="Detected Objects", use_container_width=True)

    # Show detection results in a table
    if detected_objects:
        df = pd.DataFrame(detected_objects, columns=["Class", "Confidence", "Bounding Box"])
        st.write("### Detection Results")
        st.dataframe(df)

    # Precision-Recall Table (Update manually from training logs)
    st.write("### Precision & Recall Metrics")
    pr_data = {
        "Class": ["RBC", "WBC", "Platelet"],  # Replace with actual class names
        "Precision": [0.85, 0.90, 0.78],  # Replace with actual precision values
        "Recall": [0.80, 0.88, 0.75]  # Replace with actual recall values
    }
    pr_df = pd.DataFrame(pr_data)
    st.dataframe(pr_df)
