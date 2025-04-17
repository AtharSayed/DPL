import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from yolo_model import detect_objects
import pandas as pd
import time

# Initialize the CSV log
log_data = []

# Title
st.set_page_config(page_title="Real-Time Video Analytics", layout="centered")
st.title("Real-Time Object Detection with YOLOv8")

# Sidebar for UI controls
st.sidebar.title("Control Panel")
st.sidebar.markdown("## Settings")
video_source = st.sidebar.selectbox("Choose video source", ["Webcam", "Upload Video"])

# Start/Stop button for webcam stream
start_button = st.sidebar.button("Start Webcam", key="start")
stop_button = st.sidebar.button("Stop Webcam", key="stop")
save_button = st.sidebar.button("Save Detections")

# Initialize webcam or video capture based on the user's choice
video_capture = None
if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)

# For file upload
uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi"])

# Create a container for live video
video_frame_container = st.empty()

# For displaying FPS
fps_display = st.empty()

# For displaying object count
object_count_display = st.empty()

# Function to handle video processing and object detection
def process_video():
    global log_data, video_capture, uploaded_file

    # Webcam streaming
    if video_source == "Webcam" and video_capture is not None and video_capture.isOpened():
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Detect objects in the frame
            result_frame, labels = detect_objects(frame)

            # Calculate FPS
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            fps_display.text(f"FPS: {fps}")

            # Display object counts
            object_count_display.text(f"Objects Detected: {len(labels)}")

            # Save detections to log data
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
            for label in labels:
                log_data.append({"timestamp": timestamp, "label": label})

            # Display frame in the Streamlit app
            result_image = Image.fromarray(result_frame)
            video_frame_container.image(result_image, channels="RGB", use_column_width=True)

            # Save log data on button press
            if save_button:
                df = pd.DataFrame(log_data)
                df.to_csv("detections_log.csv", index=False)
                st.success("Detections saved successfully!")

    # Video file upload
    elif video_source == "Upload Video" and uploaded_file is not None:
        video_capture = cv2.VideoCapture(uploaded_file)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Detect objects in the frame
            result_frame, labels = detect_objects(frame)

            # Display object counts
            object_count_display.text(f"Objects Detected: {len(labels)}")

            # Save detections to log data
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
            for label in labels:
                log_data.append({"timestamp": timestamp, "label": label})

            # Display frame in the Streamlit app
            result_image = Image.fromarray(result_frame)
            video_frame_container.image(result_image, channels="RGB", use_column_width=True)

            # Save log data on button press
            if save_button:
                df = pd.DataFrame(log_data)
                df.to_csv("detections_log.csv", index=False)
                st.success("Detections saved successfully!")

# Run the video processing if the Start button is clicked
if start_button:
    process_video()

# Stop video capture if the Stop button is clicked
if stop_button:
    if video_capture:
        video_capture.release()
        st.warning("Video capture stopped.")
