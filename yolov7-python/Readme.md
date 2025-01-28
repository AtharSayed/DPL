# Object Detection with YOLOv7

This Python script utilizes the YOLOv7-tiny model for real-time object detection on images or video files. It uses ONNX Runtime to load and run the pre-trained model. The script supports both image and video inputs and displays the detection results with bounding boxes, class names, and confidence scores.

## Features
- Object detection using YOLOv7-tiny on images or videos.
- Real-time detection with bounding boxes and class names.
- Adjustable confidence threshold for detections.
- Supports both local image files and video streams (e.g., webcam, video files).
- Utilizes OpenCV for image/video manipulation and display.

## Requirements
To run this script, the following libraries are required:

- **OpenCV** (`cv2`)
- **NumPy**
- **ONNX Runtime**
- **argparse** (for command-line arguments)
- **random** (for generating random colors for bounding boxes)

You can install the required dependencies using pip:

```bash
pip install opencv-python numpy onnxruntime
