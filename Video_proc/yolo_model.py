from ultralytics import YOLO

model = YOLO("yolov8n.pt").to("cpu")

def detect_objects(frame, conf=0.4):
    results = model(frame, conf=conf)
    result_frame = results[0].plot()
    labels = results[0].boxes.cls.tolist()
    names = [results[0].names[int(c)] for c in labels]
    return result_frame, names
