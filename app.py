from flask import Flask, Response, render_template
import cv2
import torch
import sys
import time
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

app = Flask(__name__)

# Path to the model and video
model_path = 'D:/smart_city_system/models/yolov7/runs/train/exp24/weights/best.pt'
video_path = 'D:/smart_city_system/data/bdd100k/sample/sample.mp4'

# Load YOLOv7 Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

# Class names
class_names = [
    'Car', 'Bus', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Other person',
    'Rider', 'Traffic light', 'Traffic sign', 'Trailer', 'Train', 'Other vehicle'
]

# Open the video file
cap = cv2.VideoCapture(video_path)

# Object Detection Function
def detect_objects(frame):
    img_tensor = torch.from_numpy(frame).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0]

    # Apply Non-Max Suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.4)

    # Draw boxes on the frame
    if pred[0] is not None and len(pred[0]):
        pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], frame.shape).round()

        for *xyxy, conf, cls in pred[0]:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Frame Generator for Video Streaming
def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.resize(frame, (640, 480))  # Resize frame
        frame = detect_objects(frame)  # Apply object detection

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
