from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    img_data = base64.b64decode(data['image'])
    img_np = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = model(img)
    output = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            output.append({
                'bbox': [x1, y1, x2, y2],
                'class': cls,
                'confidence': round(conf, 2)
            })

    return jsonify({'detections': output})

@app.route('/', methods=['GET'])
def home():
    return "YOLO Detection API is live"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
