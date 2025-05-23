import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture('Stock/stock_police.mp4')
cap.set(3,640)
cap.set(4,640)

model = YOLO("best.pt")

mycolor = (0,0,255)

classnames = ["accident","car","fire"]

while True:
    success, img = cap.read()
    results = model(img,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h))
            cv2.rectangle(img,(x1,y1),(x2,y2),mycolor,3)
            conf = math.ceil((box.conf[0]*100)) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classnames[cls]} {conf}',(max(0,x1),max(25,y1)),scale=1,thickness=1,colorB=mycolor,colorT=(255,255,255))

    cv2.imshow("Image",img)

    if cv2.waitKey(1) == ord('q'):
        break