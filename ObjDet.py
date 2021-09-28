import numpy as np
import cv2 
#Loading YOLO algo
net = cv2.dnn.readNet("yolov4-custom_best.weights", "yolov4-custom.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layerNames = net.getLayerNames()
outputLayer = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True: 
    check, frame = webcam.read()
#    print(check)
    image = cv2.resize((frame), None, fx=0.8, fy=0.8)
    width, height, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0,), True, crop=False)     #RGB extraction
    net.setInput(blob)
    outputs = net.forward(outputLayer)
    boxes = []
    confidences = []
    classIDs = []
    flag = 0
    for out in outputs:
        for detect in out:
            score = detect[5:]
            classID = np.argmax(score)
            confidence = score[classID]
            if confidence > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                print(center_x, center_y, w, h, width, height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classIDs.append(classID)
    n = len(boxes)
    for i in range(n):
        x,y,w,h = boxes[i]
        label = str(classes[classIDs[i]])
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y+30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)    
    cv2.imshow('Image',image)
    key = cv2.waitKey(500)
        
