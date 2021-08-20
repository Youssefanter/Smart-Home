import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture("testV.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1
    tmp_ptx,tmp_pty = 0,0
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (415, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[3] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            pts = np.array([[width,height],[0,height],(x+(w/2),y+(h/2))], np.int32)
            ptx, pty = x+(w/4),y+(h/2)
            if ( abs(ptx-tmp_ptx) <=20 or abs(pty-tmp_pty) <=20 ):
                motion_label = "No motion"
                motion_label_Y = " "
            elif ((ptx-tmp_ptx) < 0):
                 motion_label = "Right >>"
                 if( (pty-tmp_pty) < 0):
                     motion_label_Y = "Towrds Camera"
                 else:
                        motion_label_Y = "away from Camera"
            else:
                 motion_label = "Left <<"
                 if( (pty-tmp_pty) < 0):
                     motion_label_Y = "Towrds Camera"
                 else:
                     motion_label_Y = "away from Camera"

            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 1)
            cv2.putText(frame, motion_label + " " + motion_label_Y, (x, y+h), font, 2, (255,255,255),2)
            #cv2.polylines(frame,[pts],True,(0,255,255))
            tmp_ptx,tmp_pty = ptx, pty
            



    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1000)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()