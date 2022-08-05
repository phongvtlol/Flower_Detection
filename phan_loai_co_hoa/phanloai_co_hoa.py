# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import playsound
from gtts import gTTS


from imutils import perspective
from imutils import contours
import collections
def detect_and_predict(frame_cut, maskNet):
    faces = []

    face = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    faces.append(face)
    faces = np.array(faces, dtype="float32")
    preds = maskNet.predict(faces, batch_size=2)
    return(preds)
print("[INFO] loading ...")
maskNet = load_model("co_hoa.model")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    
    frame = vs.read()
    frame = cv2.resize(frame, (400, 400))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # ham chuyen anh tu anh mau sang anh xam
    gray = cv2.GaussianBlur(gray, (5,5), 0)  # lenh lam mo giam nhieu
    
    edged = cv2.Canny(gray, 50,80)
    kerne = np.ones((7,7), np.uint8)
    kerne2 = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kerne, iterations=1)
    edged = cv2.erode(edged, kerne2, iterations=1)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("c"):
        preds = detect_and_predict(frame, maskNet)
        print(preds)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)   # tim tap hop cac duong vien co trong khung hinh
    (cnts, _) = contours.sort_contours(cnts)

    for c in cnts:  # vong lap tinh kich thuoc tung vat mot
        #print(cv2.contourArea(c))
        
        if cv2.contourArea(c) < 1000:  # co kich thuoc nho hon 100 pixel la bo qua
             continue
        (x,y,w,h) = cv2.boundingRect(c)
        center = (x+w//2, y+h//2)
        frame_crop = frame[y:y+h, x:x+w]
        preds = detect_and_predict(frame_crop, maskNet)
        for pred in preds:
            (co, hoa) = pred
    
        label = "Co" if co >hoa else "Hoa"
        if label == "Co":
            color = (0, 255, 0) 
        else:
            color =(0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(co,hoa) * 100)
        cv2.putText(frame, label, (x, y - 10),
 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame,"{}".format(center), center,
 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(frame,(x,y), (x+w,y+h), color, 1)
        cv2.rectangle(edged,(x,y), (x+w,y+h), color, 1)

    cv2.imshow("Frame", frame)
    cv2.imshow("gray2", edged)
    
    

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()