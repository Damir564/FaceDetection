from imutils.video import VideoStream
from imutils import face_utils
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import imutils
import time
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = VideoStream().start()
time.sleep(2.0)
font = cv2.FONT_HERSHEY_SIMPLEX
nums = [27,30,33,51,62,66,57,8]

def qdist(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

while True:
    image = cap.read()
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #(x, y, w, h) = face_utils.rect_to_bb(rect)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        coords = [tuple(shape[j]) for j in nums] 
        X = np.array([shape[j][0] for j in nums])
        
        Y = np.array([shape[j][1] for j in nums])
        Y = Y[:, np.newaxis]
        print(X,Y)
        regr = LinearRegression()
        regr.fit(Y, X)
        score = regr.score(Y, X)
        if score > 0.8:
            color = (0,255,0)
        elif score > 0.5:
            color = (0,255,255)
        else:
            color = (0,0,255)
        for j in range(len(coords)-1):    
            cv2.line(image, coords[j], coords[j+1], color, 2)   
        dist1 = qdist(tuple(shape[2]), tuple(shape[30]))
        dist2 = qdist(tuple(shape[30]), tuple(shape[14]))
        if abs(dist1 - dist2) > 0.4 * (dist1+dist2)/2:
            color2 = (0,0,255)
        else:
            color2 = (0,255,0)
        cv2.line(image, tuple(shape[2]), tuple(shape[30]), color2, 2) 
        cv2.line(image, tuple(shape[30]), tuple(shape[14]), color2, 2) 
        #cv2.putText(image, str(score) ,(10,500), font, 1,color,2)
    cv2.imshow("Output", image)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()
