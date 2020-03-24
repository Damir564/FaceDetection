from imutils.video import VideoStream
from imutils import face_utils
import datetime
import imutils
import time
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = VideoStream().start()
time.sleep(2.0)
while True:
    image = cap.read()
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()
