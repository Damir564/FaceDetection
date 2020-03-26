from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from threading import Thread
import datetime
import imutils
import time
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)
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
        print(fa.check_aligment(image, gray, rect, 4))
    cv2.imshow("Output", image)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()


''' в классе FaceAligner модуля imutils добавил функцию
def check_aligment(self, image, gray, rect, a):
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = self.predictor(gray, rect)
    shape = shape_to_np(shape)

    # simple hack ;)
    if (len(shape) == 68):
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    else:
        (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    return abs(angle) >= 180 - a

https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
в этой статье он делает что-то похожее
'''
