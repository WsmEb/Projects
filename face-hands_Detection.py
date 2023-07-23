
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.ColorModule import ColorFinder
from cvzone.PIDModule import PID
import cv2



camera = cv2.VideoCapture(0)
detection = HandDetector(detectionCon=0.7,maxHands=10)
face_detection = FaceDetector(minDetectionCon=0.7)
face_mech = FaceMeshDetector(staticMode=True,maxFaces=10,minDetectionCon=0.7,minTrackCon=0.7)
# color = ColorFinder(trackBar=True)
# pid = PID(0.5,0.2,0.1)

while True:
    ret,fram = camera.read()
    if ret == True:
        fram = cv2.flip(fram,1)
        hands,fram = detection.findHands(fram,flipType=False)
        face,fram = face_detection.findFaces(fram,draw=True)
        # c = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        cv2.imshow("camera",face)
        # print(color.getColorHSV(face))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()  
