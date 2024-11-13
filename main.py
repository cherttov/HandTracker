import cv2
import mediapipe as mp
import numpy

class HandTracker():
    def __init__(self):
        self.camera_capture = cv2.VideoCapture(0)
        self.mphands = mp.solutions.hands.Hands(False)
        self.mpdraw = mp.solutions.drawing_utils

        self.tracking(self.camera_capture)
    
    # Should Be Output tbf
    def tracking(self, cam):
        while True:
            ret, img = cam.read()

            if not ret:
                print("error")
                break

            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
            cv2.imshow('img', img_RGB)
            if cv2.waitKey(1) == ord('f'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
    

if __name__=="__main__":
    app = HandTracker()