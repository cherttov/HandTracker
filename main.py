import cv2
import mediapipe as mp
import numpy

class HandTracker():
    def __init__(self):
        self.camera_capture = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.tracking(self.camera_capture, self.mp_draw)
    
    def tracking(self, cam, draw):
        with self.mp_hands.Hands() as hands:
            while True:
                # Error Check
                success, captured_img = cam.read()
                if not success:
                    print("error")
                    break
                # Converts and Saves images
                img_RGB = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB) # for some reason BGR2BGRA works incorrectly
                result_frames = hands.process(img_RGB)
                # Converts Back to "RGB" For Human Eyes
                img_RGB.flags.writeable = True
                img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
                # Draws Hand's Overlay
                if result_frames.multi_hand_landmarks:
                    for hand_landmarks in result_frames.multi_hand_landmarks:
                        draw.draw_landmarks(img_RGB,
                                            hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)
                # Outputs Image's to Window (as live video)
                cv2.imshow('Output', cv2.flip(img_RGB, 1))
                # Closes App Once "X" Has Been Hit
                key_code = cv2.waitKey(1)
                if cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1:
                    break
            # Stops Camera Usage and Closes Output Windows
            cam.release()
            cv2.destroyAllWindows()

if __name__=="__main__":
    app = HandTracker()