import cv2
import mediapipe as mp
import numpy

class HandTracker():
    def __init__(self):
        self.camera_capture = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_style = mp.solutions.drawing_styles

        self.tracking(self.camera_capture, self.mp_draw, self.mp_draw_style)
    
     

    def tracking(self, cam, draw, style):
        with self.mp_hands.Hands(min_detection_confidence=0.75) as hands:
            while True:
                # Error Check
                success, captured_img = cam.read()
                if not success:
                    print("error")
                    break
                # Converts and Saves images
                img_RGB = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
                result_frames = hands.process(img_RGB)
                # Converts Back to "RGB" For Human Eyes
                img_RGB.flags.writeable = True
                img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
                # Draws Hand's Overlay
                if result_frames.multi_hand_landmarks:
                    for hand_landmarks in result_frames.multi_hand_landmarks:
                        draw.draw_landmarks(img_RGB,
                                            hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS,
                                            style.get_default_hand_landmarks_style(),
                                            style.get_default_hand_connections_style())
                # self.hand_tracking(img_RGB, captured_img, self.mp_draw, self.mp_draw_style)
                # Outputs Inverted Image's to Window (as live video)
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