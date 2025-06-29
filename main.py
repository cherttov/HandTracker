import cv2
import mediapipe as mp
import numpy
import math

class HandTracker():
    def __init__(self):
        self.camera_capture = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_style = mp.solutions.drawing_styles

        self.windowSize = (640, 480)

        self.tracking(self.camera_capture, self.mp_draw, self.mp_draw_style)
    
    # Tracking
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
                        self.CalculateNumber(hand_landmarks)
                # self.hand_tracking(img_RGB, captured_img, self.mp_draw, self.mp_draw_style) # <-- What's that?
                # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker # <-- Why not
                # Outputs Inverted Image's to Window (as live video)
                cv2.imshow('Output', cv2.flip(img_RGB, 1))
                cv2.resizeWindow('Output', self.windowSize)
                # Closes App Once "X" Has Been Hit
                key_code = cv2.waitKey(1)
                if cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1:
                    break
            # Stops Camera Usage and Closes Output Windows
            cam.release()
            cv2.destroyAllWindows()
    
    # Calculate number show by fingers
    def CalculateNumber(self, _handLandmarks):
        count = 0

        _tipsIndex = [4, 8, 12, 16, 20]
        _palmIndex = _handLandmarks.landmark[0]

        _middleIndex = _handLandmarks.landmark[9] # Middle finger knuckle as baseline
        _referenceDistance = math.hypot(_palmIndex.x - _middleIndex.x, _palmIndex.y - _middleIndex.y)

        for n, i in enumerate(_tipsIndex):
            _tipLandmark = _handLandmarks.landmark[i]
            _tipDistance = math.hypot(_palmIndex.x - _tipLandmark.x, _palmIndex.y - _tipLandmark.y)

            _threshold = 1.3 if n == 0 else 1.8

            if _tipDistance > _referenceDistance * _threshold:
                count += 1

        print(count)

if __name__=="__main__":
    app = HandTracker()