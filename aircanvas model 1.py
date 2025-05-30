import cv2
import mediapipe as mp
import numpy as np

class HandDrawer:
    def __init__(self):
        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Drawing related variables
        self.canvas = None
        self.prev_x, self.prev_y = None, None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Create a black canvas for drawing
        self.canvas = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
            h, w, c = frame.shape

            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Convert the BGR image to RGB.
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find hands.
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Draw landmarks on the frame for visualization
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get coordinates of the tip of the index finger (landmark 8)
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)

                # Get coordinates of the tip of the thumb (landmark 4)
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_x = int(thumb_tip.x * w)
                thumb_y = int(thumb_tip.y * h)

                # Check distance between thumb and index fingertip to detect drawing mode
                distance = np.hypot(thumb_x - x, thumb_y - y)

                # If distance is small, assume drawing mode
                if distance < 40:
                    if self.prev_x is None and self.prev_y is None:
                        self.prev_x, self.prev_y = x, y
                    # Draw line from previous point to current
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (0, 0, 255), thickness=5)
                    self.prev_x, self.prev_y = x, y
                else:
                    # When fingers apart, reset the previous points to avoid drawing lines while moving without drawing intent
                    self.prev_x, self.prev_y = None, None

            # Combine the frame and the canvas
            combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

            # Show the combined frame
            cv2.imshow("Hand Drawing", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear the canvas when 'c' is pressed
                self.canvas = np.zeros_like(frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    drawer = HandDrawer()
    drawer.run()

