import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3   # NEW: voice feedback

# -----------------------------
# Function to calculate angle
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# -----------------------------
# Voice Engine Setup
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)   # speed of speech
engine.setProperty('volume', 1.0) # max volume

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# FPS Calculation
prev_time = 0

# Rep Counter Variables
counter = 0
stage = None

# -----------------------------
# Pose Model
# -----------------------------
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # LEFT ARM LANDMARKS
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate Angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Convert elbow position to screen coordinates
            elbow_coords = tuple(np.multiply(elbow, [640, 480]).astype(int))

            # -----------------------------
            # Color-coded feedback
            # -----------------------------
            if 40 < angle < 160:
                color = (0, 255, 0)   # Green = good form
            else:
                color = (0, 0, 255)   # Red = bad form
                speak("Fix your form!")   # Voice warning

            # Show angle on elbow
            cv2.putText(image, str(int(angle)), elbow_coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # -----------------------------
            # Rep Counter Logic + Voice
            # -----------------------------
            if angle > 160:
                stage = "DOWN"
            if angle < 40 and stage == "DOWN":
                stage = "UP"
                counter += 1
                speak(f"Good rep {counter}")   # Voice feedback

        except:
            pass

        # -----------------------------
        # Status Box
        # -----------------------------
        cv2.rectangle(image, (0, 0), (250, 120), (0, 0, 0), -1)

        # Reps
        cv2.putText(image, "REPS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(counter), (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Stage
        cv2.putText(image, "STAGE", (120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(stage), (120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # -----------------------------
        # FPS
        # -----------------------------
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(image, f"FPS: {int(fps)}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("AI Fitness Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
