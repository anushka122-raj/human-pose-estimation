import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv   # for workout logging

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

# Rep Counter Variables
counter = 0
stage = None

# -----------------------------
# Workout Logging Setup
# -----------------------------
session_start = time.time()
log_file = open("workout_log.csv", mode="w", newline="")
writer = csv.writer(log_file)
writer.writerow(["Rep Count", "Stage", "Time (s)"])  # header row

# -----------------------------
# User Input: Weight (kg)
# -----------------------------
user_weight = 60  # <-- change this to your weight in kg
MET_value = 3.8   # MET for bicep curls

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
            # Rep Counter Logic + Voice + Logging
            # -----------------------------
            if angle > 160:
                stage = "DOWN"
            if angle < 40 and stage == "DOWN":
                stage = "UP"
                counter += 1
                speak(f"Good rep {counter}")   # Voice feedback

                # Log rep to CSV
                duration = int(time.time() - session_start)
                writer.writerow([counter, stage, duration])

        except:
            pass

        # -----------------------------
        # Status Box
        # -----------------------------
        cv2.rectangle(image, (0, 0), (250, 150), (0, 0, 0), -1)

        # Reps
        cv2.putText(image, "REPS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(counter), (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Stage
        cv2.putText(image, "STAGE", (120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(stage), (120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Show frame
        cv2.imshow('Workout Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# -----------------------------
# Workout Summary Feature + Calories
# -----------------------------
session_end = time.time()
total_time = int(session_end - session_start)
avg_rep_time = round(total_time / counter, 2) if counter > 0 else 0

# Calories burned estimation
duration_hours = total_time / 3600
calories_burned = round(MET_value * user_weight * duration_hours, 2)

summary = f"Workout finished! Total reps: {counter}, Duration: {total_time} seconds, Avg time per rep: {avg_rep_time} seconds, Calories burned: {calories_burned} kcal."
print(summary)
speak(summary)

# Save summary to CSV
writer.writerow([])
writer.writerow(["Summary", "Total Reps", "Duration (s)", "Avg Rep Time (s)", "Calories Burned (kcal)"])
writer.writerow(["", counter, total_time, avg_rep_time, calories_burned])

log_file.close()
cap.release()
cv2.destroyAllWindows()
