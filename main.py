import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv

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
# Function to calculate form score
# -----------------------------
def form_score(angle, min_angle, max_angle):
    midpoint = (min_angle + max_angle) / 2
    deviation = abs(angle - midpoint)
    max_deviation = (max_angle - min_angle) / 2
    score = max(0, 100 - (deviation / max_deviation) * 100)
    return round(score, 1)

# -----------------------------
# Voice Engine Setup
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

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
exercise = None   # will be chosen by user
score = 0

# -----------------------------
# Workout Logging Setup
# -----------------------------
session_start = time.time()
log_file = open("workout_log.csv", mode="w", newline="")
writer = csv.writer(log_file)
writer.writerow(["Exercise", "Rep Count", "Stage", "Time (s)", "Form Score"])

# -----------------------------
# User Input: Weight (kg)
# -----------------------------
user_weight = 60
MET_values = {"Bicep Curl": 3.8, "Squat": 5.0, "Push-up": 8.0}

# -----------------------------
# Ask user to select exercise
# -----------------------------
print("Select exercise:")
print("1 - Bicep Curl")
print("2 - Squat")
print("3 - Push-up")
choice = input("Enter choice (1/2/3): ")

if choice == "1":
    exercise = "Bicep Curl"
elif choice == "2":
    exercise = "Squat"
elif choice == "3":
    exercise = "Push-up"
else:
    print("Invalid choice, defaulting to Bicep Curl")
    exercise = "Bicep Curl"

speak(f"Starting {exercise} tracking!")

# -----------------------------
# Pose Model
# -----------------------------
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Bicep Curl landmarks
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            arm_angle = calculate_angle(shoulder, elbow, wrist)

            # Squat landmarks
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            leg_angle = calculate_angle(hip, knee, ankle)

            # Push-up landmarks
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            pushup_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)

            # -----------------------------
            # Rep Counting Logic per exercise
            # -----------------------------
            if exercise == "Bicep Curl":
                score = form_score(arm_angle, 40, 160)
                if arm_angle > 160:
                    stage = "DOWN"
                if arm_angle < 40 and stage == "DOWN":
                    stage = "UP"
                    counter += 1
                    speak(f"Rep {counter} {exercise}, Form {score}%")
                    duration = int(time.time() - session_start)
                    writer.writerow([exercise, counter, stage, duration, score])

            elif exercise == "Squat":
                score = form_score(leg_angle, 70, 160)
                if leg_angle > 160:
                    stage = "UP"
                if leg_angle < 70 and stage == "UP":
                    stage = "DOWN"
                    counter += 1
                    speak(f"Rep {counter} {exercise}, Form {score}%")
                    duration = int(time.time() - session_start)
                    writer.writerow([exercise, counter, stage, duration, score])

            elif exercise == "Push-up":
                score = form_score(pushup_angle, 60, 160)
                if pushup_angle > 160:
                    stage = "UP"
                if pushup_angle < 60 and stage == "UP":
                    stage = "DOWN"
                    counter += 1
                    speak(f"Rep {counter} {exercise}, Form {score}%")
                    duration = int(time.time() - session_start)
                    writer.writerow([exercise, counter, stage, duration, score])

        except:
            pass

        # Status Box
        cv2.rectangle(image, (0, 0), (340, 200), (0, 0, 0), -1)
        cv2.putText(image, f"Exercise: {exercise}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {counter}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(image, f"Stage: {stage}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Form: {score}%", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Workout Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# -----------------------------
# Workout Summary
# -----------------------------
session_end = time.time()
total_time = int(session_end - session_start)
avg_rep_time = round(total_time / counter, 2) if counter > 0 else 0
MET_value = MET_values.get(exercise, 3.8)
calories_burned = round(MET_value * user_weight * (total_time / 3600), 2)

summary = f"Workout finished! Exercise: {exercise}, Total reps: {counter}, Duration: {total_time}s, Avg time per rep: {avg_rep_time}s, Calories burned: {calories_burned} kcal."
print(summary)
speak(summary)

writer.writerow([])
writer.writerow(["Summary", "Total Reps", "Duration (s)", "Avg Rep Time (s)", "Calories Burned (kcal)"])
writer.writerow(["", counter, total_time, avg_rep_time, calories_burned])

log_file.close()
cap.release()
cv2.destroyAllWindows()
