import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv
import matplotlib.pyplot as plt
import json
import pandas as pd

# -----------------------------
# Function to calculate angle between three points
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
# Function to calculate form score based on angle deviation
# -----------------------------
def form_score(angle, min_angle, max_angle):
    midpoint = (min_angle + max_angle) / 2
    deviation = abs(angle - midpoint)
    max_deviation = (max_angle - min_angle) / 2
    score = max(0, 100 - (deviation / max_deviation) * 100)
    return round(score, 1)

# -----------------------------
# Function to calculate calories burned using MET formula
# -----------------------------
def calculate_calories(weight, MET, duration_sec):
    duration_hr = duration_sec / 3600  # convert seconds to hours
    calories = MET * weight * duration_hr
    return round(calories, 2)

# ----------------------------
# Voice Engine Setup
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Real-Time Form Feedback
# -----------------------------
def give_feedback(score, exercise):
    if score < 70:
        speak(f"Improve your form in {exercise}!")
    elif score >= 90:
        speak("Excellent form, keep it up!")

# -----------------------------
# Rest Timer Feature
# -----------------------------
def rest_timer(seconds=30):
    speak(f"Rest for {seconds} seconds")
    for i in range(seconds, 0, -5):  # announce every 5 seconds
        time.sleep(5)
        speak(f"{i} seconds left")
    speak("Rest over, get ready!")

# -----------------------------
# Personal Best Tracking
# -----------------------------
def update_personal_best(exercise, reps):
    try:
        with open("personal_best.json", "r") as f:
            best_data = json.load(f)
    except FileNotFoundError:
        best_data = {}

    best_reps = best_data.get(exercise, 0)
    if reps > best_reps:
        best_data[exercise] = reps
        with open("personal_best.json", "w") as f:
            json.dump(best_data, f)
        speak(f"New personal best in {exercise}: {reps} reps!")
    else:
        speak(f"Your best in {exercise} is {best_reps} reps.")

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
exercise = None
score = 0
rep_times = []  # track timestamps of reps

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
                give_feedback(score, exercise)
                if arm_angle > 160:
                    stage = "DOWN"
                if arm_angle < 40 and stage == "DOWN":
                    stage = "UP"
                    counter += 1
                    rep_times.append(int(time.time() - session_start))
                    speak(f"Rep {counter} {exercise}, Form {score}%")
                    duration = int(time.time() - session_start)
                    writer.writerow([exercise, counter, stage, duration, score])

            elif exercise == "Squat":
                score = form_score(leg_angle, 70, 160)
                give_feedback(score, exercise)
                if leg_angle > 160:
                    stage = "UP"
                if leg_angle < 70 and stage == "UP":
                    stage = "DOWN"
                    counter += 1
                    rep_times.append(int(time.time() - session_start))
                    speak(f"Rep {counter} {exercise}, Form {score}%")
                    duration = int(time.time() - session_start)
                    writer.writerow([exercise, counter, stage, duration, score])

            elif exercise == "Push-up":
                score = form_score(pushup_angle, 60, 160)
                give_feedback(score, exercise)
                if pushup_angle > 160:
                    stage = "UP"
                if pushup_angle < 60 and stage == "UP":
                    stage = "DOWN"
                    counter += 1
                    rep_times.append(int(time.time() - session_start))
                    speak(f"Rep {counter} {exercise}, Form {score}%")
                    duration = int(time.time() - session_start)
                    writer.writerow([exercise, counter, stage, duration, score])

            # -----------------------------
            # Rest Timer Trigger every 10 reps
            # -----------------------------
            if counter > 0 and counter % 10 == 0:
                rest_timer(30)

        except:
            pass

        # -----------------------------
        # Status Box Overlay (optional)
        # -----------------------------
        cv2.imshow("Workout Tracker", image)
        if cv2.waitKey(10)
