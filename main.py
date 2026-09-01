import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv
import matplotlib.pyplot as plt
import json
import random


# ============================================================
# Function to calculate angle between three points
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# Form Score
# ============================================================

def form_score(angle, min_angle, max_angle):

    midpoint = (min_angle + max_angle) / 2
    deviation = abs(angle - midpoint)
    max_deviation = (max_angle - min_angle) / 2

    if max_deviation == 0:
        return 0

    score = max(0, 100 - (deviation / max_deviation) * 100)

    return round(score, 1)


# ============================================================
# Calories Calculation
# ============================================================

def calculate_calories(weight, MET, duration_sec):

    duration_hr = duration_sec / 3600

    calories = MET * weight * duration_hr

    return round(calories, 2)


# ============================================================
# Voice Engine
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)


def speak(text):

    engine.say(text)
    engine.runAndWait()


# ============================================================
# Form Feedback
# ============================================================

def give_feedback(score, exercise):

    if score < 70:

        speak(f"Improve your form in {exercise}!")

    elif score >= 90:

        speak("Excellent form, keep it up!")


# ============================================================
# Rest Timer
# ============================================================

def rest_timer(seconds=30):

    speak(f"Rest for {seconds} seconds")

    for i in range(seconds, 0, -5):

        time.sleep(5)

        speak(f"{i} seconds left")

    speak("Rest over, get ready!")


# ============================================================
# Personal Best Tracking
# ============================================================

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

        speak(
            f"New personal best in {exercise}: {reps} reps!"
        )

    else:

        speak(
            f"Your best in {exercise} is {best_reps} reps."
        )


# ============================================================
# Rep Speed Tracking
# ============================================================

def check_rep_speed(rep_times):

    if len(rep_times) >= 2:

        speed = rep_times[-1] - rep_times[-2]

        if speed < 2:

            speak("Slow down, focus on control!")

        elif speed > 6:

            speak("Try to maintain a steady rhythm.")

        else:

            speak("Good pace!")


# ============================================================
# Heart Rate Simulation
# ============================================================

def check_heart_rate():

    heart_rate = random.randint(70, 160)

    if heart_rate < 80:

        speak("Heart rate is low, push harder!")

    elif heart_rate > 140:

        speak("Heart rate is high, slow down!")

    else:

        speak("Heart rate is optimal.")

    return heart_rate


# ============================================================
# Fatigue Detection
# ============================================================

def check_fatigue(rep_times, heart_rates):

    if len(rep_times) >= 3 and len(heart_rates) >= 3:

        avg_speed = (
            rep_times[-1] - rep_times[-3]
        ) / 2

        avg_hr = sum(heart_rates[-3:]) / 3

        if avg_speed > 7 and avg_hr > 130:

            speak(
                "You may be fatigued. Consider resting."
            )

        elif avg_speed < 2 and avg_hr < 90:

            speak(
                "You might not be pushing enough."
            )

        else:

            speak(
                "Energy levels are stable."
            )


# ============================================================
# Motivation Quotes
# ============================================================

quotes = [

    "Push yourself, because no one else will do it for you!",

    "Sweat is just fat crying!",

    "Don't stop when you're tired, stop when you're done!",

    "The body achieves what the mind believes!",

    "Every rep makes you stronger!"

]


def give_motivation():

    speak(random.choice(quotes))


# ============================================================
# Workout Summary Dashboard
# ============================================================

def workout_summary(
    exercise,
    total_reps,
    target_reps,
    duration,
    avg_form,
    avg_hr,
    calories,
    rep_times
):

    print("\n")
    print("=" * 50)
    print("           WORKOUT SUMMARY")
    print("=" * 50)

    print(f"Exercise           : {exercise}")
    print(f"Total Reps         : {total_reps}")
    print(f"Target Reps        : {target_reps}")
    print(f"Workout Duration   : {round(duration, 2)} seconds")
    print(f"Average Form Score : {round(avg_form, 1)}")
    print(f"Average Heart Rate : {round(avg_hr, 1)} BPM")
    print(f"Calories Burned    : {calories} kcal")

    # --------------------------------------------------------
    # Goal Status
    # --------------------------------------------------------

    if total_reps >= target_reps:

        print("Goal Status        : ACHIEVED!")

    else:

        remaining = target_reps - total_reps

        print(
            f"Goal Status        : {remaining} reps remaining"
        )

    # --------------------------------------------------------
    # Calculate average rep speed
    # --------------------------------------------------------

    if len(rep_times) >= 2:

        speeds = []

        for i in range(1, len(rep_times)):

            speed = rep_times[i] - rep_times[i - 1]

            speeds.append(speed)

        average_speed = sum(speeds) / len(speeds)

        fastest_rep = min(speeds)

        print(
            f"Average Rep Speed  : {round(average_speed, 2)} sec"
        )

        print(
            f"Fastest Rep        : {round(fastest_rep, 2)} sec"
        )

    else:

        print("Average Rep Speed  : N/A")
        print("Fastest Rep        : N/A")

    # --------------------------------------------------------
    # Performance Rating
    # --------------------------------------------------------

    if avg_form >= 90:

        rating = "Excellent"

    elif avg_form >= 75:

        rating = "Good"

    elif avg_form >= 60:

        rating = "Average"

    else:

        rating = "Needs Improvement"

    print(f"Performance Rating : {rating}")

    print("=" * 50)

    speak(
        f"Workout complete. "
        f"You performed {total_reps} reps. "
        f"Your average form score was {round(avg_form)}. "
        f"Your performance was {rating}."
    )


# ============================================================
# MediaPipe Setup
# ============================================================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# ============================================================
# Webcam
# ============================================================

cap = cv2.VideoCapture(0)


# ============================================================
# Variables
# ============================================================

counter = 0

stage = None

exercise = None

score = 0

rep_times = []

heart_rates = []

form_scores = []


# ============================================================
# Workout Start Time
# ============================================================

session_start = time.time()


# ============================================================
# Workout Log
# ============================================================

log_file = open(
    "workout_log.csv",
    mode="w",
    newline=""
)

writer = csv.writer(log_file)

writer.writerow(
    [
        "Exercise",
        "Rep Count",
        "Stage",
        "Time (s)",
        "Form Score",
        "Heart Rate"
    ]
)


# ============================================================
# User Settings
# ============================================================

user_weight = 60

MET_values = {

    "Bicep Curl": 3.8,

    "Squat": 5.0,

    "Push-up": 8.0

}


# ============================================================
# Exercise Selection
# ============================================================

print("\nSelect exercise:")

print("1 - Bicep Curl")

print("2 - Squat")

print("3 - Push-up")


choice = input(
    "Enter choice (1/2/3): "
)


if choice == "1":

    exercise = "Bicep Curl"

elif choice == "2":

    exercise = "Squat"

elif choice == "3":

    exercise = "Push-up"

else:

    print(
        "Invalid choice, defaulting to Bicep Curl"
    )

    exercise = "Bicep Curl"


# ============================================================
# NEW FEATURE - WORKOUT GOAL
# ============================================================

while True:

    try:

        target_reps = int(
            input("Enter your target reps: ")
        )

        if target_reps > 0:

            break

        else:

            print("Please enter a positive number.")

    except ValueError:

        print("Please enter a valid number.")


goal_reached = False


speak(
    f"Starting {exercise} tracking. "
    f"Your target is {target_reps} reps."
)


# ============================================================
# Live Graph
# ============================================================

plt.ion()

fig, ax = plt.subplots()

ax.set_title(
    f"{exercise} Progress"
)

ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Reps"
)

line, = ax.plot(
    [],
    [],
    "bo-"
)


def update_graph():

    line.set_xdata(rep_times)

    line.set_ydata(
        range(
            1,
            len(rep_times) + 1
        )
    )

    ax.relim()

    ax.autoscale_view()

    plt.draw()

    plt.pause(0.01)


# ============================================================
# Pose Detection
# ============================================================

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            break

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image.flags.writeable = False

        # ----------------------------------------------------
        # Pose Detection
        # ----------------------------------------------------

        results = pose.process(image)

        image.flags.writeable = True

        # ----------------------------------------------------
        # RGB → BGR
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        try:

            landmarks = (
                results.pose_landmarks.landmark
            )

            # ------------------------------------------------
            # LEFT ARM LANDMARKS
            # ------------------------------------------------

            shoulder = [

                landmarks[
                    mp_pose.PoseLandmark
                    .LEFT_SHOULDER.value
                ].x,

                landmarks[
                    mp_pose.PoseLandmark
                    .LEFT_SHOULDER.value
                ].y

            ]

            elbow = [

                landmarks[
                    mp_pose.PoseLandmark
                    .LEFT_ELBOW.value
                ].x,

                landmarks[
                    mp_pose.PoseLandmark
                    .LEFT_ELBOW.value
                ].y

            ]

            wrist = [

                landmarks[
                    mp_pose.PoseLandmark
                    .LEFT_WRIST.value
                ].x,

                landmarks[
                    mp_pose.PoseLandmark
                    .LEFT_WRIST.value
                ].y

            ]

            # ------------------------------------------------
            # Calculate Arm Angle
            # ------------------------------------------------

            arm_angle = calculate_angle(
                shoulder,
                elbow,
                wrist
            )

            # ------------------------------------------------
            # Rep Detection
            # ------------------------------------------------

            if arm_angle > 160:

                stage = "down"

            if (
                arm_angle < 30
                and stage == "down"
            ):

                stage = "up"

                counter += 1

                # --------------------------------------------
                # NEW FEATURE - GOAL CHECK
                # --------------------------------------------

                if (
                    counter >= target_reps
                    and not goal_reached
                ):

                    goal_reached = True

                    speak(
                        f"Congratulations! "
                        f"You reached your goal of "
                        f"{target_reps} reps!"
                    )

                # --------------------------------------------
                # Rep Timestamp
                # --------------------------------------------

                current_time = (
                    time.time()
                    - session_start
                )

                rep_times.append(
                    current_time
                )

                # --------------------------------------------
                # Heart Rate
                # --------------------------------------------

                heart_rate = (
                    check_heart_rate()
                )

                heart_rates.append(
                    heart_rate
                )

                # --------------------------------------------
                # Form Score
                # --------------------------------------------

                score = form_score(
                    arm_angle,
                    30,
                    160
                )

                form_scores.append(
                    score
                )

                # --------------------------------------------
                # Feedback
                # --------------------------------------------

                give_feedback(
                    score,
                    exercise
                )

                # --------------------------------------------
                # Speed
                # --------------------------------------------

                check_rep_speed(
                    rep_times
                )

                # --------------------------------------------
                # Fatigue
                # --------------------------------------------

                check_fatigue(
                    rep_times,
                    heart_rates
                )

                # --------------------------------------------
                # Motivation
                # --------------------------------------------

                give_motivation()

                # --------------------------------------------
                # Save CSV
                # --------------------------------------------

                writer.writerow(
                    [
                        exercise,
                        counter,
                        stage,
                        round(
                            current_time,
                            2
                        ),
                        score,
                        heart_rate
                    ]
                )

                # --------------------------------------------
                # Update Graph
                # --------------------------------------------

                update_graph()

        except Exception:

            pass

        # ====================================================
        # Display Information on Webcam
        # ====================================================

        cv2.putText(
            image,
            f"Exercise: {exercise}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Reps: {counter}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Form Score: {score}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if heart_rates:

            cv2.putText(
                image,
                f"Heart Rate: {heart_rates[-1]} BPM",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ====================================================
        # NEW FEATURE - GOAL DISPLAY
        # ====================================================

        progress = min(
            (counter / target_reps) * 100,
            100
        )

        cv2.putText(
            image,
            f"Goal: {target_reps} reps",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Progress: {progress:.0f}%",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if goal_reached:

            cv2.putText(
                image,
                "GOAL ACHIEVED!",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

        # ====================================================
        # Draw Pose Landmarks
        # ====================================================

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # ====================================================
        # Exit Instructions
        # ====================================================

        cv2.putText(
            image,
            "Press Q to finish workout",
            (20, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        # Show Webcam
        # ====================================================

        cv2.imshow(
            "Workout Tracker",
            image
        )

        # ====================================================
        # Quit
        # ====================================================

        if cv2.waitKey(10) & 0xFF == ord("q"):

            break


# ============================================================
# End Workout
# ============================================================

cap.release()

cv2.destroyAllWindows()

log_file.close()

plt.ioff()


# ============================================================
# Calculate Workout Statistics
# ============================================================

session_end = time.time()

duration = session_end - session_start


# ============================================================
# Average Form Score
# ============================================================

if form_scores:

    avg_form = sum(form_scores) / len(form_scores)

else:

    avg_form = 0


# ============================================================
# Average Heart Rate
# ============================================================

if heart_rates:

    avg_hr = sum(heart_rates) / len(heart_rates)

else:

    avg_hr = 0


# ============================================================
# Calories
# ============================================================

calories = calculate_calories(
    user_weight,
    MET_values[exercise],
    duration
)


# ============================================================
# Personal Best
# ============================================================

update_personal_best(
    exercise,
    counter
)


# ============================================================
# Final Workout Dashboard
# ============================================================

workout_summary(
    exercise,
    counter,
    target_reps,
    duration,
    avg_form,
    avg_hr,
    calories,
    rep_times
)


# ============================================================
# Keep Final Graph Open
# ============================================================

plt.ioff()

plt.show()
