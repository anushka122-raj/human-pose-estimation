import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv
import matplotlib.pyplot as plt
import json
import random
from datetime import datetime


# ============================================================
# Angle Calculation
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
# Calories
# ============================================================

def calculate_calories(weight, MET, duration_sec):
    duration_hr = duration_sec / 3600
    calories = MET * weight * duration_hr
    return round(calories, 2)


# ============================================================
# Voice Engine
# ============================================================

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
except Exception:
    engine = None


def speak(text):
    print(f"[VOICE] {text}")
    if engine is not None:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


# ============================================================
# Form Feedback
# ============================================================

def give_feedback(score, exercise):
    if score < 70:
        speak(f"Improve your form in {exercise}!")
    elif score >= 90:
        speak("Excellent form, keep it up!")


# ============================================================
# Personal Best Tracking
# ============================================================

def update_personal_best(exercise, reps):
    try:
        with open("personal_best.json", "r") as f:
            best_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_data = {}

    best_reps = best_data.get(exercise, 0)

    if reps > best_reps:
        best_data[exercise] = reps

        with open("personal_best.json", "w") as f:
            json.dump(best_data, f, indent=4)

        speak(f"New personal best in {exercise}: {reps} reps!")
        return True

    return False


# ============================================================
# Rep Speed
# ============================================================

def check_rep_speed(rep_times):
    if len(rep_times) >= 2:
        speed = rep_times[-1] - rep_times[-2]

        if speed < 2:
            speak("Slow down, focus on control!")
        elif speed > 6:
            speak("Try to maintain a steady rhythm.")


# ============================================================
# Simulated Heart Rate
# ============================================================

def check_heart_rate():
    heart_rate = random.randint(70, 160)

    print(f"Simulated Heart Rate: {heart_rate} BPM")

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
        avg_speed = (rep_times[-1] - rep_times[-3]) / 2
        avg_hr = sum(heart_rates[-3:]) / 3

        if avg_speed > 7 and avg_hr > 130:
            speak("You may be fatigued. Consider resting.")
        elif avg_speed < 2 and avg_hr < 90:
            speak("You might not be pushing enough.")


# ============================================================
# Motivation
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
# Workout History
# ============================================================

HISTORY_FILE = "workout_history.json"


def load_workout_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_workout_history(
    exercise,
    total_reps,
    target_reps,
    duration,
    avg_form,
    avg_hr,
    calories
):
    history = load_workout_history()

    workout = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "exercise": exercise,
        "reps": total_reps,
        "target_reps": target_reps,
        "duration": round(duration, 2),
        "form_score": round(avg_form, 1),
        "heart_rate": round(avg_hr, 1),
        "calories": calories
    }

    history.append(workout)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    return workout


def get_previous_workout(exercise):
    history = load_workout_history()

    same_exercise = [
        workout for workout in history
        if workout["exercise"] == exercise
    ]

    if same_exercise:
        return same_exercise[-1]

    return None


# ============================================================
# Compare Workouts
# ============================================================

def compare_workout(current_workout, previous_workout):
    if previous_workout is None:
        print("\nNo previous workout available for comparison.")
        speak("This is your first recorded workout for this exercise.")
        return

    current_reps = current_workout["reps"]
    previous_reps = previous_workout["reps"]

    current_form = current_workout["form_score"]
    previous_form = previous_workout["form_score"]

    rep_difference = current_reps - previous_reps
    form_difference = current_form - previous_form

    print("\n" + "=" * 55)
    print("             PERFORMANCE COMPARISON")
    print("=" * 55)
    print(f"Previous Reps      : {previous_reps}")
    print(f"Current Reps       : {current_reps}")
    print(f"Rep Difference     : {rep_difference:+d}")
    print(f"Previous Form      : {previous_form}")
    print(f"Current Form       : {current_form}")
    print(f"Form Difference    : {form_difference:+.1f}")

    if rep_difference > 0:
        print("Progress Status    : IMPROVED")
        speak(f"Great job! You performed {rep_difference} more reps than before.")
    elif rep_difference < 0:
        print("Progress Status    : LOWER")
        speak("Your rep count was lower than before. Keep training!")
    else:
        print("Progress Status    : STABLE")

    if form_difference > 0:
        print("Form Progress      : IMPROVED")
    elif form_difference < 0:
        print("Form Progress      : NEEDS WORK")
    else:
        print("Form Progress      : STABLE")

    print("=" * 55)


# ============================================================
# Workout History Display
# ============================================================

def display_workout_history():
    history = load_workout_history()

    print("\n" + "=" * 70)
    print("                    WORKOUT HISTORY")
    print("=" * 70)

    if not history:
        print("No previous workouts found.")
        return

    for i, workout in enumerate(history, start=1):
        print(f"\nWorkout #{i}")
        print(f"Date       : {workout['date']}")
        print(f"Time       : {workout.get('time', 'N/A')}")
        print(f"Exercise   : {workout['exercise']}")
        print(f"Reps       : {workout['reps']}")
        print(f"Target     : {workout['target_reps']}")
        print(f"Form       : {workout['form_score']}")
        print(f"Heart Rate : {workout['heart_rate']} BPM")
        print(f"Calories   : {workout['calories']} kcal")

    print("=" * 70)


# ============================================================
# Progress Analytics
# ============================================================

def show_progress_graph(exercise):
    history = load_workout_history()

    exercise_history = [
        workout for workout in history
        if workout["exercise"] == exercise
    ]

    if len(exercise_history) < 2:
        print("\nNot enough workout history to display progress graph.")
        return

    reps = [w["reps"] for w in exercise_history]
    form = [w["form_score"] for w in exercise_history]
    calories = [w["calories"] for w in exercise_history]
    x = range(1, len(exercise_history) + 1)

    plt.figure()
    plt.plot(x, reps, "bo-")
    plt.title(f"{exercise} - Rep Progress")
    plt.xlabel("Workout Number")
    plt.ylabel("Reps")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.plot(x, form, "go-")
    plt.title(f"{exercise} - Form Score Progress")
    plt.xlabel("Workout Number")
    plt.ylabel("Form Score")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.plot(x, calories, "ro-")
    plt.title(f"{exercise} - Calories Burned")
    plt.xlabel("Workout Number")
    plt.ylabel("Calories")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


# ============================================================
# Workout Summary
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
    print("\n" + "=" * 55)
    print("                 WORKOUT SUMMARY")
    print("=" * 55)

    print(f"Exercise           : {exercise}")
    print(f"Total Reps         : {total_reps}")
    print(f"Target Reps        : {target_reps}")
    print(f"Workout Duration   : {round(duration, 2)} seconds")
    print(f"Average Form Score : {round(avg_form, 1)}")
    print(f"Average Heart Rate : {round(avg_hr, 1)} BPM (simulated)")
    print(f"Calories Burned    : {calories} kcal")

    if total_reps >= target_reps:
        print("Goal Status        : ACHIEVED!")
    else:
        print(f"Goal Status        : {target_reps - total_reps} reps remaining")

    if len(rep_times) >= 2:
        speeds = [
            rep_times[i] - rep_times[i - 1]
            for i in range(1, len(rep_times))
        ]

        print(f"Average Rep Speed  : {round(sum(speeds) / len(speeds), 2)} sec")
        print(f"Fastest Rep        : {round(min(speeds), 2)} sec")
        print(f"Slowest Rep        : {round(max(speeds), 2)} sec")
    else:
        print("Average Rep Speed  : N/A")
        print("Fastest Rep        : N/A")
        print("Slowest Rep        : N/A")

    if avg_form >= 90:
        rating = "Excellent"
    elif avg_form >= 75:
        rating = "Good"
    elif avg_form >= 60:
        rating = "Average"
    else:
        rating = "Needs Improvement"

    print(f"Performance Rating : {rating}")
    print("=" * 55)

    speak(
        f"Workout complete. You performed {total_reps} reps. "
        f"Your average form score was {round(avg_form)}. "
        f"Your performance was {rating}."
    )


# ============================================================
# Main Program
# ============================================================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit


# ============================================================
# Variables
# ============================================================

counter = 0
stage = None
score = 0
rep_times = []
heart_rates = []
form_scores = []

goal_reached = False

# NEW FEATURE:
# Press P to pause/resume the workout.
paused = False
paused_total = 0.0
pause_started = None


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

choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    exercise = "Bicep Curl"
elif choice == "2":
    exercise = "Squat"
elif choice == "3":
    exercise = "Push-up"
else:
    print("Invalid choice, defaulting to Bicep Curl")
    exercise = "Bicep Curl"


# ============================================================
# Workout Goal
# ============================================================

while True:
    try:
        target_reps = int(input("Enter your target reps: "))

        if target_reps > 0:
            break

        print("Please enter a positive number.")

    except ValueError:
        print("Please enter a valid number.")


# ============================================================
# Starting Message
# ============================================================

speak(
    f"Starting {exercise} tracking. "
    f"Your target is {target_reps} reps."
)

print("\nControls:")
print("Q = Finish workout")
print("P = Pause / Resume workout")


# ============================================================
# CSV Workout Log
# ============================================================

log_file = open(
    "workout_log.csv",
    mode="w",
    newline=""
)

writer = csv.writer(log_file)

writer.writerow([
    "Exercise",
    "Rep Count",
    "Stage",
    "Time (s)",
    "Form Score",
    "Heart Rate",
    "Paused"
])


# ============================================================
# Live Graph
# ============================================================

plt.ion()

fig, ax = plt.subplots()
ax.set_title(f"{exercise} Progress")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Reps")

line, = ax.plot([], [], "bo-")


def update_graph():
    line.set_xdata(rep_times)
    line.set_ydata(range(1, len(rep_times) + 1))

    ax.relim()
    ax.autoscale_view()

    plt.draw()
    plt.pause(0.01)


# ============================================================
# Workout Timer
# ============================================================

session_start = time.time()


def active_elapsed_time():
    """Returns workout time excluding paused time."""
    return time.time() - session_start - paused_total


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
            print("Could not read webcam frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ====================================================
        # Keyboard Controls
        # ====================================================

        key = cv2.waitKey(10) & 0xFF

        # NEW FEATURE: Pause / Resume
        if key == ord("p"):

            paused = not paused

            if paused:
                pause_started = time.time()
                speak("Workout paused.")
            else:
                if pause_started is not None:
                    paused_total += time.time() - pause_started

                pause_started = None
                speak("Workout resumed.")

        if key == ord("q"):
            break


        # ====================================================
        # Pose Processing
        # ====================================================

        if not paused and results.pose_landmarks:

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ].y
                ]

                elbow = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_ELBOW.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_ELBOW.value
                    ].y
                ]

                wrist = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_WRIST.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_WRIST.value
                    ].y
                ]

                arm_angle = calculate_angle(
                    shoulder,
                    elbow,
                    wrist
                )

                # ====================================================
                # Rep Detection
                # ====================================================

                if arm_angle > 160:
                    stage = "down"

                if arm_angle < 30 and stage == "down":

                    stage = "up"
                    counter += 1

                    current_time = active_elapsed_time()
                    rep_times.append(current_time)

                    # Goal
                    if counter >= target_reps and not goal_reached:
                        goal_reached = True

                        speak(
                            f"Congratulations! "
                            f"You reached your goal of "
                            f"{target_reps} reps!"
                        )

                    # Heart Rate
                    heart_rate = check_heart_rate()
                    heart_rates.append(heart_rate)

                    # Form
                    score = form_score(
                        arm_angle,
                        30,
                        160
                    )

                    form_scores.append(score)

                    # Feedback
                    give_feedback(score, exercise)

                    # Speed
                    check_rep_speed(rep_times)

                    # Fatigue
                    check_fatigue(
                        rep_times,
                        heart_rates
                    )

                    # Motivation
                    give_motivation()

                    # CSV
                    writer.writerow([
                        exercise,
                        counter,
                        stage,
                        round(current_time, 2),
                        score,
                        heart_rate,
                        paused
                    ])

                    log_file.flush()

                    # Graph
                    update_graph()

            except Exception as e:
                print(f"Pose processing warning: {e}")


        # ====================================================
        # Draw Pose
        # ====================================================

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )


        # ====================================================
        # Display Information
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

            cv2.putText(
                image,
                "(Simulated)",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        progress = min(
            (counter / target_reps) * 100,
            100
        )

        cv2.putText(
            image,
            f"Goal: {target_reps} reps",
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Progress: {progress:.0f}%",
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Active workout timer
        elapsed = active_elapsed_time()

        cv2.putText(
            image,
            f"Time: {elapsed:.0f}s",
            (20, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if goal_reached:
            cv2.putText(
                image,
                "GOAL ACHIEVED!",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

        # ====================================================
        # NEW FEATURE: PAUSE DISPLAY
        # ====================================================

        if paused:
            cv2.putText(
                image,
                "WORKOUT PAUSED",
                (20, 405),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3
            )

            cv2.putText(
                image,
                "Press P to resume",
                (20, 445),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                image,
                "P = Pause | Q = Finish",
                (20, 405),
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


# ============================================================
# End Workout
# ============================================================

cap.release()
cv2.destroyAllWindows()

log_file.close()

plt.ioff()


# ============================================================
# Calculate Statistics
# ============================================================

session_end = time.time()

# If the user ends while paused, include time up to the pause
# but exclude the paused period from workout duration.
duration = session_end - session_start - paused_total

if form_scores:
    avg_form = sum(form_scores) / len(form_scores)
else:
    avg_form = 0

if heart_rates:
    avg_hr = sum(heart_rates) / len(heart_rates)
else:
    avg_hr = 0

calories = calculate_calories(
    user_weight,
    MET_values[exercise],
    duration
)


# ============================================================
# Previous Workout
# ============================================================

previous_workout = get_previous_workout(exercise)


# ============================================================
# Personal Best
# ============================================================

is_new_best = update_personal_best(
    exercise,
    counter
)


# ============================================================
# Workout Summary
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
# Current Workout Record
# ============================================================

current_workout = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "time": datetime.now().strftime("%H:%M:%S"),
    "exercise": exercise,
    "reps": counter,
    "target_reps": target_reps,
    "duration": round(duration, 2),
    "form_score": round(avg_form, 1),
    "heart_rate": round(avg_hr, 1),
    "calories": calories
}


# ============================================================
# Save Current Workout
# ============================================================

save_workout_history(
    exercise,
    counter,
    target_reps,
    duration,
    avg_form,
    avg_hr,
    calories
)


# ============================================================
# Compare With Previous
# ============================================================

compare_workout(
    current_workout,
    previous_workout
)


# ============================================================
# Personal Best Message
# ============================================================

if is_new_best:
    print("\n🏆 NEW PERSONAL BEST!")
else:
    history = load_workout_history()

    same_exercise = [
        workout for workout in history
        if workout["exercise"] == exercise
    ]

    if same_exercise:
        best = max(
            workout["reps"]
            for workout in same_exercise
        )

        print(f"\n🏆 Personal Best: {best} reps")


# ============================================================
# Workout History
# ============================================================

display_workout_history()


# ============================================================
# Progress Analytics
# ============================================================

print("\nOpening progress analytics...")
show_progress_graph(exercise)


# ============================================================
# Final Message
# ============================================================

print("\n" + "=" * 55)
print("             WORKOUT SESSION COMPLETE")
print("=" * 55)
print("Workout saved successfully!")
print("Files created/updated:")
print("✓ workout_log.csv")
print("✓ personal_best.json")
print("✓ workout_history.json")
print("=" * 55)

speak(
    "Your workout has been saved to your workout history. "
    "Keep training and stay consistent!"
)

plt.ioff()
plt.show()
