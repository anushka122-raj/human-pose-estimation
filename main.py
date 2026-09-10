import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv
import matplotlib.pyplot as plt
import json
import random
import os
from datetime import datetime, timedelta


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

        speak(
            f"New personal best in {exercise}: {reps} reps!"
        )

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

    print(
        f"Simulated Heart Rate: {heart_rate} BPM"
    )

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
                "You may be fatigued. "
                "Consider resting."
            )

        elif avg_speed < 2 and avg_hr < 90:

            speak(
                "You might not be pushing enough."
            )


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
        workout
        for workout in history
        if workout["exercise"] == exercise
    ]

    if same_exercise:
        return same_exercise[-1]

    return None


# ============================================================
# NEW FEATURE
# Workout Streak + Achievements
# ============================================================

STREAK_FILE = "workout_streak.json"


def load_streak_data():

    try:

        with open(STREAK_FILE, "r") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):

        return {
            "current_streak": 0,
            "longest_streak": 0,
            "last_workout_date": None,
            "total_workouts": 0,
            "total_reps": 0,
            "achievements": []
        }


def save_streak_data(data):

    with open(STREAK_FILE, "w") as f:
        json.dump(data, f, indent=4)


def update_workout_streak(reps):

    data = load_streak_data()

    today = datetime.now().date()

    last_date_string = data.get("last_workout_date")

    if last_date_string is None:

        current_streak = 1

    else:

        try:

            last_date = datetime.strptime(
                last_date_string,
                "%Y-%m-%d"
            ).date()

            difference = (today - last_date).days

            if difference == 0:

                # Same day
                current_streak = data.get(
                    "current_streak",
                    1
                )

            elif difference == 1:

                # Consecutive day
                current_streak = (
                    data.get("current_streak", 0) + 1
                )

            else:

                # Streak broken
                current_streak = 1

        except ValueError:

            current_streak = 1

    data["current_streak"] = current_streak

    data["longest_streak"] = max(
        data.get("longest_streak", 0),
        current_streak
    )

    data["last_workout_date"] = today.strftime(
        "%Y-%m-%d"
    )

    data["total_workouts"] = (
        data.get("total_workouts", 0) + 1
    )

    data["total_reps"] = (
        data.get("total_reps", 0) + reps
    )

    new_achievements = []

    achievement_rules = {

        "🥉 First Workout":
            data["total_workouts"] >= 1,

        "🥈 5 Workouts":
            data["total_workouts"] >= 5,

        "🥇 10 Workouts":
            data["total_workouts"] >= 10,

        "🔥 3-Day Streak":
            data["current_streak"] >= 3,

        "🔥 7-Day Streak":
            data["current_streak"] >= 7,

        "💪 100 Total Reps":
            data["total_reps"] >= 100,

        "🏆 500 Total Reps":
            data["total_reps"] >= 500
    }

    existing = data.get(
        "achievements",
        []
    )

    for achievement, condition in achievement_rules.items():

        if condition and achievement not in existing:

            existing.append(achievement)

            new_achievements.append(achievement)

    data["achievements"] = existing

    save_streak_data(data)

    return data, new_achievements


def display_streak_and_achievements():

    data = load_streak_data()

    print("\n" + "=" * 60)
    print("           🏆 WORKOUT ACHIEVEMENTS")
    print("=" * 60)

    print(
        f"Current Streak    : {data['current_streak']} day(s)"
    )

    print(
        f"Longest Streak    : {data['longest_streak']} day(s)"
    )

    print(
        f"Total Workouts    : {data['total_workouts']}"
    )

    print(
        f"Total Reps        : {data['total_reps']}"
    )

    print("\nAchievements:")

    if data["achievements"]:

        for achievement in data["achievements"]:
            print(f"  ✓ {achievement}")

    else:

        print("  No achievements yet.")

    print("=" * 60)


# ============================================================
# NEW FEATURE - XP + LEVEL SYSTEM
# ============================================================

XP_FILE = "workout_xp.json"


def load_xp_data():
    """Load persistent XP and level information."""
    try:
        with open(XP_FILE, "r") as f:
            data = json.load(f)

        return {
            "xp": int(data.get("xp", 0)),
            "level": int(data.get("level", 1)),
            "total_xp_earned": int(data.get("total_xp_earned", 0))
        }

    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return {
            "xp": 0,
            "level": 1,
            "total_xp_earned": 0
        }


def xp_required_for_level(level):
    """Return XP needed to reach the next level."""
    return 100 + ((level - 1) * 50)


def calculate_workout_xp(reps, target_reps, avg_form, duration,
                         is_new_best, current_streak):
    """
    Calculate XP earned for the completed workout.

    XP sources:
      - Base workout XP
      - Reps
      - Goal completion
      - Good form
      - Personal best
      - Streak bonus
    """
    xp = 20

    # Rep contribution, capped so very long sessions do not dominate.
    xp += min(reps * 2, 100)

    # Goal completion bonus.
    if target_reps > 0 and reps >= target_reps:
        xp += 50

    # Form bonuses.
    if avg_form >= 90:
        xp += 40
    elif avg_form >= 75:
        xp += 25
    elif avg_form >= 60:
        xp += 10

    # Reward a new personal best.
    if is_new_best:
        xp += 75

    # Streak bonus.
    xp += min(current_streak * 5, 50)

    # Small bonus for completing a meaningful session.
    if duration >= 60:
        xp += 10

    return int(xp)


def add_workout_xp(reps, target_reps, avg_form, duration,
                   is_new_best, current_streak):
    """Add XP, calculate levels, and return the updated profile."""
    data = load_xp_data()

    earned_xp = calculate_workout_xp(
        reps,
        target_reps,
        avg_form,
        duration,
        is_new_best,
        current_streak
    )

    old_level = max(1, data.get("level", 1))
    data["xp"] = max(0, data.get("xp", 0) + earned_xp)
    data["total_xp_earned"] = (
        data.get("total_xp_earned", 0) + earned_xp
    )

    # Allow XP to carry over when a level is reached.
    while data["xp"] >= xp_required_for_level(data["level"]):
        data["xp"] -= xp_required_for_level(data["level"])
        data["level"] += 1

    new_level = data["level"]

    with open(XP_FILE, "w") as f:
        json.dump(data, f, indent=4)

    leveled_up = new_level > old_level

    return data, earned_xp, leveled_up


def display_xp_status(data):
    """Display current level and XP progress."""
    required = xp_required_for_level(data["level"])
    current_xp = data["xp"]
    progress = min((current_xp / required) * 100, 100)

    print("\n" + "=" * 60)
    print("                    ⭐ XP & LEVEL")
    print("=" * 60)
    print(f"Current Level        : {data['level']}")
    print(f"XP                  : {current_xp}/{required}")
    print(f"Level Progress       : {progress:.0f}%")
    print(f"Total XP Earned      : {data['total_xp_earned']}")
    print("=" * 60)


# ============================================================
# Compare Workouts
# ============================================================

def compare_workout(
    current_workout,
    previous_workout
):

    if previous_workout is None:

        print(
            "\nNo previous workout available "
            "for comparison."
        )

        speak(
            "This is your first recorded "
            "workout for this exercise."
        )

        return

    current_reps = current_workout["reps"]
    previous_reps = previous_workout["reps"]

    current_form = current_workout["form_score"]
    previous_form = previous_workout["form_score"]

    rep_difference = (
        current_reps - previous_reps
    )

    form_difference = (
        current_form - previous_form
    )

    print("\n" + "=" * 55)
    print("             PERFORMANCE COMPARISON")
    print("=" * 55)

    print(
        f"Previous Reps      : {previous_reps}"
    )

    print(
        f"Current Reps       : {current_reps}"
    )

    print(
        f"Rep Difference     : {rep_difference:+d}"
    )

    print(
        f"Previous Form      : {previous_form}"
    )

    print(
        f"Current Form       : {current_form}"
    )

    print(
        f"Form Difference    : {form_difference:+.1f}"
    )

    if rep_difference > 0:

        print(
            "Progress Status    : IMPROVED"
        )

        speak(
            f"Great job! You performed "
            f"{rep_difference} more reps than before."
        )

    elif rep_difference < 0:

        print(
            "Progress Status    : LOWER"
        )

        speak(
            "Your rep count was lower "
            "than before. Keep training!"
        )

    else:

        print(
            "Progress Status    : STABLE"
        )

    if form_difference > 0:

        print(
            "Form Progress      : IMPROVED"
        )

    elif form_difference < 0:

        print(
            "Form Progress      : NEEDS WORK"
        )

    else:

        print(
            "Form Progress      : STABLE"
        )

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

    for i, workout in enumerate(
        history,
        start=1
    ):

        print(f"\nWorkout #{i}")

        print(
            f"Date       : {workout['date']}"
        )

        print(
            f"Time       : "
            f"{workout.get('time', 'N/A')}"
        )

        print(
            f"Exercise   : "
            f"{workout['exercise']}"
        )

        print(
            f"Reps       : "
            f"{workout['reps']}"
        )

        print(
            f"Target     : "
            f"{workout['target_reps']}"
        )

        print(
            f"Form       : "
            f"{workout['form_score']}"
        )

        print(
            f"Heart Rate : "
            f"{workout['heart_rate']} BPM"
        )

        print(
            f"Calories   : "
            f"{workout['calories']} kcal"
        )

    print("=" * 70)


# ============================================================
# Progress Analytics
# ============================================================

def show_progress_graph(exercise):

    history = load_workout_history()

    exercise_history = [
        workout
        for workout in history
        if workout["exercise"] == exercise
    ]

    if len(exercise_history) < 2:

        print(
            "\nNot enough workout history "
            "to display progress graph."
        )

        return

    reps = [
        w["reps"]
        for w in exercise_history
    ]

    form = [
        w["form_score"]
        for w in exercise_history
    ]

    calories = [
        w["calories"]
        for w in exercise_history
    ]

    x = range(
        1,
        len(exercise_history) + 1
    )

    # Rep graph
    plt.figure()

    plt.plot(
        x,
        reps,
        "bo-"
    )

    plt.title(
        f"{exercise} - Rep Progress"
    )

    plt.xlabel("Workout Number")
    plt.ylabel("Reps")
    plt.grid(True)
    plt.tight_layout()

    plt.show(block=False)

    # Form graph
    plt.figure()

    plt.plot(
        x,
        form,
        "go-"
    )

    plt.title(
        f"{exercise} - Form Score Progress"
    )

    plt.xlabel("Workout Number")
    plt.ylabel("Form Score")

    plt.ylim(
        0,
        100
    )

    plt.grid(True)
    plt.tight_layout()

    plt.show(block=False)

    # Calories graph
    plt.figure()

    plt.plot(
        x,
        calories,
        "ro-"
    )

    plt.title(
        f"{exercise} - Calories Burned"
    )

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

    print(
        f"Exercise           : {exercise}"
    )

    print(
        f"Total Reps         : {total_reps}"
    )

    print(
        f"Target Reps        : {target_reps}"
    )

    print(
        f"Workout Duration   : "
        f"{round(duration, 2)} seconds"
    )

    print(
        f"Average Form Score : "
        f"{round(avg_form, 1)}"
    )

    print(
        f"Average Heart Rate : "
        f"{round(avg_hr, 1)} BPM (simulated)"
    )

    print(
        f"Calories Burned    : "
        f"{calories} kcal"
    )

    if total_reps >= target_reps:

        print(
            "Goal Status        : ACHIEVED!"
        )

    else:

        print(
            f"Goal Status        : "
            f"{target_reps - total_reps} reps remaining"
        )

    if len(rep_times) >= 2:

        speeds = [
            rep_times[i] - rep_times[i - 1]
            for i in range(
                1,
                len(rep_times)
            )
        ]

        print(
            f"Average Rep Speed  : "
            f"{round(sum(speeds) / len(speeds), 2)} sec"
        )

        print(
            f"Fastest Rep        : "
            f"{round(min(speeds), 2)} sec"
        )

        print(
            f"Slowest Rep        : "
            f"{round(max(speeds), 2)} sec"
        )

    else:

        print(
            "Average Rep Speed  : N/A"
        )

        print(
            "Fastest Rep        : N/A"
        )

        print(
            "Slowest Rep        : N/A"
        )

    if avg_form >= 90:

        rating = "Excellent"

    elif avg_form >= 75:

        rating = "Good"

    elif avg_form >= 60:

        rating = "Average"

    else:

        rating = "Needs Improvement"

    print(
        f"Performance Rating : {rating}"
    )

    print("=" * 55)

    speak(
        f"Workout complete. "
        f"You performed {total_reps} reps. "
        f"Your average form score was "
        f"{round(avg_form)}. "
        f"Your performance was {rating}."
    )


# ============================================================
# SMART EXERCISE ANALYSIS
# ============================================================
# This feature gives each exercise its own pose measurement and
# rep rules instead of using the same arm angle for every exercise.


def landmark_point(landmarks, landmark_id):
    """Return a MediaPipe landmark as [x, y]."""
    point = landmarks[landmark_id.value]
    return [point.x, point.y]


def landmark_visibility(landmarks, landmark_id):
    """Return landmark visibility, falling back safely to 0."""
    return getattr(landmarks[landmark_id.value], "visibility", 0.0)


def choose_visible_side(landmarks, left_ids, right_ids):
    """
    Choose the side with the better combined visibility.
    This makes the tracker more tolerant when one side is hidden.
    """
    left_visibility = sum(
        landmark_visibility(landmarks, landmark_id)
        for landmark_id in left_ids
    )
    right_visibility = sum(
        landmark_visibility(landmarks, landmark_id)
        for landmark_id in right_ids
    )

    if max(left_visibility, right_visibility) < 0.45:
        return None

    return "left" if left_visibility >= right_visibility else "right"


def get_exercise_pose(landmarks, exercise):
    """
    Return:
        angle, angle_name, measurement_label, feedback

    Bicep Curl:
        shoulder-elbow-wrist angle.

    Squat:
        hip-knee-ankle angle, using the more visible side.

    Push-up:
        shoulder-elbow-wrist angle, using the more visible side.
    """
    if exercise == "Bicep Curl":
        side = choose_visible_side(
            landmarks,
            [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
            ],
            [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
            ],
        )

        if side is None:
            return None, "Elbow Angle", "Arm", "Move into camera view"

        if side == "left":
            shoulder_id = mp_pose.PoseLandmark.LEFT_SHOULDER
            elbow_id = mp_pose.PoseLandmark.LEFT_ELBOW
            wrist_id = mp_pose.PoseLandmark.LEFT_WRIST
        else:
            shoulder_id = mp_pose.PoseLandmark.RIGHT_SHOULDER
            elbow_id = mp_pose.PoseLandmark.RIGHT_ELBOW
            wrist_id = mp_pose.PoseLandmark.RIGHT_WRIST

        angle = calculate_angle(
            landmark_point(landmarks, shoulder_id),
            landmark_point(landmarks, elbow_id),
            landmark_point(landmarks, wrist_id),
        )

        return angle, "Elbow Angle", side.title(), ""

    if exercise == "Squat":
        side = choose_visible_side(
            landmarks,
            [
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
            ],
            [
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
            ],
        )

        if side is None:
            return None, "Knee Angle", "Leg", "Show your full body"

        if side == "left":
            hip_id = mp_pose.PoseLandmark.LEFT_HIP
            knee_id = mp_pose.PoseLandmark.LEFT_KNEE
            ankle_id = mp_pose.PoseLandmark.LEFT_ANKLE
        else:
            hip_id = mp_pose.PoseLandmark.RIGHT_HIP
            knee_id = mp_pose.PoseLandmark.RIGHT_KNEE
            ankle_id = mp_pose.PoseLandmark.RIGHT_ANKLE

        angle = calculate_angle(
            landmark_point(landmarks, hip_id),
            landmark_point(landmarks, knee_id),
            landmark_point(landmarks, ankle_id),
        )

        return angle, "Knee Angle", side.title(), ""

    if exercise == "Push-up":
        side = choose_visible_side(
            landmarks,
            [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
            ],
            [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
            ],
        )

        if side is None:
            return None, "Elbow Angle", "Arm", "Move into camera view"

        if side == "left":
            shoulder_id = mp_pose.PoseLandmark.LEFT_SHOULDER
            elbow_id = mp_pose.PoseLandmark.LEFT_ELBOW
            wrist_id = mp_pose.PoseLandmark.LEFT_WRIST
        else:
            shoulder_id = mp_pose.PoseLandmark.RIGHT_SHOULDER
            elbow_id = mp_pose.PoseLandmark.RIGHT_ELBOW
            wrist_id = mp_pose.PoseLandmark.RIGHT_WRIST

        angle = calculate_angle(
            landmark_point(landmarks, shoulder_id),
            landmark_point(landmarks, elbow_id),
            landmark_point(landmarks, wrist_id),
        )

        return angle, "Elbow Angle", side.title(), ""

    return None, "Angle", "Body", "Unknown exercise"


def exercise_rep_state(exercise, angle, stage):
    """
    Exercise-specific rep state machine.

    Bicep Curl:
        extended -> curled = 1 rep

    Squat:
        standing -> deep squat -> standing = 1 rep

    Push-up:
        arms extended -> lowered -> arms extended = 1 rep
    """
    if exercise == "Bicep Curl":
        if angle >= 155:
            return "down", False
        if angle <= 45 and stage == "down":
            return "up", True
        return stage, False

    if exercise == "Squat":
        if angle >= 160:
            return "up", stage == "bottom"
        if angle <= 100:
            return "bottom", False
        return stage, False

    if exercise == "Push-up":
        if angle >= 160:
            return "up", stage == "down"
        if angle <= 95:
            return "down", False
        return stage, False

    return stage, False


def exercise_form_score(exercise, minimum_angle, maximum_angle):
    """
    Score the quality of the completed movement using the deepest
    and most extended positions reached during the rep.
    """
    if minimum_angle is None or maximum_angle is None:
        return 0

    if exercise == "Bicep Curl":
        bottom_score = max(
            0,
            100 - abs(minimum_angle - 35) * 2.0
        )
        top_score = max(
            0,
            100 - abs(maximum_angle - 170) * 1.5
        )

    elif exercise == "Squat":
        bottom_score = max(
            0,
            100 - abs(minimum_angle - 90) * 2.0
        )
        top_score = max(
            0,
            100 - abs(maximum_angle - 170) * 1.5
        )

    elif exercise == "Push-up":
        bottom_score = max(
            0,
            100 - abs(minimum_angle - 90) * 1.8
        )
        top_score = max(
            0,
            100 - abs(maximum_angle - 170) * 1.5
        )

    else:
        return 0

    return round((bottom_score + top_score) / 2, 1)


def exercise_form_feedback(exercise, minimum_angle, maximum_angle):
    """Return a short, exercise-specific form cue."""
    if minimum_angle is None or maximum_angle is None:
        return "Complete a full movement"

    if exercise == "Bicep Curl":
        if minimum_angle > 55:
            return "Curl a little higher"
        if maximum_angle < 145:
            return "Fully extend your arm"
        return "Good curl range"

    if exercise == "Squat":
        if minimum_angle > 110:
            return "Squat a little deeper"
        if maximum_angle < 150:
            return "Stand fully upright"
        return "Good squat depth"

    if exercise == "Push-up":
        if minimum_angle > 110:
            return "Lower your chest more"
        if maximum_angle < 150:
            return "Extend your arms fully"
        return "Good push-up range"

    return "Keep moving"


# ============================================================
# Main Program
# ============================================================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

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

# Smart exercise-analysis state
current_angle = None
angle_name = "Angle"
tracking_side = ""
rep_min_angle = None
rep_max_angle = None
form_feedback_text = "Get into position"

goal_reached = False


# ============================================================
# Pause / Resume Feature
# ============================================================

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

choice = input(
    "Enter choice (1/2/3): "
).strip()


if choice == "1":

    exercise = "Bicep Curl"

elif choice == "2":

    exercise = "Squat"

elif choice == "3":

    exercise = "Push-up"

else:

    print(
        "Invalid choice, defaulting "
        "to Bicep Curl"
    )

    exercise = "Bicep Curl"


# ============================================================
# Workout Goal
# ============================================================

while True:

    try:

        target_reps = int(
            input(
                "Enter your target reps: "
            )
        )

        if target_reps > 0:
            break

        print(
            "Please enter a positive number."
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )


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
    "Angle",
    "Angle Type",
    "Side",
    "Form Score",
    "Heart Rate",
    "Paused"
])


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

    line.set_xdata(
        rep_times
    )

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
# Workout Timer
# ============================================================

session_start = time.time()


def active_elapsed_time():

    return (
        time.time()
        - session_start
        - paused_total
    )


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

            print(
                "Could not read webcam frame."
            )

            break

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image.flags.writeable = False

        results = pose.process(
            image
        )

        image.flags.writeable = True

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )


        # ====================================================
        # Keyboard Controls
        # ====================================================

        key = cv2.waitKey(10) & 0xFF


        # ====================================================
        # Pause / Resume
        # ====================================================

        if key == ord("p"):

            paused = not paused

            if paused:

                pause_started = time.time()

                speak(
                    "Workout paused."
                )

            else:

                if pause_started is not None:

                    paused_total += (
                        time.time()
                        - pause_started
                    )

                pause_started = None

                speak(
                    "Workout resumed."
                )


        # ====================================================
        # Quit
        # ====================================================

        if key == ord("q"):

            break


        # ====================================================
        # Smart Exercise Pose Processing
        # ====================================================

        if (
            not paused
            and results.pose_landmarks
        ):

            try:

                landmarks = (
                    results.pose_landmarks.landmark
                )

                (
                    current_angle,
                    angle_name,
                    tracking_side,
                    pose_message
                ) = get_exercise_pose(
                    landmarks,
                    exercise
                )

                if current_angle is not None:

                    # Track the full range of motion reached
                    # during the current repetition.
                    if rep_min_angle is None:
                        rep_min_angle = current_angle
                        rep_max_angle = current_angle
                    else:
                        rep_min_angle = min(
                            rep_min_angle,
                            current_angle
                        )
                        rep_max_angle = max(
                            rep_max_angle,
                            current_angle
                        )

                    new_stage, completed_rep = (
                        exercise_rep_state(
                            exercise,
                            current_angle,
                            stage
                        )
                    )

                    # Reset range tracking when the user first
                    # reaches the starting position.
                    if stage is None and new_stage is not None:
                        rep_min_angle = current_angle
                        rep_max_angle = current_angle

                    stage = new_stage

                    form_feedback_text = (
                        exercise_form_feedback(
                            exercise,
                            rep_min_angle,
                            rep_max_angle
                        )
                    )

                    if completed_rep:

                        counter += 1

                        current_time = (
                            active_elapsed_time()
                        )

                        rep_times.append(
                            current_time
                        )

                        # Calculate form from the actual range
                        # of motion of this completed repetition.
                        score = exercise_form_score(
                            exercise,
                            rep_min_angle,
                            rep_max_angle
                        )

                        form_scores.append(
                            score
                        )

                        # Prepare range tracking for the next rep.
                        completed_min_angle = rep_min_angle
                        completed_max_angle = rep_max_angle
                        rep_min_angle = None
                        rep_max_angle = None

                        # =================================================
                        # Goal Detection
                        # =================================================

                        if (
                            counter >= target_reps
                            and not goal_reached
                        ):

                            goal_reached = True

                            speak(
                                f"Congratulations! "
                                f"You reached your goal "
                                f"of {target_reps} reps!"
                            )

                        # =================================================
                        # Heart Rate
                        # =================================================

                        heart_rate = (
                            check_heart_rate()
                        )

                        heart_rates.append(
                            heart_rate
                        )

                        # =================================================
                        # Feedback
                        # =================================================

                        give_feedback(
                            score,
                            exercise
                        )

                        if score < 75:
                            speak(
                                form_feedback_text
                            )

                        # =================================================
                        # Rep Speed
                        # =================================================

                        check_rep_speed(
                            rep_times
                        )

                        # =================================================
                        # Fatigue
                        # =================================================

                        check_fatigue(
                            rep_times,
                            heart_rates
                        )

                        # =================================================
                        # Motivation
                        # =================================================

                        give_motivation()

                        # =================================================
                        # CSV
                        # =================================================

                        writer.writerow([
                            exercise,
                            counter,
                            stage,
                            round(
                                current_time,
                                2
                            ),
                            round(
                                completed_max_angle,
                                1
                            ),
                            angle_name,
                            tracking_side,
                            score,
                            heart_rate,
                            paused
                        ])

                        log_file.flush()

                        # =================================================
                        # Graph
                        # =================================================

                        update_graph()

                else:
                    form_feedback_text = (
                        pose_message
                    )

            except Exception as e:

                print(
                    f"Pose processing warning: {e}"
                )


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


        # ====================================================
        # Smart Exercise Measurement
        # ====================================================

        if current_angle is not None:

            cv2.putText(
                image,
                f"{angle_name}: {current_angle:.0f} deg",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                f"Tracking: {tracking_side}",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        cv2.putText(
            image,
            f"Cue: {form_feedback_text}",
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Heart Rate
        # ====================================================

        if heart_rates:

            cv2.putText(
                image,
                f"Heart Rate: "
                f"{heart_rates[-1]} BPM",
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


        # ====================================================
        # XP / Level Display
        # ====================================================

        current_xp_data = load_xp_data()

        cv2.putText(
            image,
            f"Level: {current_xp_data['level']}  XP: {current_xp_data['xp']}/{xp_required_for_level(current_xp_data['level'])}",
            (20, 410),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # Progress
        # ====================================================

        progress = min(
            (counter / target_reps) * 100,
            100
        )


        cv2.putText(
            image,
            f"Goal: {target_reps} reps",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            image,
            f"Progress: {progress:.0f}%",
            (20, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Timer
        # ====================================================

        elapsed = active_elapsed_time()


        cv2.putText(
            image,
            f"Time: {elapsed:.0f}s",
            (20, 375),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Goal Achieved
        # ====================================================

        if goal_reached:

            cv2.putText(
                image,
                "GOAL ACHIEVED!",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )


        # ====================================================
        # Pause Display
        # ====================================================

        if paused:

            cv2.putText(
                image,
                "WORKOUT PAUSED",
                (20, 515),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3
            )

            cv2.putText(
                image,
                "Press P to resume",
                (20, 535),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                image,
                "P = Pause | Q = Finish",
                (20, 515),
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


duration = (
    session_end
    - session_start
    - paused_total
)


if form_scores:

    avg_form = (
        sum(form_scores)
        / len(form_scores)
    )

else:

    avg_form = 0


if heart_rates:

    avg_hr = (
        sum(heart_rates)
        / len(heart_rates)
    )

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

previous_workout = (
    get_previous_workout(exercise)
)


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

    "date":
        datetime.now().strftime(
            "%Y-%m-%d"
        ),

    "time":
        datetime.now().strftime(
            "%H:%M:%S"
        ),

    "exercise":
        exercise,

    "reps":
        counter,

    "target_reps":
        target_reps,

    "duration":
        round(duration, 2),

    "form_score":
        round(avg_form, 1),

    "heart_rate":
        round(avg_hr, 1),

    "calories":
        calories
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

    print(
        "\n🏆 NEW PERSONAL BEST!"
    )

else:

    history = load_workout_history()

    same_exercise = [

        workout
        for workout in history
        if workout["exercise"] == exercise
    ]

    if same_exercise:

        best = max(
            workout["reps"]
            for workout in same_exercise
        )

        print(
            f"\n🏆 Personal Best: "
            f"{best} reps"
        )


# ============================================================
# NEW FEATURE
# Update Workout Streak
# ============================================================

streak_data, new_achievements = (
    update_workout_streak(counter)
)


# ============================================================
# NEW FEATURE - Award XP and Check Level-Up
# ============================================================

xp_data, earned_xp, leveled_up = add_workout_xp(
    counter,
    target_reps,
    avg_form,
    duration,
    is_new_best,
    streak_data["current_streak"]
)

print("\n" + "=" * 60)
print("                    ⭐ XP REWARD")
print("=" * 60)
print(f"XP Earned This Workout : +{earned_xp}")
print(f"Current Level          : {xp_data['level']}")
print(
    f"XP Progress            : "
    f"{xp_data['xp']}/{xp_required_for_level(xp_data['level'])}"
)

if leveled_up:
    print("\n🎉 LEVEL UP!")
    speak(
        f"Level up! You are now level {xp_data['level']}!"
    )

print("=" * 60)


print("\n" + "=" * 60)
print("                 🔥 STREAK UPDATE")
print("=" * 60)

print(
    f"Current Workout Streak : "
    f"{streak_data['current_streak']} day(s)"
)

print(
    f"Longest Workout Streak : "
    f"{streak_data['longest_streak']} day(s)"
)

print(
    f"Total Workouts         : "
    f"{streak_data['total_workouts']}"
)

print(
    f"Total Reps             : "
    f"{streak_data['total_reps']}"
)

print("=" * 60)


# ============================================================
# New Achievements
# ============================================================

if new_achievements:

    print(
        "\n🎉 NEW ACHIEVEMENTS UNLOCKED!"
    )

    for achievement in new_achievements:

        print(
            f"   🏆 {achievement}"
        )

        speak(
            f"Achievement unlocked: "
            f"{achievement}"
        )

else:

    print(
        "\nNo new achievements this time."
    )


# ============================================================
# Show All Achievements
# ============================================================

display_streak_and_achievements()


# ============================================================
# XP / Level Status
# ============================================================

display_xp_status(xp_data)


# ============================================================
# Workout History
# ============================================================

display_workout_history()


# ============================================================
# Progress Analytics
# ============================================================

print(
    "\nOpening progress analytics..."
)

show_progress_graph(exercise)


# ============================================================
# Final Message
# ============================================================

print("\n" + "=" * 55)

print(
    "             WORKOUT SESSION COMPLETE"
)

print("=" * 55)

print(
    "Workout saved successfully!"
)

print(
    "Files created/updated:"
)

print(
    "✓ workout_log.csv"
)

print(
    "✓ personal_best.json"
)

print(
    "✓ workout_history.json"
)

print(
    "✓ workout_streak.json"
)

print(
    "✓ workout_xp.json"
)

print("=" * 55)


speak(
    "Your workout has been saved "
    "to your workout history. "
    "Keep training and stay consistent!"
)


plt.ioff()

plt.show()
