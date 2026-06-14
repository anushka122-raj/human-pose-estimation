import cv2
import mediapipe as mp
import numpy as np
import time  # used to calculate FPS (Frames Per Second)

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# MediaPipe drawing utility (used to draw landmarks and connections)
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Pose model (used for human pose detection)
mp_pose = mp.solutions.pose

# Open webcam
cap = cv2.VideoCapture(0)

# Variable used for FPS calculation
prev_time = 0

# Initialize MediaPipe Pose
with mp_pose.Pose() as pose:

    # Run loop while webcam is active
    while cap.isOpened():

        # Read frame from webcam
        ret, frame = cap.read()

        # Exit if frame is not captured
        if not ret:
            break

        # Convert BGR image to RGB
        # MediaPipe works with RGB images
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect body landmarks
        results = pose.process(image)

        # Convert image back to BGR
        # OpenCV displays images in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if pose landmarks are detected
        if results.pose_landmarks:

            # Draw body skeleton and landmark points
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Get all landmarks
            landmarks = results.pose_landmarks.landmark

            # Left shoulder landmark
            left_shoulder = landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ]

            # Left elbow landmark
            left_elbow = landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW.value
            ]

            # Left wrist landmark
            left_wrist = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST.value
            ]

            # Convert landmarks to x,y coordinates
            shoulder = [left_shoulder.x, left_shoulder.y]
            elbow = [left_elbow.x, left_elbow.y]
            wrist = [left_wrist.x, left_wrist.y]

            # Calculate elbow angle
            angle = calculate_angle(
                shoulder,
                elbow,
                wrist
            )

            # Print angle in terminal
            print("Left Elbow Angle:", int(angle))

        # Calculate FPS
        current_time = time.time()

        # FPS = Frames Processed Per Second
        fps = 1 / (current_time - prev_time)

        # Update previous time for next calculation
        prev_time = current_time

        # Display FPS on screen
        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show output window
        cv2.imshow("Pose Detection", image)

        # Press 'q' to quit program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam resources
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
