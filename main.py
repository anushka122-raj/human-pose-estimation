import cv2
import mediapipe as mp
import time  # used to calculate FPS (Frames Per Second)

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
