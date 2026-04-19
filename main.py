import cv2
import time  # used for tracking time
import winsound  # used for beep sound (Windows)

# Haar Cascade is a pre-trained model used to detect faces in an image
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # used to open webcam

count = 0  # frame counter
start_time = None  # to track when person appears
person_present = False  # to control beep (avoid continuous sound)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1  # increase frame count

    # Convert to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale detects multiple faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 🏷️ Draw rectangle and label each detected face
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Label each face (Face 1, Face 2, ...)
        cv2.putText(frame, f"Face {i+1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Number of persons = number of faces detected
    num_persons = len(faces)

    # Draw background box to improve text visibility
    cv2.rectangle(frame, (5, 5), (350, 200), (50, 50, 50), -1)

    # Display different messages based on number of persons
    if num_persons == 0:
        # No person detected
        cv2.putText(frame, "No Person Detected", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Reset timer and beep state
        start_time = None
        person_present = False

    elif num_persons == 1:
        # 🔊 Beep only when person appears first time
        if not person_present:
            winsound.Beep(1000, 300)
            person_present = True

        # Start timer if not started
        if start_time is None:
            start_time = time.time()

        elapsed_time = int(time.time() - start_time)

        cv2.putText(frame, "1 Person Detected", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Time: {elapsed_time} sec", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    else:
        # Multiple persons detected
        cv2.putText(frame, "Multiple Persons Detected", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # Reset timer and beep state
        start_time = None
        person_present = False

    # Show number of detected persons
    cv2.putText(frame, f"Persons: {num_persons}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Show frame count
    cv2.putText(frame, f"Frame: {count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Person Count", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
