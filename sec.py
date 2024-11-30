import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands solution for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Store previous coordinates and time for speed calculation
prev_coords = None
prev_time = None  # Set initial prev_time as None

# Function to calculate Euclidean distance between two points
def calculate_distance(prev_coords, current_coords):
    return np.sqrt((current_coords[0] - prev_coords[0]) ** 2 + (current_coords[1] - prev_coords[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame for a mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Convert to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # If hands are detected, draw landmarks on the image
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract the coordinates of the index finger tip (landmark 8)
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Get current coordinates of the index finger tip
            current_coords = (index_finger_tip.x, index_finger_tip.y)

            # Get the current time to calculate time difference for speed
            current_time = time.time()  # Initialize current_time here

            # If we have previous coordinates, calculate distance and speed
            if prev_coords is not None and prev_time is not None:
                distance = calculate_distance(prev_coords, current_coords)

                time_diff = current_time - prev_time

                # Calculate speed (distance/time)
                speed = distance / time_diff if time_diff != 0 else 0

                # Display speed on the frame
                cv2.putText(frame, f"Speed: {speed:.2f} px/sec", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Print the speed to the console
                print(f"Speed: {speed:.2f} pixels/sec")

            # Update previous coordinates and time for the next frame
            prev_coords = current_coords
            prev_time = current_time  # Set the current_time as prev_time for the next iteration

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # If no hands are detected, display the original frame
    else:
        cv2.putText(frame, "No Hand Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with landmarks and speed
    cv2.imshow("Hand Movement Speed", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
