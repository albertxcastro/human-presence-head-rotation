import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

video_path = 0  # Use integer 0 instead of string '0'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open webcam. Try a different video_path value.")
    exit()

# Add a check to confirm webcam is opened
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame from webcam. Check if it's connected properly.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

# Real-time frame processing loop
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    input_frame = cv2.resize(frame, (128, 128))
    input_frame = input_frame.astype('float32') / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    model = load_model('../models/yaw_angle_classifier.keras')
    
    # Predict the yaw value
    yaw_value = model.predict(input_frame)

    # Extract the scalar value and format it
    yaw_scalar = yaw_value[0][0] if yaw_value.ndim > 1 else yaw_value[0]
    cv2.putText(frame, f"Yaw: {yaw_scalar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Yaw Prediction', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.2)

cap.release()
cv2.destroyAllWindows()
