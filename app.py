import cv2
from deepface import DeepFace
import numpy as np
import streamlit as st

# Initialize Streamlit app
st.title("Emotion Recognition App")
st.text("Press 'Start' to begin emotion recognition")

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Initialize face detector (Haar Cascade or Dlib can be used)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

start_button = st.button('Start')
stop_button = st.button('Stop')

if start_button:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Count the number of faces
        people_count = len(faces)

        for (x, y, w, h) in faces:
            # Draw a rectangle around each face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the face from the frame for emotion detection
            face_region = frame[y:y+h, x:x+w]

            # Emotion detection
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            
            # Display emotion text on the face
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the people count on the frame
        cv2.putText(frame, f'People count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame in Streamlit
        st.image(frame, channels="BGR")

        # Stop if 'Stop' button is pressed
        if stop_button:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

