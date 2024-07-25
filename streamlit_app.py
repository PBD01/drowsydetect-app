import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import pygame

# Load pre-trained model and cascade classifier
model_path = r'C:\Users\User\FYP final\4transfer_model.keras'
model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize pygame for alarm
pygame.mixer.init()
pygame.mixer.music.load('./alert.wav')

# Drowsiness detection function
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    drowsy = False
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye = roi_gray[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (24, 24))
            eye = eye / 255.0
            eye = eye.reshape(24, 24, 1)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)
            if prediction < 0.5:  # Adjust threshold as necessary
                drowsy = True
    return frame, drowsy

st.title("Drowsiness Detection App")

run = st.checkbox('Run Drowsiness Detection')

if run:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    alarm_on = False

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        frame, drowsy = detect_drowsiness(frame)
        st.image(frame, channels="BGR")

        if drowsy:
            if not alarm_on:
                alarm_on = True
                pygame.mixer.music.play(-1)
            st.write("**You are Drowsy!**")
        else:
            if alarm_on:
                alarm_on = False
                pygame.mixer.music.stop()

        if st.button('Stop'):
            run = False

    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Click the checkbox to start the drowsiness detection.")
