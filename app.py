import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model

# Define word dictionary
word_dict = {0:'One', 1:'Ten', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

# Load the trained models
custom_cnn = load_model('custom_cnn.h5')
mobilenet = load_model('custom_mobilenet.h5')

# Streamlit app layout
st.title("Real-Time Hand Gesture Recognition")
st.write("This app uses an ensemble of a custom CNN and MobileNetV2 to recognize hand gestures (0-9) in real-time via webcam.")

# Placeholder for webcam feed and prediction
FRAME_WINDOW = st.image([])
PREDICTION_TEXT = st.empty()

# Webcam settings
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# Main loop for real-time prediction
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture frame.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Extract ROI
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    # Convert ROI to RGB and preprocess
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (64, 64))
    roi_input = np.reshape(roi_resized, (1, 64, 64, 3))
    roi_input = tf.keras.applications.mobilenet_v2.preprocess_input(roi_input)

    # Predict using both models
    pred_cnn = custom_cnn.predict(roi_input, verbose=0)
    pred_mobilenet = mobilenet.predict(roi_input, verbose=0)

    # Ensemble: Average the predictions
    ensemble_pred = (pred_cnn + pred_mobilenet) / 2
    predicted_gesture = word_dict[np.argmax(ensemble_pred)]

    # Draw ROI rectangle and display prediction on frame
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    cv2.putText(frame_copy, predicted_gesture, 
                (ROI_right, ROI_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert frame to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

    # Update Streamlit UI
    FRAME_WINDOW.image(frame_rgb)
    PREDICTION_TEXT.write(f"Predicted Gesture: {predicted_gesture}")

cap.release()
cv2.destroyAllWindows()