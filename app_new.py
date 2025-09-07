import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# 📁 CONFIGURATION
IMG_SIZE = 224  # Optimized for new trained models (98.33% accuracy)
CONFIDENCE_THRESHOLD = 0.8  # Lower than before since we expect better performance

# 🔲 ROI COORDINATES - EXACT MATCH with data collection
ROI_top = 80        # Same as data_capture.py
ROI_bottom = 320    # Same as data_capture.py  
ROI_left = 130      # Same as data_capture.py
ROI_right = 370     # Same as data_capture.py

# Define word dictionary
word_dict = {0:'Zero', 1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

# 🚀 LOAD NEW TRAINED MODELS
@st.cache_resource
def load_models():
    """Load the new trained models with caching"""
    try:
        # Load new high-performance models (98.33% accuracy)
        custom_cnn = load_model('new_custom_cnn.h5')
        mobilenet = load_model('new_mobilenet.h5') 
        print("✅ New trained models loaded successfully!")
        return custom_cnn, mobilenet, "NEW"
    except Exception as e:
        try:
            # Fallback to old models if new ones fail
            custom_cnn = load_model('custom_cnn.h5')
            mobilenet = load_model('custom_mobilenet.h5')
            st.warning("⚠️ Using old models - new models failed to load!")
            st.error(f"🔍 New model loading error: {str(e)}")
            return custom_cnn, mobilenet, "OLD_FALLBACK"
        except Exception as e2:
            st.error(f"❌ Error loading NEW models: {str(e)}")
            st.error(f"❌ Error loading OLD models: {str(e2)}")
            st.error("💡 Make sure these files exist:")
            st.code("new_custom_cnn.h5\nnew_mobilenet.h5")
            st.info("📋 Run training notebook if models are missing")
            st.stop()

# Load models with detailed feedback
with st.spinner('Loading models...'):
    custom_cnn, mobilenet, model_version = load_models()
    
# Model verification (silent - only show errors)
try:
    # Test models with dummy data
    test_input = np.random.random((1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    _ = custom_cnn.predict(test_input, verbose=0)
    _ = mobilenet.predict(test_input, verbose=0)
    # Models working correctly (silent)
except Exception as e:
    st.error(f"⚠️ Model verification failed: {str(e)}")
    st.stop()

def preprocess_roi_simple(frame, ROI_top, ROI_bottom, ROI_left, ROI_right):
    """
    🔥 EXACT MATCH preprocessing pipeline:
    Data Collection: 240x240 BGR raw → Training: resize+normalize+RGB → Inference: same
    """
    
    # 1. Extract ROI - EXACT SAME coordinates as data_capture.py
    roi = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]  # 240x240 BGR
    
    # 2. Resize to training size - SAME as ImageDataGenerator target_size
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)  # 224x224 BGR
    
    # 3. Convert BGR→RGB - SAME as ImageDataGenerator automatic conversion
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # 224x224 RGB
    
    # 4. Normalize [0,255]→[0,1] - SAME as ImageDataGenerator rescale=1./255
    roi_normalized = roi_rgb.astype(np.float32) / 255.0  # [0,1] range
    
    # 5. Add batch dimension - SAME as training expects
    roi_batch = np.expand_dims(roi_normalized, axis=0)  # (1, 224, 224, 3)
    
    return roi_batch, roi_resized

def simple_hand_detection(roi):
    """
    Simple hand detection based on content analysis
    Much faster than complex segmentation
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Check for sufficient contrast (hand vs background)
    contrast = np.std(gray)
    
    # Check for non-uniform regions (indicating hand presence)
    mean_intensity = np.mean(gray)
    
    # Simple thresholds based on typical hand characteristics
    has_hand = (contrast > 15 and 50 < mean_intensity < 200)
    
    # Calculate a simple confidence score
    hand_confidence = min(contrast / 30.0, 1.0) if has_hand else 0.0
    
    return has_hand, hand_confidence

def create_ensemble_prediction(roi_batch):
    """Create ensemble prediction from both models"""
    
    # Get predictions from both models
    pred_cnn = custom_cnn.predict(roi_batch, verbose=0)
    pred_mobilenet = mobilenet.predict(roi_batch, verbose=0)
    
    # Simple ensemble: average predictions
    ensemble_pred = (pred_cnn + pred_mobilenet) / 2.0
    
    # Get prediction details
    predicted_class = np.argmax(ensemble_pred)
    confidence = np.max(ensemble_pred)
    predicted_gesture = word_dict[predicted_class]
    
    # Get all class probabilities for analysis
    class_probs = ensemble_pred[0]
    
    return predicted_gesture, confidence, class_probs, predicted_class

# Streamlit app layout
st.title("🎯 ASL Hand Gesture Recognition - Enhanced")
# Status messages removed for cleaner UI

st.write("Real-time ASL digit recognition (0-9) with ensemble deep learning models")

# Configuration section removed for cleaner UI

# Placeholders for webcam feed and prediction
FRAME_WINDOW = st.image([])
PREDICTION_TEXT = st.empty()
METRICS_TEXT = st.empty()

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🚀 Start Camera", type="primary")
with col2:
    stop_button = st.button("⏹️ Stop", type="secondary")

# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Camera control
if start_button:
    st.session_state.camera_running = True
    
if stop_button:
    st.session_state.camera_running = False

# Main camera loop
if st.session_state.camera_running:
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Error: Could not open webcam")
        st.stop()
    
    # Performance metrics
    frame_count = 0
    total_inference_time = 0
    
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Error: Failed to capture frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        
        # Draw ROI rectangle (GREEN - matches data collection)
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (0, 255, 0), 3)
        
        # Extract and preprocess ROI with timing
        start_time = cv2.getTickCount()
        roi_batch, roi_display = preprocess_roi_simple(frame, ROI_top, ROI_bottom, ROI_left, ROI_right)
        
        # Simple hand detection (using raw ROI)
        roi_bgr = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]  # 240x240 raw BGR for hand detection
        has_hand, hand_confidence = simple_hand_detection(roi_bgr)
        
        prediction_text = "Position hand in green box"
        status_color = (255, 255, 255)  # White
        
        if has_hand and hand_confidence > 0.3:
            # Make prediction
            predicted_gesture, confidence, class_probs, predicted_class = create_ensemble_prediction(roi_batch)
            
            # Calculate inference time
            end_time = cv2.getTickCount()
            inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
            
            # Update metrics
            frame_count += 1
            total_inference_time += inference_time
            avg_inference_time = total_inference_time / frame_count
            
            # Determine prediction quality and display
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_text = f"✅ {predicted_gesture} ({confidence:.2f})"
                status_color = (0, 255, 0)  # Green
                
                # Show detailed metrics
                metrics_info = {
                    "🎯 Prediction": predicted_gesture,
                    "📊 Confidence": f"{confidence:.3f}",
                    "⚡ Inference": f"{inference_time:.1f}ms",
                    "📈 Avg Speed": f"{avg_inference_time:.1f}ms",
                    "🔍 Hand Detection": f"{hand_confidence:.2f}"
                }
                
                # Show top 3 predictions for analysis
                top3_indices = np.argsort(class_probs)[-3:][::-1]
                top3_text = " | ".join([f"{word_dict[i]}:{class_probs[i]:.2f}" for i in top3_indices])
                
                PREDICTION_TEXT.success(f"**{predicted_gesture}** (Confidence: {confidence:.3f})")
                METRICS_TEXT.info(f"🏆 Top 3: {top3_text} | ⚡ Speed: {inference_time:.1f}ms")
                
            elif confidence >= 0.5:  # Medium confidence
                prediction_text = f" {predicted_gesture} ({confidence:.2f})"
                status_color = (0, 165, 255)  # Orange
                PREDICTION_TEXT.warning(f"Medium confidence: **{predicted_gesture}** ({confidence:.3f})")
                METRICS_TEXT.info(f"⚡ Speed: {inference_time:.1f}ms")
                
            else:  # Low confidence
                prediction_text = f"Unclear ({confidence:.2f})"
                status_color = (0, 100, 255)  # Red-orange
                PREDICTION_TEXT.error("Low confidence - adjust hand position")
                METRICS_TEXT.info(f"⚡ Speed: {inference_time:.1f}ms")
        else:
            # No hand detected
            PREDICTION_TEXT.info("👋 Position your hand in the green box")
            METRICS_TEXT.info("🔍 Analyzing...")
            
        # Add prediction text to frame
        cv2.putText(frame_copy, prediction_text, (ROI_left, ROI_top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Add hand detection status
        hand_status = f"Hand: {hand_confidence:.2f}" if has_hand else "No hand"
        cv2.putText(frame_copy, hand_status, (ROI_left, ROI_bottom + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add ROI info
        roi_info = f"ROI: {ROI_right-ROI_left}x{ROI_bottom-ROI_top}"
        cv2.putText(frame_copy, roi_info, (ROI_left, ROI_bottom + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        
        # Update display
        FRAME_WINDOW.image(frame_rgb)
        
        # Break condition (Streamlit handles this)
        if not st.session_state.camera_running:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final performance summary
    if frame_count > 0:
        st.success(f"✅ Session complete! Avg inference time: {avg_inference_time:.1f}ms ({frame_count} frames)")

# Footer information removed for cleaner UI

# Performance status section removed for cleaner UI
