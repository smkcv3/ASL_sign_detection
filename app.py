import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime


# Create directory for saving images
if not os.path.exists('Dataset/raw_webcam_images'):
    os.makedirs('Dataset/raw_webcam_images')

# Define word dictionary
word_dict = {0:'Zero', 1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

# Load the trained models
custom_cnn = load_model('custom_cnn.h5')
mobilenet = load_model('custom_mobilenet.h5')

# Streamlit app layout
st.title("Real-Time Hand Gesture Recognition")
st.write("This app uses an ensemble of a custom CNN and MobileNetV2 to recognize hand gestures (0-9) in real-time via webcam.")

# Placeholder for webcam feed and prediction
FRAME_WINDOW = st.image([])
PREDICTION_TEXT = st.empty()

# 🔥 FIXED ROI COORDINATES
ROI_top = 100
ROI_bottom = 300
ROI_left = 150      # FIXED: LEFT < RIGHT
ROI_right = 350     # FIXED: RIGHT > LEFT

# 🔥 ADVANCED hand segmentation function
def segment_hand_advanced(roi_rgb):
    """Advanced hand segmentation to preserve finger details"""
    original = roi_rgb.copy()
    
    # Method 1: Multi-range HSV skin detection (adaptive to lighting)
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    
    # Multiple skin tone ranges for better coverage
    skin_ranges = [
        ([0, 20, 70], [20, 255, 255]),      # Light skin
        ([0, 25, 80], [25, 255, 255]),      # Medium skin  
        ([0, 30, 60], [30, 255, 255]),      # Darker skin
        ([160, 20, 70], [180, 255, 255])   # Reddish skin tones
    ]
    
    # Combine all skin masks
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in skin_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Method 2: Add YCrCb color space for better skin detection
    ycrcb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YCrCb)
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combine HSV and YCrCb masks
    skin_mask = cv2.bitwise_or(combined_mask, ycrcb_mask)
    
    # 🔧 ADVANCED morphological operations to preserve finger details
    # Use smaller kernel to preserve finger boundaries
    kernel_small = np.ones((2,2), np.uint8)
    kernel_medium = np.ones((3,3), np.uint8)
    
    # Remove noise but preserve structure
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
    # Fill small gaps without losing finger separation
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # 🔍 Find largest contour (main hand)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Create clean mask from largest contour
        mask_clean = np.zeros(skin_mask.shape, dtype=np.uint8)
        cv2.fillPoly(mask_clean, [largest_contour], 255)
        skin_mask = mask_clean
    
    # Apply mask with edge preservation
    segmented = original.copy()
    segmented[skin_mask == 0] = [0, 0, 0]  # Black background
    
    return segmented, skin_mask

# 📊 Additional preprocessing to match training data better
def enhance_for_training_match(segmented_hand):
    """Additional preprocessing to better match training data characteristics"""
    # 1. Enhance contrast (training images likely had good contrast)
    enhanced = cv2.convertScaleAbs(segmented_hand, alpha=1.3, beta=5)
    
    # 2. Apply adaptive histogram equalization to hand regions only
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 3. Slight edge enhancement to improve finger definition
    # Create edge mask
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply subtle edge enhancement
    for i in range(3):
        enhanced[:,:,i] = cv2.addWeighted(enhanced[:,:,i], 0.9, edges, 0.1, 0)
    
    # 4. Apply slight Gaussian blur to reduce camera noise (but preserve edges)
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return enhanced

def preprocess_webcam_roi(frame, ROI_top, ROI_bottom, ROI_left, ROI_right):
    """🔥 ADVANCED preprocessing to EXACTLY match training data preprocessing"""
    
    # 1. Extract ROI (FIXED coordinates)
    roi = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]
    
    # 2. Convert BGR to RGB (OpenCV uses BGR, but training expects RGB)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # 3. 🔥 ADVANCED hand segmentation to preserve finger details
    segmented_hand, hand_mask = segment_hand_advanced(roi_rgb)
    
    # 4. 📊 Additional enhancement to match training characteristics
    enhanced_hand = enhance_for_training_match(segmented_hand)
    
    # 5. EXACT MATCH: Resize to (64, 64) like training target_size
    roi_resized = cv2.resize(enhanced_hand, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    
    # 6. EXACT MATCH: Apply MobileNetV2 preprocessing (same as training)
    roi_input = np.expand_dims(roi_resized, axis=0)  # Add batch dimension
    roi_input = tf.keras.applications.mobilenet_v2.preprocess_input(roi_input.astype(np.float32))
    
    return roi_input, hand_mask, roi_resized

def save_prediction_image(roi_resized, predicted_gesture, confidence, timestamp):
    """Save the preprocessed image with prediction info"""
    # Create filename: "prediction+_+timestamp.jpg"
    filename = f"{predicted_gesture.lower()}+_+{timestamp}.jpg"
    filepath = os.path.join('Dataset/raw_webcam_images', filename)
    
    # Convert RGB back to BGR for saving
    roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, roi_bgr)
    
    print(f"✅ PREDICTION SAVED: {predicted_gesture} (confidence: {confidence:.3f}) -> {filename}")
    return filepath

def analyze_prediction_quality(ensemble_pred, predicted_class, confidence):
    """Analyze prediction quality and provide insights"""
    # Calculate prediction entropy (measure of uncertainty)
    probs = ensemble_pred[0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Find top 3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    
    # Calculate margin (difference between top 2 predictions)
    margin = probs[top3_indices[0]] - probs[top3_indices[1]]
    
    # Quality assessment
    if confidence > 0.9 and margin > 0.3:
        quality = "EXCELLENT"
    elif confidence > 0.8 and margin > 0.2:
        quality = "GOOD"
    elif confidence > 0.7 and margin > 0.1:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    return {
        'quality': quality,
        'entropy': entropy,
        'margin': margin,
        'top3': [(word_dict[i], probs[i]) for i in top3_indices]
    }

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# Main loop for real-time prediction  
print("🚀 Starting IMPROVED Real-Time Hand Gesture Recognition...")
print("🔥 ENHANCED FEATURES:")
print("   • Advanced multi-range skin detection")
print("   • Finger detail preservation")
print("   • Training data matching enhancement")
print("   • Stricter quality thresholds")
print("📋 Confidence threshold: 0.85 (HIGH)")
print("📊 Hand detection: >2000px + >15% area")
print("📁 Images will be saved to: Dataset/raw_webcam_images/")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture frame.")
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # 🔥 IMPROVED PREPROCESSING - Exact match to training pipeline
    roi_input, hand_mask, roi_resized = preprocess_webcam_roi(
        frame, ROI_top, ROI_bottom, ROI_left, ROI_right
    )
    
    # 🔍 More sophisticated hand detection
    hand_pixels = np.sum(hand_mask)
    hand_area_ratio = hand_pixels / (hand_mask.shape[0] * hand_mask.shape[1])
    
    # 🎯 STRICTER thresholds for better reliability
    if hand_pixels > 2000 and hand_area_ratio > 0.15:  # More restrictive thresholds
        # Predict using both models (same as training)
        pred_cnn = custom_cnn.predict(roi_input, verbose=0)
        pred_mobilenet = mobilenet.predict(roi_input, verbose=0)
        
        # Ensemble: Average the predictions (same as training)
        ensemble_pred = (pred_cnn + pred_mobilenet) / 2
        predicted_class = np.argmax(ensemble_pred)
        predicted_gesture = word_dict[predicted_class]
        confidence = np.max(ensemble_pred)
        
        # 🎯 STRICTER CONFIDENCE-BASED LOGIC
        if confidence > 0.85:  # Much higher confidence threshold for reliability
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            
            # 📸 SAVE IMAGE with prediction info
            saved_path = save_prediction_image(roi_resized, predicted_gesture, confidence, timestamp)
            
            # 🔍 ADVANCED PREDICTION ANALYSIS
            analysis = analyze_prediction_quality(ensemble_pred, predicted_class, confidence)
            
            # 🖨️ PRINT DETAILED PREDICTION INFO
            print(f"🎯 PREDICTION: {predicted_gesture}")
            print(f"📊 CONFIDENCE: {confidence:.3f} ({analysis['quality']})")
            print(f"📏 MARGIN: {analysis['margin']:.3f} (top2 difference)")
            print(f"🌀 ENTROPY: {analysis['entropy']:.3f} (uncertainty)")
            print(f"🏆 TOP 3 PREDICTIONS:")
            for i, (gesture, prob) in enumerate(analysis['top3']):
                icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"   {icon} {gesture}: {prob:.3f}")
            print(f"📸 SAVED: {saved_path}")
            print("-" * 40)
            
            prediction_text = f"{predicted_gesture} ({confidence:.2f})"
            text_color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.75:  # Medium confidence range
            prediction_text = f"Medium: {predicted_gesture} ({confidence:.2f})"
            text_color = (0, 165, 255)  # Orange for medium confidence
            # Also print medium confidence predictions for analysis
            print(f"⚠️  MEDIUM CONFIDENCE: {predicted_gesture} ({confidence:.3f})")
            analysis = analyze_prediction_quality(ensemble_pred, predicted_class, confidence)
            print(f"   Quality: {analysis['quality']}, Margin: {analysis['margin']:.3f}")
            print(f"   Top 2: {analysis['top3'][0][0]}({analysis['top3'][0][1]:.3f}) vs {analysis['top3'][1][0]}({analysis['top3'][1][1]:.3f})")
        else:
            prediction_text = f"Low Confidence ({confidence:.2f})"
            text_color = (0, 100, 255)  # Red-orange for low confidence
    else:
        prediction_text = "No Hand Detected"
        text_color = (0, 0, 255)  # Red for no detection

    # Draw ROI rectangle and prediction on frame
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    cv2.putText(frame_copy, prediction_text, 
                (ROI_left, ROI_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    # 📊 Show enhanced detection status  
    status_text = f"Hand: {hand_pixels}px ({hand_area_ratio:.1%})"
    cv2.putText(frame_copy, status_text,
                (ROI_left, ROI_bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show segmentation quality indicator
    if hand_pixels > 2000 and hand_area_ratio > 0.15:
        quality_color = (0, 255, 0)  # Green - good
        quality_text = "✓ Good Detection"
    elif hand_pixels > 1000:
        quality_color = (0, 165, 255)  # Orange - fair
        quality_text = "⚠ Fair Detection" 
    else:
        quality_color = (0, 0, 255)  # Red - poor
        quality_text = "✗ Poor Detection"
    
    cv2.putText(frame_copy, quality_text,
                (ROI_left, ROI_bottom + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)

    # Convert frame to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

    # Update Streamlit UI
    FRAME_WINDOW.image(frame_rgb)
    PREDICTION_TEXT.write(f"Predicted Gesture: {prediction_text}")

cap.release()
cv2.destroyAllWindows()
print("🔚 Application stopped.")