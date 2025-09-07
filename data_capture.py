import cv2
import os
import time
import argparse

# Dataset directory
DATASET_DIR = "Dataset_new"

# ROI coordinates (bigger than app.py for better data collection)
ROI_top = 80
ROI_bottom = 320
ROI_left = 130
ROI_right = 370

def capture_data(gesture_class, interval_seconds=5):
    """
    Simple data capture function
    
    Args:
        gesture_class (int): Gesture class number (0-9) - determines subfolder
        interval_seconds (int): Time interval between captures in seconds
    """
    
    # Create main dataset directory if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    # Create subfolder for this gesture class
    save_dir = os.path.join(DATASET_DIR, str(gesture_class))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"📸 CAPTURING DATA FOR GESTURE CLASS: {gesture_class}")
    print(f"⏰ Capture interval: {interval_seconds} seconds")
    print(f"📁 Saving to: {save_dir}")
    print(f"🎯 Target: 60 images (0.jpg to 59.jpg)")
    print(f"🔲 ROI size: {ROI_right-ROI_left}x{ROI_bottom-ROI_top} pixels")
    print("-" * 50)
    print("📋 INSTRUCTIONS:")
    print("   1. Position your hand inside the GREEN rectangle")
    print("   2. Show the gesture clearly")
    print("   3. Hold steady - auto-capture every few seconds")
    print("   4. Press 'q' to quit anytime")
    print("-" * 50)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("🚀 Starting capture in 3 seconds...")
    time.sleep(3)
    
    image_count = 0
    max_images = 60
    last_capture_time = time.time()
    
    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        
        # Draw ROI rectangle (GREEN for simplicity)
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (0, 255, 0), 3)
        
        # Add info text
        info_text = f"Gesture: {gesture_class} | Progress: {image_count}/{max_images}"
        cv2.putText(frame_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Countdown timer
        current_time = time.time()
        time_since_last = current_time - last_capture_time
        
        if time_since_last < interval_seconds:
            countdown = int(interval_seconds - time_since_last)
            countdown_text = f"Next capture in: {countdown}s"
            cv2.putText(frame_copy, countdown_text, (ROI_left, ROI_bottom + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Auto capture when time is up
        if time_since_last >= interval_seconds:
            # Extract ROI
            roi = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]
            
            # Save image
            filename = f"{image_count}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, roi)
            
            # Flash effect
            flash_frame = frame_copy.copy()
            cv2.rectangle(flash_frame, (0, 0), (frame_copy.shape[1], frame_copy.shape[0]), (255, 255, 255), -1)
            cv2.imshow('Data Capture', flash_frame)
            cv2.waitKey(100)
            
            # Console output
            print(f"📸 CAPTURED: {filename} | Size: {roi.shape[1]}x{roi.shape[0]} | Saved: {filepath}")
            
            image_count += 1
            last_capture_time = current_time
        
        # Show completion status
        if image_count >= max_images:
            cv2.putText(frame_copy, "✅ COMPLETED!", (ROI_left, ROI_top - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Data Capture', frame_copy)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n🛑 Capture stopped by user at {image_count}/{max_images}")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ CAPTURE COMPLETE!")
    print(f"📊 Total images captured: {image_count}/{max_images}")
    print(f"📁 Saved in: {save_dir}")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Simple ASL Data Capture')
    parser.add_argument('gesture_class', type=int, help='Gesture class number (0-9)')
    parser.add_argument('--interval', type=int, default=5, help='Capture interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Validate gesture class
    if not (0 <= args.gesture_class <= 9):
        print("❌ Error: Gesture class must be between 0-9")
        return
    
    # Validate interval
    if args.interval < 1:
        print("❌ Error: Interval must be at least 1 second")
        return
    
    # Start capture
    capture_data(args.gesture_class, args.interval)

if __name__ == "__main__":
    print("🎥 SIMPLE ASL DATA CAPTURE SYSTEM")
    print("=" * 40)
    main()