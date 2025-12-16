import cv2
import time
import sys
import os

# Add the current directory to path so we can import the vlm folder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add the parent directory to path to access the Qwen model (once moved)
sys.path.append(os.path.dirname(current_dir))

from vlm.model import VLMModel

# ==========================================
# CONFIGURATION
# ==========================================
# REPLACE THIS with your actual Pinggy URL from the Raspberry Pi
# Example: "https://random-name.pinggy.link/video_feed"
STREAM_URL = "https://zntkf-103-114-209-107.a.free.pinggy.link/video_feed"
# ==========================================

def main():
    # 1. Initialize the VLM Model (from the separate folder)
    print("Loading VLM Model...")
    model = VLMModel()
    print("Model loaded.")

    # 2. Connect to the Video Stream
    print(f"Connecting to stream: {STREAM_URL}", flush=True)
    try:
        cap = cv2.VideoCapture(STREAM_URL)
    except Exception as e:
        print(f"Exception while connecting: {e}", flush=True)
        return

    if not cap.isOpened():
        print("Error: Could not open video stream. Check the URL and your internet connection.", flush=True)
        return

    print("Stream started. Press 'q' to quit.", flush=True)

    # 3. Main Loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Stream interrupted. Retrying...")
            time.sleep(1)
            cap.open(STREAM_URL)
            continue

        # -------------------------------------------------
        # FRONTEND / VISUALIZATION
        # -------------------------------------------------
        
        # Run Model Inference
        prediction = model.analyze(frame)

        # Draw UI on the frame (The "Frontend")
        # Add a nice header bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
        
        # Display Prediction
        cv2.putText(frame, f"VLM Output: {prediction}", (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame in a window
        cv2.imshow('PC Client - VLM Stream Receiver', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
