"""
Camera capture module for machine defect detection
Handles camera access, preview, and image capture
Cross-platform: Windows, Linux, Raspberry Pi
"""

import cv2
import time
import platform
from datetime import datetime
from config import CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_DELAY, TEMP_IMAGE_PATH, IMAGE_QUALITY


def detect_platform():
    """Detect the current platform"""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        # Check if Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read() or 'BCM' in f.read():
                    return "rpi"
        except:
            pass
        return "linux"
    else:
        return "other"


class CameraCapture:
    """Handle camera operations for capturing machine images"""
    
    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.platform = detect_platform()
        
        print(f"[Camera] Platform detected: {self.platform}")
        
    def initialize_camera(self):
        """Initialize camera connection (cross-platform)"""
        print(f"[Camera] Initializing camera {self.camera_index}...")
        
        # Platform-specific camera initialization
        if self.platform == "windows":
            # Windows: Use DirectShow
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        elif self.platform == "rpi":
            # Raspberry Pi: Use V4L2 for better compatibility
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        else:
            # Linux/Other: Default backend
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        
        # For RPi: Additional settings for better performance
        if self.platform == "rpi":
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for RPi
        
        # Warm up camera
        time.sleep(1)
        for _ in range(5):
            self.cap.read()
        
        print(f"[Camera] Camera initialized successfully")
        return True
    
    def capture_image(self, save_path=TEMP_IMAGE_PATH, show_preview=True):
        """
        Capture image from camera with live preview
        
        Args:
            save_path: Path to save captured image
            show_preview: Show live preview window
            
        Returns:
            str: Path to saved image, or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            self.initialize_camera()
        
        print("[Camera] Starting live preview...")
        print("[Camera] Press SPACE to capture, ESC to cancel")
        
        captured_frame = None
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[Camera] ERROR: Failed to read frame")
                return None
            
            # Display preview if enabled
            if show_preview:
                # Add instruction overlay
                display_frame = frame.copy()
                
                # Adjust font size for RPi (smaller screen)
                font_scale = 0.5 if self.platform == "rpi" else 0.7
                thickness = 1 if self.platform == "rpi" else 2
                
                cv2.putText(display_frame, "SPACE = Capture | ESC = Cancel", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                cv2.putText(display_frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
                
                cv2.imshow('Machine Inspection Camera', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key
                captured_frame = frame.copy()
                print("[Camera] Image captured!")
                break
            elif key == 27:  # ESC key
                print("[Camera] Capture cancelled")
                cv2.destroyAllWindows()
                return None
        
        # Close preview window
        cv2.destroyAllWindows()
        
        if captured_frame is not None:
            # Save image
            success = cv2.imwrite(save_path, captured_frame, 
                                 [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
            
            if success:
                print(f"[Camera] Image saved to: {save_path}")
                return save_path
            else:
                print(f"[Camera] ERROR: Failed to save image")
                return None
        
        return None
    
    def capture_auto(self, countdown=CAPTURE_DELAY, save_path=TEMP_IMAGE_PATH):
        """
        Auto-capture after countdown (no manual trigger)
        
        Args:
            countdown: Seconds to wait before capture
            save_path: Path to save captured image
            
        Returns:
            str: Path to saved image, or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            self.initialize_camera()
        
        print(f"[Camera] Auto-capture in {countdown} seconds...")
        
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[Camera] ERROR: Failed to read frame")
                return None
            
            elapsed = time.time() - start_time
            remaining = max(0, countdown - elapsed)
            
            # Display countdown
            display_frame = frame.copy()
            
            # Adjust font for platform
            font_scale = 1.0 if self.platform == "rpi" else 1.5
            thickness = 2 if self.platform == "rpi" else 3
            
            if remaining > 0:
                cv2.putText(display_frame, f"Capturing in {remaining:.1f}s", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            else:
                cv2.putText(display_frame, "CAPTURED!", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            cv2.imshow('Machine Inspection Camera', display_frame)
            cv2.waitKey(1)
            
            if remaining <= 0:
                captured_frame = frame.copy()
                time.sleep(0.5)  # Brief pause to show "CAPTURED"
                break
        
        cv2.destroyAllWindows()
        
        # Save image
        success = cv2.imwrite(save_path, captured_frame, 
                             [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        
        if success:
            print(f"[Camera] Image saved to: {save_path}")
            return save_path
        else:
            print(f"[Camera] ERROR: Failed to save image")
            return None
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[Camera] Camera released")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()


def test_camera():
    """Test camera functionality"""
    print("=== Camera Test ===")
    camera = CameraCapture()
    
    try:
        camera.initialize_camera()
        image_path = camera.capture_image()
        
        if image_path:
            print(f"✓ Camera test successful! Image saved to: {image_path}")
        else:
            print("✗ Camera test failed")
    except Exception as e:
        print(f"✗ Camera test error: {e}")
    finally:
        camera.release()


if __name__ == "__main__":
    test_camera()
