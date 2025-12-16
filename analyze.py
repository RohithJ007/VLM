"""
Command-line tool for analyzing images with the Qwen2-VL model
Usage:
    python analyze.py <image_path>
    python analyze.py --webcam
"""
import sys
import os
import cv2
import argparse
import torch

# Force CUDA if available
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("[WARNING] CUDA not available, running on CPU")

from model import MachineInspectionModel

def analyze_image(image_path):
    """Analyze a single image file"""
    print(f"Loading image: {image_path}")
    
    inspector = MachineInspectionModel(use_quantization=False)
    inspector.load_model()
    result = inspector.inspect_machine(image_path)
    
    print("\n" + "="*60)
    print("ANALYSIS RESULT:")
    print("="*60)
    print(result)
    print("="*60 + "\n")

def analyze_webcam():
    """Analyze frames from webcam - press 's' to capture and analyze, 'q' to quit"""
    print("Opening webcam... Press 's' to capture and analyze, 'q' to quit")
    
    # Try DirectShow backend for better Windows compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    inspector = MachineInspectionModel(use_quantization=False)
    inspector.load_model()
    print(f"Model loaded on: {inspector.device}")
    print("Ready to analyze.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam - Press S to analyze, Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("\nCapturing and analyzing...")
            # Save temporary file
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, frame)
            
            result = inspector.inspect_machine(temp_path)
            
            print("\n" + "="*60)
            print("ANALYSIS RESULT:")
            print("="*60)
            print(result)
            print("="*60 + "\n")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze images using Qwen2-VL for machine inspection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py image.jpg          # Analyze a single image
  python analyze.py --webcam           # Use webcam (press 's' to capture)
        """
    )
    
    parser.add_argument('image_path', nargs='?', help='Path to the image file to analyze')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for live analysis')
    
    args = parser.parse_args()
    
    if args.webcam:
        analyze_webcam()
    elif args.image_path:
        analyze_image(args.image_path)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
