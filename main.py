"""
Main application for offline machine defect detection
Integrates camera capture + Qwen2-VL-2B analysis
"""

import os
import sys
import time
from datetime import datetime
from camera import CameraCapture
from model import MachineInspectionModel
from config import TEMP_IMAGE_PATH


class MachineDefectDetector:
    """Complete machine inspection system"""
    
    def __init__(self, use_quantization=False, use_8bit=False):
        """
        Initialize the detection system
        
        Args:
            use_quantization: Use 4-bit quantization (recommended for systems with 8GB RAM)
            use_8bit: Use 8-bit quantization (recommended for RPi4 with 4GB RAM)
        """
        self.camera = CameraCapture()
        self.model = MachineInspectionModel(use_quantization=use_quantization, use_8bit=use_8bit)
        self.model_loaded = False
    
    def initialize(self):
        """Initialize camera and load model"""
        print("=" * 70)
        print("  MACHINE DEFECT DETECTION SYSTEM - OFFLINE MODE")
        print("=" * 70)
        print()
        
        # Initialize camera
        try:
            self.camera.initialize_camera()
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            return False
        
        # Load model
        try:
            self.model.load_model()
            self.model_loaded = True
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            print()
            print("Did you download the model?")
            print("Run: python download_model.py")
            return False
        
        print()
        print("=" * 70)
        print("  ✓ SYSTEM READY")
        print("=" * 70)
        print()
        
        return True
    
    def run_inspection(self, mode="detailed", capture_mode="manual"):
        """
        Run complete inspection: capture + analysis
        
        Args:
            mode: "detailed" or "quick" analysis
            capture_mode: "manual" (press SPACE) or "auto" (countdown)
            
        Returns:
            str: Analysis text
        """
        print("\n" + "=" * 70)
        print(f"  STARTING INSPECTION - {mode.upper()} MODE")
        print("=" * 70)
        print()
        
        # Step 1: Capture image
        print("[Step 1/2] Image Capture")
        print("-" * 70)
        
        if capture_mode == "auto":
            image_path = self.camera.capture_auto()
        else:
            image_path = self.camera.capture_image()
        
        if image_path is None:
            print("✗ Image capture failed or cancelled")
            return None
        
        print()
        
        # Step 2: Analyze image
        print("[Step 2/2] AI Analysis")
        print("-" * 70)
        
        try:
            start_time = time.time()
            
            analysis = self.model.inspect_machine(image_path, mode=mode)
            
            elapsed = time.time() - start_time
            
            print(f"[Model] Analysis completed in {elapsed:.1f} seconds")
            print()
            
            return analysis
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            return None
    
    def display_report(self, analysis):
        """Display formatted analysis report with fire/smoke alert highlighting"""
        if analysis is None:
            return
        
        # Check if fire/smoke/hazard detected in the analysis
        analysis_upper = analysis.upper()
        has_fire = any(word in analysis_upper for word in ["FIRE DETECTED", "FIRE HAZARD", "ACTIVE FIRE", "FLAMES"])
        has_smoke = any(word in analysis_upper for word in ["SMOKE DETECTED", "SMOKE:", "FUMES DETECTED"])
        has_critical = any(word in analysis_upper for word in ["CRITICAL HAZARD", "EVACUATE", "EMERGENCY"])
        
        print("=" * 70)
        print("  MACHINE CONDITION REPORT")
        print("=" * 70)
        
        # Display priority alert if fire/smoke/critical hazard
        if has_fire or has_smoke or has_critical:
            print()
            print("🚨" * 35)
            if has_fire:
                print("  ⚠️  FIRE HAZARD DETECTED - IMMEDIATE ACTION REQUIRED  ⚠️")
            elif has_smoke:
                print("  ⚠️  SMOKE DETECTED - CHECK IMMEDIATELY  ⚠️")
            elif has_critical:
                print("  ⚠️  CRITICAL HAZARD - URGENT ATTENTION NEEDED  ⚠️")
            print("🚨" * 35)
        
        print()
        print(analysis)
        print()
        print("=" * 70)
        print(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
    
    def run_continuous(self, mode="detailed"):
        """Run continuous inspection loop"""
        print("\n" + "=" * 70)
        print("  CONTINUOUS INSPECTION MODE")
        print("=" * 70)
        print()
        print("The system will continuously capture and analyze machines.")
        print("Press Ctrl+C to stop.")
        print()
        
        inspection_count = 0
        
        try:
            while True:
                inspection_count += 1
                print(f"\n### INSPECTION #{inspection_count} ###\n")
                
                analysis = self.run_inspection(mode=mode, capture_mode="manual")
                
                if analysis:
                    self.display_report(analysis)
                    
                    # Ask to continue
                    response = input("Continue? (y/n): ").strip().lower()
                    if response != 'y':
                        break
                else:
                    print("Inspection failed. Try again? (y/n): ")
                    response = input().strip().lower()
                    if response != 'y':
                        break
        
        except KeyboardInterrupt:
            print("\n\nInspection stopped by user")
        
        print(f"\nTotal inspections completed: {inspection_count}")
    
    def run_live_analysis(self, interval=10):
        """Run live video analysis with continuous terminal output"""
        import cv2
        
        print("\n" + "=" * 70)
        print("  LIVE VIDEO ANALYSIS MODE")
        print("=" * 70)
        print()
        print(f"Analyzing every {interval} seconds...")
        print("Video preview: Press 'q' or ESC to stop")
        print("Terminal output: Real-time analysis results")
        print()
        
        if self.camera.cap is None or not self.camera.cap.isOpened():
            self.camera.initialize_camera()
        
        analysis_count = 0
        last_analysis_time = time.time() - interval  # Analyze immediately on first frame
        
        print("[Live Analysis] Starting video stream...\n")
        
        try:
            while True:
                ret, frame = self.camera.cap.read()
                
                if not ret:
                    print("[Error] Failed to read frame")
                    break
                
                # Display live video with overlay
                display_frame = frame.copy()
                current_time = time.time()
                time_since_last = current_time - last_analysis_time
                
                # Add status overlay
                if time_since_last < interval:
                    status = f"Next analysis in: {interval - time_since_last:.1f}s"
                    color = (0, 255, 255)  # Yellow
                else:
                    status = "ANALYZING..."
                    color = (0, 255, 0)  # Green
                
                cv2.putText(display_frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_frame, f"Analyses: {analysis_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, "Press 'q' or ESC to exit", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('Live Machine Analysis', display_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[Live Analysis] Stopped by user")
                    break
                
                # Perform analysis at intervals
                if time_since_last >= interval:
                    analysis_count += 1
                    last_analysis_time = current_time
                    
                    # Save frame temporarily
                    cv2.imwrite(TEMP_IMAGE_PATH, frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Analyze
                    print("\n" + "=" * 70)
                    print(f"  ANALYSIS #{analysis_count} - {datetime.now().strftime('%H:%M:%S')}")
                    print("=" * 70)
                    
                    try:
                        analysis = self.model.inspect_machine(TEMP_IMAGE_PATH, mode="quick")
                        
                        # Check for fire/smoke/hazard in analysis
                        analysis_upper = analysis.upper()
                        if any(word in analysis_upper for word in ["FIRE DETECTED", "FIRE HAZARD", "SMOKE DETECTED", "CRITICAL HAZARD", "EVACUATE"]):
                            print("\n🚨🚨🚨 ALERT: FIRE/SMOKE/HAZARD DETECTED 🚨🚨🚨\n")
                        
                        print(analysis)
                        print("=" * 70 + "\n")
                    except Exception as e:
                        print(f"[Error] Analysis failed: {e}")
                        print("=" * 70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n[Live Analysis] Interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            print(f"\n[Live Analysis] Total analyses completed: {analysis_count}")
    
    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        self.camera.release()
        self.model.unload_model()
        
        # Remove temporary image
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
        
        print("✓ Cleanup complete")


def print_menu():
    """Display main menu"""
    print("\n" + "=" * 70)
    print("  MAIN MENU")
    print("=" * 70)
    print()
    print("  1. Single Inspection (Detailed Analysis)")
    print("  2. Single Inspection (Quick Analysis)")
    print("  3. Continuous Inspection Mode")
    print("  4. Live Video Analysis (Real-time)")
    print("  5. Fire & Smoke Safety Check")
    print("  6. Test Camera Only")
    print("  7. Exit")
    print()
    print("=" * 70)


def detect_raspberry_pi():
    """Detect if running on Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                return True
    except:
        pass
    return False

def main():
    """Main entry point"""
    
    # Check if model is downloaded
    from config import MODEL_CACHE_DIR
    if not os.path.exists(MODEL_CACHE_DIR):
        print("=" * 70)
        print("  MODEL NOT FOUND")
        print("=" * 70)
        print()
        print("The AI model has not been downloaded yet.")
        print()
        print("Steps:")
        print("1. Make sure you have internet connection")
        print("2. Run: python download_model.py")
        print("3. Wait for download to complete (~4-5 GB)")
        print("4. After download, you can work completely offline")
        print()
        print("=" * 70)
        return
    
    # Auto-detect platform and configure quantization
    is_rpi = detect_raspberry_pi()
    
    if is_rpi:
        print("\n[Platform] Raspberry Pi detected - enabling 8-bit quantization")
        detector = MachineDefectDetector(use_quantization=False, use_8bit=True)
    else:
        # For PC: Auto-detect based on available memory or let user configure
        # Set use_quantization=True if you have 8GB RAM or less
        import torch
        use_quant = False  # Change to True if you have limited GPU memory (8GB or less)
        use_8bit = not torch.cuda.is_available()  # Use 8-bit on CPU systems
        
        if use_8bit:
            print("\n[Platform] CPU detected - enabling 8-bit quantization for better performance")
        
        detector = MachineDefectDetector(use_quantization=use_quant, use_8bit=use_8bit)
    
    # Initialize system
    if not detector.initialize():
        print("\n✗ System initialization failed")
        return
    
    try:
        while True:
            print_menu()
            choice = input("Select option (1-7): ").strip()
            
            if choice == "1":
                # Detailed inspection
                analysis = detector.run_inspection(mode="detailed", capture_mode="manual")
                if analysis:
                    detector.display_report(analysis)
                    input("\nPress Enter to continue...")
            
            elif choice == "2":
                # Quick inspection
                analysis = detector.run_inspection(mode="quick", capture_mode="manual")
                if analysis:
                    detector.display_report(analysis)
                    input("\nPress Enter to continue...")
            
            elif choice == "3":
                # Continuous mode
                detector.run_continuous(mode="detailed")
            
            elif choice == "4":
                # Live video analysis
                print("\nLive Analysis Settings:")
                
                # Detect platform and suggest appropriate interval
                is_rpi = detect_raspberry_pi()
                default_interval = 60 if is_rpi else 10
                min_interval = 30 if is_rpi else 5
                
                if is_rpi:
                    print(f"[RPi4 Mode] Recommended interval: 60+ seconds (model needs ~30-60s)")
                
                try:
                    interval = input(f"Analysis interval in seconds (default {default_interval}): ").strip()
                    interval = int(interval) if interval else default_interval
                    if interval < min_interval:
                        print(f"Minimum interval is {min_interval} seconds (model processing time)")
                        interval = min_interval
                except ValueError:
                    interval = default_interval
                
                detector.run_live_analysis(interval=interval)
                input("\nPress Enter to continue...")
            
            elif choice == "5":
                # Fire & Smoke safety check
                print("\n🔥 FIRE & SMOKE SAFETY CHECK 🔥")
                print("=" * 70)
                print("Capturing image for safety inspection...")
                
                image_path = detector.camera.capture_image()
                if image_path:
                    print("\nAnalyzing for fire, smoke, and safety hazards...")
                    result = detector.model.check_fire_smoke(image_path)
                    
                    print("\n" + "=" * 70)
                    if result["has_hazard"]:
                        print(f"  ⚠️  {result['hazard_type']} DETECTED ⚠️")
                    else:
                        print(f"  ✓ SAFETY STATUS: {result['hazard_type']}")
                    print("=" * 70)
                    print()
                    print(result["analysis"])
                    print()
                    print("=" * 70)
                    
                input("\nPress Enter to continue...")
            
            elif choice == "6":
                # Test camera
                print("\nTesting camera...")
                try:
                    detector.camera.capture_image()
                    print("✓ Camera test complete")
                except Exception as e:
                    print(f"✗ Camera test failed: {e}")
                input("\nPress Enter to continue...")
            
            elif choice == "7":
                # Exit
                print("\nExiting...")
                break
            
            else:
                print("\n✗ Invalid option. Please select 1-7.")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        detector.cleanup()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
