"""
Demo script - Test camera + placeholder analysis
Use this to test the complete flow WITHOUT downloading the 4GB model
"""

import time
from datetime import datetime
from camera import CameraCapture


def mock_analysis(image_path):
    """Simulate model analysis (for testing without model download)"""
    print("[Demo] Simulating AI analysis...")
    time.sleep(2)  # Simulate processing time
    
    analysis = """
MACHINE CONDITION REPORT
========================

Overall Condition: GOOD

Visual Inspection Results:
--------------------------
✓ Surface Condition: Minor wear visible on operating surfaces
✓ Structural Integrity: No cracks or fractures detected
✓ Corrosion Check: Light surface rust on mounting bracket (low severity)
✓ Fluid Systems: No visible leaks detected
✓ Component Status: All major components present and secured
✓ Contamination: Light dust accumulation on cooling fins

Detected Issues:
----------------
1. Minor surface rust on lower mounting bracket
   - Severity: LOW
   - Location: Bottom left mounting point
   - Recommendation: Schedule preventive maintenance

2. Dust buildup on cooling system
   - Severity: LOW
   - Location: Rear cooling fins
   - Recommendation: Clean during next service interval

Safety Assessment:
------------------
✓ No critical safety issues detected
✓ Machine safe for continued operation

Recommended Actions:
--------------------
1. Monitor rust progression - inspect again in 30 days
2. Schedule cleaning of cooling system
3. Apply rust inhibitor to mounting brackets
4. Continue normal operation with routine monitoring

Urgency: LOW - Schedule maintenance within 30 days

========================
This is a DEMO analysis. Real analysis requires model download.
"""
    
    return analysis


def main():
    """Run demo inspection"""
    print("=" * 70)
    print("  DEMO MODE - Camera + Mock Analysis")
    print("  (Real AI model not required for this demo)")
    print("=" * 70)
    print()
    
    # Initialize camera
    camera = CameraCapture()
    
    try:
        camera.initialize_camera()
        print()
        print("=" * 70)
        print("  Camera ready! Position machine in frame and press SPACE")
        print("=" * 70)
        print()
        
        # Capture image
        image_path = camera.capture_image()
        
        if image_path is None:
            print("\n✗ Capture cancelled")
            return
        
        print()
        print("=" * 70)
        print("  Analyzing machine condition...")
        print("=" * 70)
        print()
        
        # Mock analysis
        analysis = mock_analysis(image_path)
        
        # Display results
        print("\n" + "=" * 70)
        print("  INSPECTION RESULTS")
        print("=" * 70)
        print()
        print(analysis)
        print()
        print("=" * 70)
        print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        print("Demo complete!")
        print()
        print("Next steps:")
        print("1. To use REAL AI analysis, run: python download_model.py")
        print("2. After download, run: python main.py")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
    finally:
        camera.release()


if __name__ == "__main__":
    main()
