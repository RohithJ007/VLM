"""
System verification script - Check all components
"""

import sys
import os


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python_version():
    """Check Python version"""
    print("\n[1/7] Checking Python version...")
    version = sys.version_info
    print(f"    Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("    ✓ Python version OK")
        return True
    else:
        print("    ✗ Python 3.9+ required")
        return False


def check_packages():
    """Check required packages"""
    print("\n[2/7] Checking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'huggingface_hub': 'Hugging Face Hub'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"    ✓ {name}")
        except ImportError:
            print(f"    ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_camera():
    """Check camera availability"""
    print("\n[3/7] Checking camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"    ✓ Camera detected")
                print(f"    Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return True
            else:
                print("    ✗ Camera detected but cannot read frames")
                return False
        else:
            print("    ✗ No camera detected")
            print("    Check: Camera connected? Used by another app?")
            return False
            
    except Exception as e:
        print(f"    ✗ Camera check failed: {e}")
        return False


def check_model():
    """Check if model is downloaded"""
    print("\n[4/7] Checking AI model...")
    
    from config import MODEL_CACHE_DIR
    
    if os.path.exists(MODEL_CACHE_DIR):
        # Check if directory has content
        try:
            contents = os.listdir(MODEL_CACHE_DIR)
            if contents:
                print(f"    ✓ Model found in {MODEL_CACHE_DIR}")
                return True
            else:
                print(f"    ✗ Model directory empty")
                return False
        except:
            print(f"    ✗ Cannot access model directory")
            return False
    else:
        print(f"    ✗ Model not downloaded")
        print(f"    Run: python download_model.py")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n[5/7] Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free / (1024**3)
        print(f"    Free space: {free_gb:.1f} GB")
        
        if free_gb >= 5:
            print(f"    ✓ Sufficient space")
            return True
        else:
            print(f"    ⚠ Low disk space (need ~5GB for model)")
            return False
            
    except Exception as e:
        print(f"    ? Cannot check disk space: {e}")
        return True  # Don't fail on this


def check_memory():
    """Check system memory"""
    print("\n[6/7] Checking system memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"    Total RAM: {total_gb:.1f} GB")
        print(f"    Available: {available_gb:.1f} GB")
        
        if total_gb >= 8:
            print(f"    ✓ Sufficient memory")
            return True
        else:
            print(f"    ⚠ Low memory (8GB+ recommended)")
            print(f"    Tip: Enable quantization in main.py")
            return False
            
    except ImportError:
        print(f"    ? Cannot check memory (psutil not installed)")
        return True


def check_config():
    """Check configuration file"""
    print("\n[7/7] Checking configuration...")
    
    try:
        import config
        print(f"    Model: {config.MODEL_NAME}")
        print(f"    Camera index: {config.CAMERA_INDEX}")
        print(f"    Cache dir: {config.MODEL_CACHE_DIR}")
        print(f"    ✓ Configuration loaded")
        return True
    except Exception as e:
        print(f"    ✗ Configuration error: {e}")
        return False


def main():
    """Run all checks"""
    print_section("SYSTEM VERIFICATION")
    
    checks = [
        check_python_version(),
        check_packages(),
        check_camera(),
        check_model(),
        check_disk_space(),
        check_memory(),
        check_config()
    ]
    
    print_section("SUMMARY")
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total} checks")
    print()
    
    if checks[0] and checks[1] and checks[2]:  # Python, packages, camera OK
        print("✓ Core system ready!")
        print()
        
        if checks[3]:  # Model downloaded
            print("✓ AI model ready!")
            print()
            print("You can now run:")
            print("  python main.py")
        else:
            print("⚠ AI model not downloaded")
            print()
            print("Next step:")
            print("  python download_model.py  (requires internet)")
            print()
            print("Or test without AI:")
            print("  python demo.py  (camera + mock analysis)")
    else:
        print("✗ System not ready")
        print()
        
        if not checks[0]:
            print("- Install Python 3.9+")
        if not checks[1]:
            print("- Run: setup_venv.bat")
        if not checks[2]:
            print("- Connect camera and close other apps using it")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
