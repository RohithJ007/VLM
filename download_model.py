"""
Model download script - Run this ONCE while online to download Qwen2-VL-2B
After download, the system works completely offline
"""

import os
from huggingface_hub import snapshot_download
from config import MODEL_NAME, MODEL_CACHE_DIR


def download_model():
    """Download Qwen2-VL-2B model for offline use"""
    
    print("=" * 60)
    print("Downloading Qwen2-VL-2B Model for Offline Use")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Cache directory: {MODEL_CACHE_DIR}")
    print(f"Size: ~4-5 GB")
    print()
    print("This is a ONE-TIME download. After completion, you can work offline.")
    print("=" * 60)
    print()
    
    # Create cache directory
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    try:
        print("[Download] Starting model download...")
        print("[Download] This may take 10-30 minutes depending on your connection")
        print()
        
        # Download model files
        model_path = snapshot_download(
            repo_id=MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            resume_download=True,
            local_files_only=False
        )
        
        print()
        print("=" * 60)
        print("✓ Model downloaded successfully!")
        print("=" * 60)
        print(f"Model location: {model_path}")
        print()
        print("Next steps:")
        print("1. You can now disconnect from the internet")
        print("2. Run the system: python main.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ Download failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space (~5 GB)")
        print("3. Try running again - download will resume from where it stopped")
        print("=" * 60)
        
        return False


if __name__ == "__main__":
    success = download_model()
    
    if not success:
        exit(1)
