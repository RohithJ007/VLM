# Quick Start Guide

## Fast Setup (3 Steps)

### Step 1: Setup (Run Once)
```batch
setup_venv.bat
```
Wait 5-10 minutes for installation.

### Step 2: Download Model (Run Once - Requires Internet)
```batch
venv\Scripts\activate
python download_model.py
```
Downloads ~4-5 GB. Takes 10-30 minutes.

### Step 3: Run System
```batch
run.bat
```

## OR Use This Single Command After Setup:
```batch
venv\Scripts\activate
python main.py
```

## Camera Controls
- **SPACE** = Capture image
- **ESC** = Cancel

## Without Model Download (Test Camera Only)
```batch
venv\Scripts\activate
python camera.py
```

## Troubleshooting

### "Camera not found"
- Check camera is connected
- Try camera.py test script
- Change CAMERA_INDEX in config.py

### "Model not found"
- Run: `python download_model.py`
- Ensure internet connection
- Check disk space (~5GB needed)

### "Out of memory"
- Edit main.py: set `use_quant = True`
- Close other applications
- Restart computer

## System Files

- `main.py` - Main application ⭐
- `camera.py` - Camera capture
- `model.py` - AI model
- `config.py` - Settings
- `download_model.py` - Model downloader
- `run.bat` - Quick launcher

## For Raspberry Pi

1. Copy entire folder to Pi
2. Run on Pi:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Complete Documentation

See `README.md` for detailed information.
