# Machine Defect Detection System - Offline Mode

Complete offline machine inspection system using Qwen2-VL-2B vision AI model. Captures images from camera and provides detailed technical analysis of machine condition, defects, and maintenance needs.

## Features

✅ **Completely Offline** - Works without internet after initial setup
✅ **Real-time Camera Capture** - Live preview and capture
✅ **AI-Powered Analysis** - Qwen2-VL-2B vision-language model
✅ **Defect Detection** - Rust, cracks, wear, leaks, surface defects, dust
✅ **Text-Only Output** - Focused machine condition reports
✅ **Quick & Detailed Modes** - Choose inspection depth
✅ **Continuous Inspection** - Process multiple machines

## System Requirements

### Windows PC (for initial setup and testing)
- **OS:** Windows 10/11
- **RAM:** 16GB recommended (8GB minimum with quantization)
- **Storage:** 10GB free space (model + dependencies)
- **Python:** 3.9, 3.10, or 3.11
- **Camera:** USB webcam or integrated camera
- **GPU:** Optional (NVIDIA with CUDA for faster inference)

### Raspberry Pi 5 (future deployment)
- **RAM:** 16GB model recommended
- **Storage:** 16GB+ SD card
- **Camera:** USB webcam or Raspberry Pi Camera Module

## Installation & Setup

### Step 1: Install Python
Download and install Python 3.9-3.11 from [python.org](https://www.python.org/downloads/)

**Important:** Check "Add Python to PATH" during installation

### Step 2: Setup Virtual Environment
Run the setup script to create virtual environment and install dependencies:

```batch
setup_venv.bat
```

This will:
- Create a virtual environment (`venv` folder)
- Install all required packages
- Take 5-10 minutes depending on your connection

### Step 3: Download AI Model (ONE-TIME, requires internet)
Activate the virtual environment and download the model:

```batch
venv\Scripts\activate
python download_model.py
```

**Download details:**
- Size: ~4-5 GB
- Time: 10-30 minutes (depends on connection speed)
- Downloads to `./models` folder
- Only needed ONCE
- After this, system works completely offline

### Step 4: Run the System
```batch
venv\Scripts\activate
python main.py
```

## Usage

### Main Menu Options

1. **Single Inspection (Detailed)** - Comprehensive machine analysis
2. **Single Inspection (Quick)** - Fast defect check
3. **Continuous Inspection** - Process multiple machines in sequence
4. **Test Camera** - Verify camera functionality
5. **Exit** - Close application

### Camera Operation

**Manual Capture Mode:**
- Live preview window opens
- Position machine in frame
- Press **SPACE** to capture
- Press **ESC** to cancel

**Auto Capture Mode:**
- Countdown timer (default 2 seconds)
- Automatic capture after countdown

### Understanding the Output

The system provides text-only machine condition reports including:

1. **Overall Condition:** Excellent/Good/Fair/Poor/Critical
2. **Specific Defects Detected:**
   - Surface defects (scratches, dents, deformations)
   - Corrosion and rust (location and severity)
   - Wear and tear (belts, gears, bearings)
   - Cracks or fractures
   - Leaks (oil, fluid, air)
   - Dust or contamination buildup
   - Loose or missing components
   - Thermal damage or discoloration
3. **Severity Assessment:** Low/Medium/High/Critical
4. **Recommended Actions:** Immediate repair/Schedule maintenance/Monitor/No action

**Note:** The system focuses ONLY on the machine itself, not the background or environment.

## Configuration

Edit `config.py` to customize:

```python
# Camera settings
CAMERA_INDEX = 0              # Change if you have multiple cameras
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Model settings
MAX_NEW_TOKENS = 512          # Longer = more detailed analysis
TEMPERATURE = 0.3             # Lower = more focused, higher = creative
```

## Performance Optimization

### For 8GB RAM Systems
Edit `main.py` and set:
```python
use_quant = True  # Enable 4-bit quantization
```

### Expected Performance (Windows PC)
- **With GPU (NVIDIA):** ~2-4 seconds per analysis
- **CPU only (16GB RAM):** ~8-12 seconds per analysis
- **CPU with quantization (8GB RAM):** ~12-18 seconds per analysis

### Expected Performance (Raspberry Pi 5 - 16GB)
- **Standard mode:** ~8-15 seconds per analysis
- **Optimized mode:** ~5-10 seconds per analysis

## Troubleshooting

### Camera not detected
```batch
# Test camera separately
python camera.py
```
- Check camera is connected
- Try changing `CAMERA_INDEX` in `config.py`
- Make sure no other app is using the camera

### Model not found error
```batch
# Re-download model
python download_model.py
```

### Out of memory error
- Enable quantization: `use_quant = True` in `main.py`
- Close other applications
- Reduce image resolution in `config.py`

### Slow inference
- First inference is always slower (model loading)
- Subsequent analyses are faster
- Consider enabling GPU support if available

## File Structure

```
Mini Florense/
├── main.py              # Main application
├── camera.py            # Camera capture module
├── model.py             # AI model inference
├── config.py            # Configuration settings
├── download_model.py    # Model download script
├── requirements.txt     # Python dependencies
├── setup_venv.bat       # Setup script
├── README.md            # This file
├── venv/                # Virtual environment (created by setup)
├── models/              # Downloaded AI model (created by download)
└── temp_capture.jpg     # Temporary captured image
```

## Raspberry Pi 5 Deployment (Future)

After testing on Windows, transfer to Raspberry Pi:

1. **Copy entire project folder** to Raspberry Pi
2. **On Raspberry Pi, run:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Copy the downloaded model folder** from Windows to Pi:
   - Copy `models/` folder to same location on Pi
   - Or re-download on Pi if you have good internet

4. **Run the system:**
```bash
python main.py
```

**Pi-specific optimizations:**
- Always use quantization mode (`use_quant = True`)
- Consider lower resolution: `CAPTURE_WIDTH = 640`, `CAPTURE_HEIGHT = 480`
- Use Raspberry Pi Camera Module for better performance than USB webcam

## Advanced Usage

### Batch Processing
Process multiple images from a folder (modify `main.py`):
```python
# Process folder of images
import glob
for img_path in glob.glob("images/*.jpg"):
    analysis = model.inspect_machine(img_path, mode="detailed")
    print(f"\n{img_path}:\n{analysis}\n")
```

### Custom Prompts
Edit `SYSTEM_PROMPT` in `config.py` to focus on specific defects:
```python
SYSTEM_PROMPT = """Focus only on detecting rust and corrosion..."""
```

### Integration with Other Systems
Export results as JSON, CSV, or send to database by modifying the output section in `main.py`.

## Support & Resources

- **Qwen2-VL Documentation:** https://github.com/QwenLM/Qwen2-VL
- **PyTorch Documentation:** https://pytorch.org/docs/
- **OpenCV Documentation:** https://docs.opencv.org/

## License

This project uses open-source models and libraries. Check individual component licenses for commercial use.

## Version History

- **v1.0.0** - Initial release with offline capability
  - Qwen2-VL-2B integration
  - Camera capture system
  - Detailed and quick inspection modes
  - Windows support

## Next Steps

1. ✅ Run `setup_venv.bat`
2. ✅ Run `python download_model.py` (one-time, requires internet)
3. ✅ Run `python main.py`
4. 📸 Start inspecting machines!

---

**Important:** After model download, you can disconnect from internet and the system will work completely offline.
