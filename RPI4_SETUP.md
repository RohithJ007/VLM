# Raspberry Pi 4 Setup Guide

## System Requirements
- **Raspberry Pi 4 (4GB or 8GB RAM)** - 4GB minimum
- **64-bit Raspberry Pi OS** (required for PyTorch)
- **8GB+ microSD card** (16GB+ recommended)
- **Swap space:** At least 2GB enabled

---

## Quick Start for RPi4

### 1. Check Your System
```bash
# Verify 64-bit OS
uname -m
# Should show: aarch64

# Check available RAM
free -h

# Check swap space (should have at least 2GB)
swapon --show
```

### 2. Enable Swap (If Needed)
```bash
# Create 2GB swap file
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv python3-dev -y
sudo apt install libopenblas-dev libjpeg-dev zlib1g-dev -y

# For camera support
sudo apt install python3-opencv -y
```

### 4. Setup Virtual Environment
```bash
cd /path/to/Mini\ QWEN
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. Download Model (One-Time, Requires Internet)
```bash
# This will download ~4.5GB
python download_model.py
```

### 6. Run Application
```bash
# The app auto-detects RPi4 and enables 8-bit quantization
python main.py
```

---

## Performance Expectations

### Raspberry Pi 4 4GB:
- **Model loading:** ~60-90 seconds (first time)
- **Inference time:** 30-60 seconds per image
- **RAM usage:** ~3-3.5 GB
- **Mode:** 8-bit quantization (automatic)

### Raspberry Pi 4 8GB:
- **Model loading:** ~60-90 seconds (first time)
- **Inference time:** 25-45 seconds per image
- **RAM usage:** ~3-3.5 GB
- **Mode:** 8-bit quantization (automatic)

---

## Optimizations Applied

The system automatically detects Raspberry Pi and applies:
1. ✅ **8-bit quantization** - Better CPU compatibility than 4-bit
2. ✅ **Reduced token generation** - 256 tokens (vs 512 on PC)
3. ✅ **Memory limiting** - Max 3GB allocation
4. ✅ **CPU offloading** - Efficient layer management
5. ✅ **FP32 precision** - Better for ARM CPU

---

## Troubleshooting

### Out of Memory Errors:
```bash
# Increase swap space to 4GB
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Slow Performance:
- Close unnecessary applications
- Ensure active cooling (heatsink + fan)
- Consider overclocking (at your own risk)

### Camera Not Working:
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable

# Or use USB camera (recommended)
```

---

## Manual Quantization Control

If auto-detection doesn't work, edit `main.py`:

```python
# Line ~222: Force RPi4 mode
detector = MachineDefectDetector(use_quantization=False, use_8bit=True)
```

---

## Jetson Orin Nano Alternative

For better performance on edge devices, consider **Jetson Orin Nano**:
- **10x faster** than RPi4 (2-4 seconds per image)
- **GPU acceleration** with CUDA
- Same code works without modification

---

## Notes
- First run will be slower due to model compilation
- Subsequent runs are faster with cached optimizations
- Works **completely offline** after initial model download
- For production: Use Jetson Orin Nano or edge TPU accelerators
