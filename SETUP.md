# Setup Instructions for GitHub Contributors

## For New Users

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Mini QWEN"
```

### 2. Download Model Files Separately

**⚠️ IMPORTANT:** Model files (~4.5 GB) are NOT included in the repository due to size.

#### Option A: Automatic Download (Requires Internet)
```bash
# Activate virtual environment first
venv\Scripts\activate

# Download model
python download_model.py
```

#### Option B: Manual Download
1. Visit: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
2. Download all files
3. Place in: `models/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/[hash]/`

### 3. Install Dependencies
```bash
# Windows
venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify Setup
```bash
python verify_setup.py
```

## Project Structure

```
Mini QWEN/
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
├── SETUP.md                      # This file
├── requirements.txt              # Python dependencies
├── model.py                      # Core ML model
├── config.py                     # Configuration
├── download_model.py             # Model downloader
├── verify_setup.py               # Setup verification
├── venv/                         # Virtual env (NOT in git)
├── models/                       # Model files (NOT in git)
│   └── models--Qwen--Qwen2-VL-2B-Instruct/
└── pc_client/
    └── pc_client/
        └── web_app/
            ├── app.py            # Flask server
            └── templates/
                └── index.html    # Web UI
```

## Files NOT Tracked by Git

These are excluded via `.gitignore`:

- `venv/` - Virtual environment (recreate locally)
- `models/` - Large model files (download separately)
- `__pycache__/` - Python cache files
- `*.pyc`, `*.pyo` - Compiled Python
- `temp_*.jpg` - Temporary image files
- `.vscode/`, `.idea/` - IDE settings
- `.env` - Environment variables

## Running the Application

### Web Interface (Continuous Monitoring)
```bash
cd pc_client\pc_client\web_app
python app.py
```
Access at: http://127.0.0.1:5000

### CLI Analysis
```bash
python analyze.py --image path/to/image.jpg
```

## GPU Requirements

- **Minimum:** 6GB VRAM (NVIDIA GPU with CUDA)
- **Recommended:** 8GB+ VRAM (RTX 4060 or better)
- **CPU-only:** Possible but very slow (not recommended)

## Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model not found in models/
```
**Solution:** Download model files (see step 2)

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** 
- Close other applications using GPU
- Reduce `MAX_NEW_TOKENS` in `config.py`
- Enable quantization in `model.py`

### Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution:** Activate venv and reinstall:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m "Description"`
4. Push to branch: `git push origin feature-name`
5. Open Pull Request

## Model Download Details

- **Model Name:** Qwen/Qwen2-VL-2B-Instruct
- **Size:** ~4.5 GB
- **Format:** SafeTensors
- **Source:** Hugging Face
- **Files Needed:**
  - `model-00001-of-00002.safetensors`
  - `model-00002-of-00002.safetensors`
  - `config.json`
  - `tokenizer.json`
  - `processor_config.json`
  - And other config files

## License

[Add your license]

## Contact

[Add contact information]
