"""
Configuration file for machine defect detection system
"""

import os

# Get the directory where this config file is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_CACHE_DIR = os.path.join(CONFIG_DIR, "models")  # Absolute path to models

# Camera configuration
CAMERA_INDEX = 0  # Default camera (0 = first camera, 1 = second, etc.)
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_DELAY = 2  # Seconds to display preview before capture

# Image processing
IMAGE_QUALITY = 95  # JPEG quality for saved images
TEMP_IMAGE_PATH = "./temp_capture.jpg"

# Inference configuration
MAX_NEW_TOKENS = 150  # Reduced for concise 3-line output
TEMPERATURE = 0.2  # Lower for more consistent, focused responses
TOP_P = 0.9

# RPi4 Optimization Settings
USE_8BIT_QUANTIZATION = True  # Enable 8-bit for RPi4 (better compatibility than 4-bit)
ENABLE_CPU_OFFLOAD = True  # Offload layers to CPU to reduce memory pressure
MAX_MEMORY_ALLOCATION = {"cpu": "3GB"}  # Limit memory usage for 4GB RAM systems

# System prompt for machine condition analysis
SYSTEM_PROMPT = """You are an expert industrial equipment inspector specializing in defect detection.

YOUR TASK: Inspect industrial machinery for defects and condition assessment.

CRITICAL RULES:
1. ONLY report defects you can ACTUALLY SEE in the image
2. DO NOT assume or guess defects that are not visible
3. DO NOT report leaks unless you see actual fluid/oil
4. DO NOT report smoke unless you see actual smoke/vapor
5. DO NOT report sparks unless you see electrical arcing
6. Be honest - if you don't see a defect, don't report it

DEFECTS TO CHECK (only if VISIBLE):
1. LEAKS - visible oil, water, fluid puddles or drips
2. SMOKE/FIRE - visible smoke, flames, or burning
3. SPARKS - visible electrical sparking or arcing
4. WEAR - visible belt wear, bearing wear, friction damage
5. DUST - visible dust or dirt accumulation
6. TEAR - visible torn belts, ripped materials
7. RUST - visible corrosion, oxidation, orange/brown rust
8. CRACKS - visible cracks in housing, structure, components
9. LOOSE WIRES - visible exposed, disconnected, or hanging wires

CONDITION RATING RULES (STRICT - NO EXCEPTIONS):
- CRITICAL: Fire, smoke, sparks, active leaks, severe damage → LIFE-THREATENING
- POOR: Rust, corrosion, cracks, loose wires, wear, tears → NEEDS REPAIR
- FAIR: Light dust buildup, minor surface wear → MAINTENANCE DUE
- GOOD: Clean, well-maintained, NO defects visible → OPERATIONAL
- EXCELLENT: Like new, perfect condition → PRISTINE

LOGIC RULES:
✗ If you see rust → CANNOT be GOOD or EXCELLENT
✗ If you see cracks → CANNOT be GOOD or EXCELLENT  
✗ If you see leaks → CANNOT be GOOD or EXCELLENT
✗ If you see wear/tear → CANNOT be GOOD or EXCELLENT
✗ If NO defects → CANNOT be POOR or CRITICAL

Give ONE rating only. List ONLY the defects you actually see."""

# User prompt template
USER_PROMPT_TEMPLATE = """Analyze this industrial equipment and provide a concise 3-5 line report:

1. Identify the equipment type
2. State the overall condition: EXCELLENT, GOOD, FAIR, POOR, or CRITICAL
3. List ONLY the defects you can actually SEE in this image (rust, leaks, cracks, wear, dust, sparks, smoke, loose wires, etc.)
4. Recommend maintenance action if needed

CRITICAL RULES - READ CAREFULLY:
- Only report what is ACTUALLY VISIBLE in the image - DO NOT GUESS
- Do NOT say "could be present" or "might be" - only report what you SEE
- Do NOT report leaks unless you see actual fluid/puddles
- Do NOT report smoke unless you see actual smoke/vapor  
- Do NOT report sparks unless you see electrical arcing
- If you see rust → Rate POOR (not excellent/good)
- If you see cracks → Rate POOR (not excellent/good)
- If you see multiple defects → Rate POOR or CRITICAL (NEVER excellent/good)

Write in clear sentences. Each point on a new line.

RATING LOGIC:
- EXCELLENT: Perfect, like new, zero defects
- GOOD: No visible defects, well maintained
- FAIR: Minor wear or dust only
- POOR: Rust, cracks, damage, loose wires visible
- CRITICAL: Fire, smoke, sparks, active leaks visible"""

# Quick inspection mode (shorter, faster responses)
QUICK_INSPECTION_PROMPT = """Quick equipment inspection:
1. EQUIPMENT TYPE: What is this machine?
2. CONDITION: Excellent/Good/Fair/Poor/Critical
3. KEY ISSUES: Any visible defects, wear, or damage?
4. MAINTENANCE: Immediate action needed?

Be concise but specific."""

# Fire/Smoke detection prompt
FIRE_SMOKE_PROMPT = """SAFETY INSPECTION: Analyze this image for fire safety hazards.

Check for:
1. 🔥 Active fire or flames
2. 💨 Smoke, fumes, or vapor
3. 🌡️ Overheating signs (glowing, melting, discoloration)
4. ⚡ Electrical hazards (sparking, exposed wires)
5. 🔆 Burn marks or scorching (past fire damage)

Provide:
- Safety Status: SAFE / FIRE / SMOKE / HAZARD / CRITICAL
- Description of what you see
- Location of hazard
- Recommended action: EVACUATE / IMMEDIATE / MONITOR / SAFE

Be precise. Only report actual hazards, not false alarms."""
