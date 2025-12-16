"""
Vision model module for machine defect detection
Handles Qwen2-VL-2B model loading and inference (offline capable)
"""

import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from config import (
    MODEL_NAME, 
    MODEL_CACHE_DIR, 
    MAX_NEW_TOKENS, 
    TEMPERATURE, 
    TOP_P,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    QUICK_INSPECTION_PROMPT,
    FIRE_SMOKE_PROMPT
)


class MachineInspectionModel:
    """Qwen2-VL-2B model for machine condition analysis"""
    
    def __init__(self, use_quantization=False, use_8bit=False):
        """
        Initialize the model
        
        Args:
            use_quantization: Use 4-bit quantization to reduce memory (recommended for 8GB RAM)
            use_8bit: Use 8-bit quantization for RPi4 (better CPU compatibility)
        """
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = use_quantization
        self.use_8bit = use_8bit
        
        print(f"[Model] Device: {self.device}")
        if self.use_8bit:
            print(f"[Model] Quantization: Enabled (8-bit - RPi4 optimized)")
        elif self.use_quantization:
            print(f"[Model] Quantization: Enabled (4-bit)")
    
    def load_model(self):
        """Load model from local cache (offline capable)"""
        print("[Model] Loading Qwen2-VL-2B...")
        print("[Model] This may take 1-2 minutes on first load...")
        
        try:
            # Check if model exists locally
            model_exists = os.path.exists(MODEL_CACHE_DIR)
            
            if not model_exists:
                print("[Model] ERROR: Model not found locally!")
                print("[Model] Please run 'python download_model.py' first while connected to internet")
                raise FileNotFoundError(f"Model not found in {MODEL_CACHE_DIR}")
            
            # Load processor
            print("[Model] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                local_files_only=True  # Force offline mode
            )
            
            # Load model with optional quantization
            print("[Model] Loading model weights...")
            
            if self.use_8bit:
                # 8-bit quantization for RPi4 (better CPU compatibility)
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_NAME,
                        cache_dir=MODEL_CACHE_DIR,
                        local_files_only=True,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        max_memory={"cpu": "3GB"}  # Limit memory for RPi4
                    )
                    print("[Model] Using 8-bit quantization (RPi4 optimized)")
                except Exception as e:
                    print(f"[Model] 8-bit quantization failed: {e}")
                    print("[Model] Falling back to standard loading...")
                    self.use_8bit = False
                    # Fall through to standard loading
            
            if self.use_quantization and not self.use_8bit:
                # 4-bit quantization for lower memory usage
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    local_files_only=True,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            
            if not self.use_quantization and not self.use_8bit:
                # Standard loading with GPU/CPU optimization
                if torch.cuda.is_available():
                    # GPU loading with FP16
                    print("[Model] Loading on CUDA GPU...")
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_NAME,
                        cache_dir=MODEL_CACHE_DIR,
                        local_files_only=True,
                        torch_dtype=torch.float16,
                        device_map="cuda"
                    )
                    self.device = "cuda"
                else:
                    # CPU loading with FP32
                    print("[Model] Loading on CPU...")
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_NAME,
                        cache_dir=MODEL_CACHE_DIR,
                        local_files_only=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        max_memory={"cpu": "3GB"}
                    )
                    self.model = self.model.to(self.device)
            
            self.model.eval()
            
            print("[Model] ✓ Model loaded successfully!")
            print(f"[Model] Memory usage: ~{self._get_model_memory():.1f} GB")
            
            return True
            
        except Exception as e:
            print(f"[Model] ✗ Failed to load model: {e}")
            raise
    
    def _get_model_memory(self):
        """Estimate model memory usage in GB"""
        if self.model is None:
            return 0
        
        try:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            total_bytes = param_size + buffer_size
            return total_bytes / (1024 ** 3)  # Convert to GB
        except:
            return 0
    
    def _validate_machine_image(self, image):
        """
        Validate that the image contains LARGE INDUSTRIAL machinery/equipment
        Excludes: PCBs, circuit boards, electronics, small devices
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {"is_machine": bool, "detected": str, "message": str}
        """
        validation_prompt = """Look at this image carefully and identify what is shown.

INDUSTRIAL MACHINES (YES):
- Electric motors (AC/DC motors)
- Pumps (centrifugal, hydraulic)
- Conveyor belt systems
- CNC machines, lathes, milling machines
- Drill presses, cutting machines
- Compressors, turbines, generators
- Industrial fans, blowers
- Gearboxes, drive systems
- Manufacturing equipment
- Heavy machinery, industrial robots

NOT INDUSTRIAL MACHINES (NO):
- Circuit boards, PCBs, microcontrollers
- ESP32, Arduino, electronics
- Computers, phones, tablets
- Small devices, gadgets
- Humans, animals, people
- Furniture, buildings, vehicles
- Food, nature, household items

Answer in ONE WORD: Is this a LARGE INDUSTRIAL MACHINE?
YES or NO"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": validation_prompt}
                ]
            }
        ]
        
        # Process inputs
        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate validation response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,  # Very low for precise classification
                do_sample=False
            )
        
        # Decode output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip().upper()
        
        print(f"[Model] Validation response: {response}")
        
        # Parse response
        is_machine = False
        detected = "Unknown"
        
        # First check for explicit YES/NO answer
        if "YES" in response[:50]:  # Check first 50 chars for clear answer
            is_machine = True
            detected = "Industrial Machine"
        elif "NO" in response[:50]:
            is_machine = False
            detected = "Not Industrial Machine"
        
        # Check for LARGE industrial machine keywords
        machine_keywords = ["MOTOR", "ENGINE", "PUMP", "CONVEYOR", "CNC", 
                          "DRILL", "LATHE", "MILLING", "COMPRESSOR", "TURBINE", 
                          "GEAR", "GENERATOR", "FAN", "BLOWER", "INDUSTRIAL MACHINE",
                          "MACHINERY", "EQUIPMENT"]
        
        # Check for non-machine keywords (electronics, PCBs, humans, etc.)
        invalid_keywords = ["CIRCUIT", "PCB", "BOARD", "ELECTRONICS", "ESP32", 
                          "ARDUINO", "MICROCONTROLLER", "CHIP", "PROCESSOR",
                          "HUMAN", "PERSON", "PEOPLE", "FACE", "MAN", "WOMAN",
                          "CHILD", "ANIMAL", "DOG", "CAT", "BIRD", "FOOD",
                          "NATURE", "TREE", "PLANT", "BUILDING", "HOUSE",
                          "PHONE", "COMPUTER", "LAPTOP", "TABLET", "GADGET"]
        
        response_lower = response.lower()
        
        # Only override YES if we find invalid keywords
        for keyword in invalid_keywords:
            if keyword.lower() in response_lower:
                is_machine = False
                detected = keyword.capitalize()
                break
        
        # If still not determined, check for machine keywords
        if not is_machine and detected == "Unknown":
            for keyword in machine_keywords:
                if keyword.lower() in response_lower:
                    is_machine = True
                    detected = "Machine/Equipment"
                    break
        
        message = "Image validated" if is_machine else "Not a machine or equipment"
        
        return {
            "is_machine": is_machine,
            "detected": detected,
            "message": message
        }
    
    def _validate_consistency(self, analysis):
        """
        Check for contradictions in the analysis and fix them
        
        Args:
            analysis: Raw analysis text from model
            
        Returns:
            str: Validated and corrected analysis
        """
        analysis_upper = analysis.upper()
        
        # First, check for explicit "no defects" statements
        no_defects_phrases = [
            "NO VISIBLE DEFECTS",
            "NO DEFECTS DETECTED",
            "NO DEFECTS",
            "CLEAN AND WELL-MAINTAINED",
            "WELL-MAINTAINED",
            "NO MAINTENANCE ACTIONS REQUIRED",
            "EXCELLENT CONDITION",
            "NO VISIBLE RUST",
            "NO VISIBLE LEAKS",
            "NO VISIBLE CRACKS",
            "NO VISIBLE WEAR",
            "NO VISIBLE DUST",
            "NO VISIBLE SPARKS",
            "NO VISIBLE SMOKE",
            "NO VISIBLE LOOSE WIRES"
        ]
        
        has_no_defects_statement = any(phrase in analysis_upper for phrase in no_defects_phrases)
        
        # Count how many "NO VISIBLE" statements there are
        no_visible_count = analysis_upper.count("NO VISIBLE")
        
        # If there are 5+ "NO VISIBLE" statements, it's clearly saying no defects
        if no_visible_count >= 5:
            has_no_defects_statement = True
            print(f"[Model] Detected {no_visible_count} 'NO VISIBLE' statements - equipment has no defects")
        
        # Look for actual defect mentions (phrases that indicate real problems)
        positive_defect_indicators = [
            "RUST: VISIBLE", "RUST VISIBLE", "RUST ON", "RUST DETECTED", "RUSTED", "RUSTING", "CORRODED", "CORROSION ON",
            "CRACK: VISIBLE", "CRACK VISIBLE", "CRACKS: VISIBLE", "CRACKS VISIBLE", "CRACKS AT", "CRACKED", "CRACKING",
            "LEAK: VISIBLE", "LEAK VISIBLE", "LEAKING", "LEAKAGE", "OIL LEAK", "FLUID LEAK",
            "WEAR: VISIBLE", "WEAR VISIBLE", "WORN", "WEARING", "WEAR ON",
            "TEAR: VISIBLE", "TEAR VISIBLE", "TORN", "TEARING",
            "DUST: VISIBLE", "DUST VISIBLE", "DUST ON", "DUST ACCUMULATION", "DUSTY", "DUST BUILDUP",
            "DAMAGE: VISIBLE", "DAMAGE VISIBLE", "DAMAGED", "BROKEN",
            "LOOSE WIRE: VISIBLE", "WIRE: LOOSE", "EXPOSED WIRE", "WIRING LOOSE", "WIRE EXPOSED"
        ]
        
        # Check if any actual defects are mentioned
        has_defect = False
        found_defects = []
        
        for defect_indicator in positive_defect_indicators:
            if defect_indicator in analysis_upper:
                # Make sure it's not in a "NO X" or "NOT VISIBLE" or "NO VISIBLE X" context
                pos = analysis_upper.find(defect_indicator)
                before_text = analysis_upper[max(0, pos-20):pos]
                
                # Only count if NOT preceded by "NO ", "NOT ", or phrases indicating absence
                if "NO " not in before_text and "NOT " not in before_text and "NO VISIBLE" not in before_text:
                    has_defect = True
                    found_defects.append(defect_indicator)
                    
        # If we found defects but also have "no defects" statements, prioritize the actual defects found
        if has_defect and has_no_defects_statement and len(found_defects) <= 2:
            # Likely a mixed message - recheck by counting defect vs no-defect ratio
            if no_visible_count >= len(found_defects) * 2:
                # More "NO VISIBLE" statements than defects found - likely no real issues
                print(f"[Model] Mixed message detected: {len(found_defects)} defects but {no_visible_count} 'NO VISIBLE' - likely false positive")
                has_defect = False
                found_defects = []
                    
        print(f"[Model] Defect check: has_defect={has_defect}, found={found_defects}, no_defects_statement={has_no_defects_statement}")
        
        # Check current rating
        has_good = "GOOD" in analysis_upper or "EXCELLENT" in analysis_upper
        has_poor = "POOR" in analysis_upper
        has_critical = "CRITICAL" in analysis_upper
        
        # Fix contradiction: defects mentioned but rated GOOD/EXCELLENT
        if has_defect and has_good and not has_poor and not has_critical:
            print(f"[Model] ⚠️ CONTRADICTION DETECTED: Found {found_defects} but rated GOOD/EXCELLENT")
            print("[Model] ✓ Auto-correcting to POOR rating")
            
            # Replace GOOD/EXCELLENT with POOR
            analysis = analysis.replace("EXCELLENT", "POOR")
            analysis = analysis.replace("Excellent", "POOR")
            analysis = analysis.replace("GOOD", "POOR")
            analysis = analysis.replace("Good", "POOR")
            analysis = analysis.replace("good condition", "POOR condition - requires maintenance")
            analysis = analysis.replace("Good condition", "POOR condition - requires maintenance")
            
        # Fix contradiction: no defects but rated POOR/CRITICAL
        if (not has_defect or has_no_defects_statement) and (has_poor or has_critical):
            print("[Model] ⚠️ CONTRADICTION DETECTED: No defects found but rated POOR/CRITICAL")
            print("[Model] ✓ Auto-correcting to GOOD rating")
            
            # Replace POOR/CRITICAL with GOOD
            analysis = analysis.replace("CRITICAL", "GOOD")
            analysis = analysis.replace("Critical", "GOOD") 
            analysis = analysis.replace("POOR", "GOOD")
            analysis = analysis.replace("Poor", "GOOD")
            analysis = analysis.replace("poor", "good")
        
        return analysis
    
    def inspect_machine(self, image_path, mode="detailed"):
        """
        Analyze machine image and return condition report
        
        Args:
            image_path: Path to machine image
            mode: "detailed" or "quick" inspection
            
        Returns:
            str: Text analysis of machine condition or error message
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"[Model] Analyzing image: {image_path}")
        print(f"[Model] Inspection mode: {mode}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Step 1: Validate that image contains a machine (not human/other)
            print("[Model] Step 1: Validating image content...")
            validation_result = self._validate_machine_image(image)
            
            if not validation_result["is_machine"]:
                return f"❌ NOT INDUSTRIAL MACHINE\n\nDetected: {validation_result['detected']}\n\nThis system is designed for LARGE INDUSTRIAL MACHINES only:\n- Motors, Pumps, Conveyors\n- CNC machines, Drills, Lathes\n- Compressors, Turbines, Generators\n\nPlease point camera at industrial equipment."
            
            print("[Model] ✓ Machine detected, proceeding with analysis...")
            
            # Step 2: Analyze machine condition
            # Prepare prompt based on mode
            if mode == "quick":
                user_prompt = QUICK_INSPECTION_PROMPT
            else:
                user_prompt = USER_PROMPT_TEMPLATE
            
            # Create conversation format for Qwen2-VL
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Process inputs
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response
            print("[Model] Generating analysis...")
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True if TEMPERATURE > 0 else False,
                    repetition_penalty=1.1
                )
            
            # Decode output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]
            
            analysis = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            print("[Model] ✓ Analysis complete")
            
            # Post-process to catch contradictions
            analysis = self._validate_consistency(analysis)
            
            return analysis.strip()
            
        except Exception as e:
            print(f"[Model] ✗ Analysis failed: {e}")
            raise
    
    def check_fire_smoke(self, image_path):
        """
        Quick fire and smoke safety check
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: {"has_hazard": bool, "hazard_type": str, "analysis": str}
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"[Fire/Smoke Check] Analyzing image: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Create fire/smoke detection prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": FIRE_SMOKE_PROMPT}
                    ]
                }
            ]
            
            # Process inputs
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,  # Very low for safety detection
                    do_sample=False
                )
            
            # Decode output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]
            
            analysis = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            # Parse response for hazard detection
            analysis_upper = analysis.upper()
            has_hazard = False
            hazard_type = "SAFE"
            
            # Check for fire
            if any(word in analysis_upper for word in ["FIRE", "FLAME", "BURNING", "IGNITION"]):
                has_hazard = True
                hazard_type = "FIRE"
            # Check for smoke
            elif any(word in analysis_upper for word in ["SMOKE", "FUMES"]):
                has_hazard = True
                hazard_type = "SMOKE"
            # Check for critical status
            elif "CRITICAL" in analysis_upper or "EVACUATE" in analysis_upper:
                has_hazard = True
                hazard_type = "CRITICAL HAZARD"
            # Check for hazard
            elif "HAZARD" in analysis_upper and "SAFE" not in analysis_upper:
                has_hazard = True
                hazard_type = "HAZARD"
            
            print(f"[Fire/Smoke Check] Status: {hazard_type}")
            
            return {
                "has_hazard": has_hazard,
                "hazard_type": hazard_type,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"[Fire/Smoke Check] ✗ Check failed: {e}")
            return {
                "has_hazard": False,
                "hazard_type": "ERROR",
                "analysis": f"Error during safety check: {e}"
            }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("[Model] Model unloaded")


def test_model():
    """Test model loading and basic inference"""
    print("=== Model Test ===")
    
    # Check if model is downloaded
    if not os.path.exists(MODEL_CACHE_DIR):
        print("✗ Model not found!")
        print("Please run: python download_model.py")
        return
    
    try:
        # Auto-detect: Use 8-bit for CPU (RPi4), 4-bit for GPU
        import torch
        is_cpu = not torch.cuda.is_available()
        
        if is_cpu:
            print("[Test] CPU detected - using 8-bit quantization (RPi4 mode)")
            model = MachineInspectionModel(use_quantization=False, use_8bit=True)
        else:
            print("[Test] GPU detected - using standard mode")
            model = MachineInspectionModel(use_quantization=False)
        
        model.load_model()
        print("✓ Model test successful!")
        model.unload_model()
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")


if __name__ == "__main__":
    test_model()
