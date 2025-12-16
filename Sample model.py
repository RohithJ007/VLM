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
                # Standard loading with CPU optimization
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    local_files_only=True,
                    torch_dtype=torch.float32,  # FP32 for CPU
                    low_cpu_mem_usage=True,
                    max_memory={"cpu": "3GB"} if self.device == "cpu" else None
                )
                
                if self.device == "cpu":
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
        Validate that the image contains industrial machinery/equipment
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {"is_machine": bool, "detected": str, "message": str}
        """
        validation_prompt = """Look at this image and identify what is shown.

Answer in this EXACT format:
CATEGORY: [Human/Animal/Machine/Vehicle/Building/Food/Nature/Other]
CONFIDENCE: [High/Medium/Low]
DESCRIPTION: [brief 3-5 word description]

Be precise and honest about what you see."""
        
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
        
        # Parse response
        is_machine = False
        detected = "Unknown"
        
        # Check for machine/equipment keywords
        machine_keywords = ["MACHINE", "EQUIPMENT", "MOTOR", "ENGINE", "PUMP", 
                          "CONVEYOR", "CNC", "DRILL", "LATHE", "ROBOT", 
                          "COMPRESSOR", "TURBINE", "GEAR", "INDUSTRIAL"]
        
        # Check for non-machine keywords (humans, animals, etc.)
        invalid_keywords = ["HUMAN", "PERSON", "PEOPLE", "FACE", "MAN", "WOMAN",
                          "CHILD", "ANIMAL", "DOG", "CAT", "BIRD", "FOOD",
                          "NATURE", "TREE", "PLANT", "BUILDING", "HOUSE"]
        
        response_lower = response.lower()
        
        # Check if any machine keyword is present
        for keyword in machine_keywords:
            if keyword.lower() in response_lower:
                is_machine = True
                detected = "Machine/Equipment"
                break
        
        # Check if any invalid keyword is present
        for keyword in invalid_keywords:
            if keyword.lower() in response_lower:
                is_machine = False
                detected = keyword.capitalize()
                break
        
        # Parse category line if present
        if "CATEGORY:" in response:
            category_line = response.split("CATEGORY:")[1].split("\n")[0].strip()
            if "MACHINE" in category_line or "VEHICLE" in category_line:
                is_machine = True
                detected = category_line
            elif any(kw in category_line for kw in ["HUMAN", "ANIMAL", "FOOD", "NATURE"]):
                is_machine = False
                detected = category_line
        
        message = "Image validated" if is_machine else "Not a machine or equipment"
        
        return {
            "is_machine": is_machine,
            "detected": detected,
            "message": message
        }
    
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
                return f"❌ ERROR: {validation_result['message']}\n\nThis system is designed for MACHINE inspection only.\nDetected: {validation_result['detected']}"
            
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
                    do_sample=True if TEMPERATURE > 0 else False
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
