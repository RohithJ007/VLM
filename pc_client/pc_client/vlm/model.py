import sys
import os
import cv2
import numpy as np
from PIL import Image

# Add the project root to path to import the main model
# Assuming this file is in c:\Mini QWEN\pc_client\pc_client\vlm\model.py
# We need to reach c:\Mini QWEN
current_file = os.path.abspath(__file__)
vlm_dir = os.path.dirname(current_file) # vlm
inner_pc_client = os.path.dirname(vlm_dir) # pc_client (inner)
outer_pc_client = os.path.dirname(inner_pc_client) # pc_client (outer)
project_root = os.path.dirname(outer_pc_client) # Mini QWEN

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from model import MachineInspectionModel
    print("Successfully imported MachineInspectionModel from root")
except ImportError as e:
    print(f"Error importing MachineInspectionModel: {e}")
    MachineInspectionModel = None

class VLMModel:
    def __init__(self):
        print("Initializing VLM Model...")
        self.model = None
        
        if MachineInspectionModel:
            try:
                # Initialize the real model
                # We use quantization=False by default for PC, or True if memory is low
                # The user is on PC, so we can probably use False or let it auto-detect
                self.model = MachineInspectionModel(use_quantization=False)
                self.model.load_model()
                print("Qwen Model loaded successfully.")
            except Exception as e:
                print(f"Error loading Qwen model: {e}")
        else:
            print("Warning: MachineInspectionModel class not found.")

    def analyze(self, frame):
        """
        Analyze the frame and return predictions.
        Args:
            frame: OpenCV image (numpy array)
        """
        if self.model:
            try:
                # The inspect_machine method expects a file path
                # We need to save the frame to a temp file
                temp_path = os.path.join(project_root, "temp_analysis.jpg")
                cv2.imwrite(temp_path, frame)
                
                # Run inspection
                # We use "detailed" mode by default
                result = self.model.inspect_machine(temp_path, mode="detailed")
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                return result
            except Exception as e:
                return f"Error during analysis: {e}"

        return "Error: Model not loaded"
