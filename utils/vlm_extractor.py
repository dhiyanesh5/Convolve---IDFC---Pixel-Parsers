"""
VLM Extractor - Qwen 2.5-VL (Disk Offload Enabled)
"""

import logging
from typing import Dict, Any
import numpy as np
from PIL import Image
import torch
import os

logger = logging.getLogger(__name__)

# Try to import strictly as per Qwen README
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

class VLMExtractor:
    def __init__(self, model_path: str = "models/qwen2.5-vl-3b"):
        logger.info("Initializing VLM Extractor (Disk Offload Mode)...")
        
        if not VLM_AVAILABLE:
            self.available = False
            return
        
        # FIX: Create an offload folder for the overflow weights
        offload_dir = os.path.join("models", "offload_weights")
        os.makedirs(offload_dir, exist_ok=True)
        
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading from: {model_path}")
            
            # 1. Load Model with DISK OFFLOAD enabled
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype="auto", 
                device_map="auto",
                low_cpu_mem_usage=True,
                offload_folder=offload_dir,  # <--- CRITICAL FIX
                offload_state_dict=True      # <--- CRITICAL FIX
            )
            
            # 2. Load Processor
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
            )
            
            self.available = True
            logger.info("✓ VLM Initialized (Disk Offload Active)")
            
        except Exception as e:
            logger.error(f"VLM Load Error: {e}")
            self.available = False
    
    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.available: return {}
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image[:, :, ::-1])
            
            fields = {}
            # Reduced prompts for speed
            queries = {
                'dealer_name': "Read the Dealer Name.",
                'model_name': "Read the Model Name.",
                'horse_power': "Read the HP value.",
                'asset_cost': "Read the Total Amount."
            }
            
            for field, prompt in queries.items():
                val = self._query(image, prompt)
                if val:
                    fields[field] = {'value': val, 'confidence': 0.85, 'source': 'vlm_qwen'}
            
            return fields
            
        except Exception as e:
            logger.error(f"Extraction Error: {e}")
            return {}

    def _query(self, image, prompt):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Query Error: {e}")
            return None