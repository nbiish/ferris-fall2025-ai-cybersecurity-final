"""
HunyuanOCR Integration for document text extraction.

Purpose: Extract text from documents using Tencent's HunyuanOCR VLM
Inputs: Image/PDF file paths
Outputs: Extracted text with coordinates
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import torch
from PIL import Image


class HunyuanOCRProcessor:
    """
    Processor for HunyuanOCR document text extraction.
    
    HunyuanOCR is a 1B parameter VLM that excels at:
    - Complex multilingual document parsing
    - Text spotting
    - Open-field information extraction
    - Video subtitle extraction
    - Photo translation
    """
    
    MODEL_NAME = "tencent/HunyuanOCR"
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load_model(self) -> None:
        """Lazy load the HunyuanOCR model."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
            
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_NAME, 
                use_fast=False
            )
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                attn_implementation="eager",
                dtype=torch.bfloat16,
                device_map=self.device
            )
            self._loaded = True
        except ImportError as e:
            raise ImportError(
                f"Failed to load HunyuanOCR. Install transformers: {e}"
            )
    
    def process_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image and extract text.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt for extraction
            
        Returns:
            Dictionary with extracted text and metadata
        """
        self.load_model()
        
        # Default prompt for text extraction
        if prompt is None:
            prompt = "检测并识别图片中的文字，将文本坐标格式化输出。"
        
        # Load image
        image = Image.open(image_path)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]
        
        inputs = self.processor(
            text=texts,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        
        # Generate
        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=16384, 
                do_sample=False
            )
        
        # Decode
        input_ids = inputs.input_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        output_text = self._clean_output(
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        )
        
        return {
            "text": output_text,
            "source": image_path,
            "model": self.MODEL_NAME,
        }
    
    def _clean_output(self, text: str) -> str:
        """Clean repeated substrings in output text."""
        if isinstance(text, list):
            text = text[0] if text else ""
        
        n = len(text)
        if n < 8000:
            return text
        
        for length in range(2, n // 10 + 1):
            candidate = text[-length:]
            count = 0
            i = n - length
            while i >= 0 and text[i:i + length] == candidate:
                count += 1
                i -= length
            if count >= 10:
                return text[:n - length * (count - 1)]
        
        return text
    
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images."""
        return [self.process_image(path) for path in image_paths]
