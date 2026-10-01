"""
PaddleOCR Module for License Plate Recognition
"""

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.WARNING)

class PlateOCR:
    """License plate OCR using PaddleOCR"""
    
    def __init__(self):
        """Initialize PaddleOCR"""
        try:
            print("[OCR] Initializing PaddleOCR...")
            # Disable mkldnn to avoid tensor conversion errors on Windows CPU
            # See: https://github.com/PaddlePaddle/PaddleOCR/discussions/13052
            self.ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)
            print("[OCR] PaddleOCR initialized successfully")
        except Exception as e:
            print(f"[OCR ERROR] Failed to initialize PaddleOCR: {e}")
            self.ocr = None
    
    def preprocess_plate(self, plate_image, upscale=True, upscale_factor=2):
        """
        Preprocess plate image for better OCR
        
        Args:
            plate_image: Extracted plate region (BGR)
            upscale: Whether to upscale the image before OCR
            upscale_factor: Factor to upscale by (default: 2x)
            
        Returns:
            Preprocessed image (BGR, 3-channel)
        """
        try:
            # Start with the input image
            processed = plate_image.copy()
            
            # Upscale if image is small and upscale flag is True
            if upscale:
                h, w = processed.shape[:2]
                # Only upscale if smaller than 200px width (reasonable threshold)
                if w < 200:
                    new_w = int(w * upscale_factor)
                    new_h = int(h * upscale_factor)
                    processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale for filtering
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed
            
            # Apply bilateral filter for noise removal
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            # Convert back to BGR (3-channel) for PaddleOCR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
        except Exception as e:
            print(f"[OCR WARNING] Preprocessing failed: {e}, using original image")
            return plate_image
    
    def recognize_text(self, plate_image):
        """
        Recognize text in plate using PaddleOCR
        
        Args:
            plate_image: Plate region image
            
        Returns:
            Dictionary with recognition results
        """
        if self.ocr is None:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": "OCR not initialized"
            }
        
        try:
            # Run OCR directly on original image
            result = self.ocr.ocr(plate_image)
            
            if not result or len(result) == 0:
                return {"success": False, "text": "", "confidence": 0.0}
            
            # In PaddleOCR 3.7, result is a list containing OCRResult object(s)
            ocr_result = result[0]
            
            # Extract recognized texts and scores from OCRResult dict
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [])
            
            # Combine all recognized text
            recognized_text = ' '.join(rec_texts) if rec_texts else ""
            
            # Calculate average confidence
            avg_confidence = np.mean(rec_scores) if rec_scores else 0.0
            success = len(recognized_text.strip()) > 0
            
            return {
                "success": success,
                "text": recognized_text.strip(),
                "confidence": round(float(avg_confidence), 4)
            }
        
        except Exception as e:
            print(f"[OCR ERROR] Recognition failed: {e}")
            return {"success": False, "text": "", "confidence": 0.0, "error": str(e)}
    
    def validate_plate_format(self, text):
        """
        Validate if text matches Indian plate format
        
        Args:
            text: Recognized text
            
        Returns:
            True if valid format
        """
        text = text.replace(" ", "").upper()
        
        if len(text) < 8:
            return False
        
        # Check for mix of letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        return has_letters and has_numbers
    
    def process_plate(self, image, bbox, return_crop=False):
        """
        Complete pipeline: extract -> preprocess -> recognize
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            return_crop: If True, also return the cropped plate image
            
        Returns:
            Dictionary with OCR results (and cropped image if return_crop=True)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract plate region
        plate_region = image[y1:y2, x1:x2]
        
        if plate_region.size == 0:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": "Could not extract plate region",
                "cropped_image": None
            }
        
        # Store original crop for UI display
        original_crop = plate_region.copy()
        
        # Preprocess for OCR
        preprocessed = self.preprocess_plate(plate_region, upscale=True, upscale_factor=2)
        
        # Recognize text
        ocr_result = self.recognize_text(preprocessed)
        
        if ocr_result.get("success"):
            # Validate format
            is_valid = self.validate_plate_format(ocr_result.get("text", ""))
            ocr_result["is_valid_format"] = is_valid
        else:
            ocr_result["is_valid_format"] = False
        
        # Optionally return cropped image
        if return_crop:
            ocr_result["cropped_image"] = original_crop
        
        return ocr_result
