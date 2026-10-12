"""
OCR Preprocessing Variants for License Plate Images

This module implements multiple preprocessing techniques for single license plates.
Each variant is independent and can be compared to find the best approach.

Do NOT mix variants - each should be tested independently.
"""

import cv2
import numpy as np


class PlatePreprocessor:
    """Multiple preprocessing variants for plate OCR"""
    
    @staticmethod
    def _upscale_if_small(image, threshold=200, factor=2):
        """
        Helper: Upscale image if smaller than threshold
        
        Args:
            image: Input image (grayscale or BGR)
            threshold: Width threshold in pixels
            factor: Upscale factor
            
        Returns:
            Potentially upscaled image
        """
        h, w = image.shape[:2]
        if w < threshold:
            new_w = int(w * factor)
            new_h = int(h * factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image
    
    @staticmethod
    def _to_bgr(image):
        """Convert grayscale to BGR if needed"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    @staticmethod
    def _to_gray(image):
        """Convert BGR to grayscale if needed"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def variant_baseline(plate_image):
        """
        VARIANT 1 — Baseline (Current Phase 1)
        
        Pipeline:
        BGR → Upscale → Bilateral Filter → CLAHE → BGR
        
        Args:
            plate_image: BGR color image
            
        Returns:
            Preprocessed BGR image ready for PaddleOCR
        """
        processed = plate_image.copy()
        
        # Upscale if small
        processed = PlatePreprocessor._upscale_if_small(processed)
        
        # Convert to grayscale
        gray = PlatePreprocessor._to_gray(processed)
        
        # Bilateral filter for noise removal
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Convert back to BGR
        result = PlatePreprocessor._to_bgr(enhanced)
        
        return result
    
    @staticmethod
    def variant_otsu_threshold(plate_image):
        """
        VARIANT 2 — Grayscale + Otsu Thresholding
        
        Pipeline:
        BGR → Grayscale → Upscale → Otsu Threshold → BGR
        
        Otsu's method automatically finds optimal threshold
        Good for high-contrast plates
        
        Args:
            plate_image: BGR color image
            
        Returns:
            Preprocessed BGR image ready for PaddleOCR
        """
        # Convert to grayscale
        gray = PlatePreprocessor._to_gray(plate_image)
        
        # Upscale if small
        gray = PlatePreprocessor._upscale_if_small(gray)
        
        # Apply Otsu's threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR (3-channel) for consistency
        result = PlatePreprocessor._to_bgr(binary)
        
        return result
    
    @staticmethod
    def variant_adaptive_threshold(plate_image):
        """
        VARIANT 3 — Grayscale + Adaptive Thresholding
        
        Pipeline:
        BGR → Grayscale → Upscale → Adaptive Threshold → BGR
        
        Adaptive threshold is better for varying lighting conditions
        Uses local neighborhood instead of global threshold
        
        Args:
            plate_image: BGR color image
            
        Returns:
            Preprocessed BGR image ready for PaddleOCR
        """
        # Convert to grayscale
        gray = PlatePreprocessor._to_gray(plate_image)
        
        # Upscale if small
        gray = PlatePreprocessor._upscale_if_small(gray)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to BGR
        result = PlatePreprocessor._to_bgr(binary)
        
        return result
    
    @staticmethod
    def variant_clahe_sharpening(plate_image):
        """
        VARIANT 4 — CLAHE + Sharpening
        
        Pipeline:
        BGR → Grayscale → Upscale → CLAHE → Sharpening → BGR
        
        Sharpening enhances edges and character boundaries
        Good for slightly blurry plates
        
        Args:
            plate_image: BGR color image
            
        Returns:
            Preprocessed BGR image ready for PaddleOCR
        """
        # Convert to grayscale
        gray = PlatePreprocessor._to_gray(plate_image)
        
        # Upscale if small
        gray = PlatePreprocessor._upscale_if_small(gray)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to BGR
        result = PlatePreprocessor._to_bgr(sharpened)
        
        return result
    
    @staticmethod
    def variant_denoise_clahe(plate_image):
        """
        VARIANT 5 — Denoising + CLAHE
        
        Pipeline:
        BGR → Grayscale → Upscale → Bilateral Denoise → CLAHE → BGR
        
        Bilateral filter reduces noise while preserving edges
        Combined with CLAHE for better contrast
        
        Args:
            plate_image: BGR color image
            
        Returns:
            Preprocessed BGR image ready for PaddleOCR
        """
        # Convert to grayscale
        gray = PlatePreprocessor._to_gray(plate_image)
        
        # Upscale if small
        gray = PlatePreprocessor._upscale_if_small(gray)
        
        # Bilateral filter (denoise)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Convert back to BGR
        result = PlatePreprocessor._to_bgr(enhanced)
        
        return result
    
    @staticmethod
    def get_all_variants(plate_image):
        """
        Apply all 5 preprocessing variants to a single plate image
        
        Args:
            plate_image: BGR color image (cropped plate)
            
        Returns:
            Dictionary with all variant results:
            {
                "baseline": processed_image,
                "otsu": processed_image,
                "adaptive": processed_image,
                "clahe_sharp": processed_image,
                "denoise_clahe": processed_image
            }
        """
        variants = {
            "baseline": PlatePreprocessor.variant_baseline(plate_image),
            "otsu": PlatePreprocessor.variant_otsu_threshold(plate_image),
            "adaptive": PlatePreprocessor.variant_adaptive_threshold(plate_image),
            "clahe_sharp": PlatePreprocessor.variant_clahe_sharpening(plate_image),
            "denoise_clahe": PlatePreprocessor.variant_denoise_clahe(plate_image)
        }
        return variants
    
    @staticmethod
    def get_variant_descriptions():
        """Return human-readable descriptions of all variants"""
        return {
            "baseline": "Upscale → Bilateral Filter → CLAHE",
            "otsu": "Grayscale → Upscale → Otsu Threshold",
            "adaptive": "Grayscale → Upscale → Adaptive Threshold",
            "clahe_sharp": "Grayscale → Upscale → CLAHE → Sharpening",
            "denoise_clahe": "Grayscale → Upscale → Bilateral Denoise → CLAHE"
        }
