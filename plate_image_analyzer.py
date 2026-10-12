"""
Phase 2.4: Plate Image Analysis Component

Analyzes cropped plate images to extract measurable characteristics
that can be used for adaptive preprocessing variant routing.

Metrics calculated:
- Brightness: mean grayscale intensity
- Contrast: standard deviation of grayscale intensity
- Sharpness: variance of Laplacian
- Blur score: inverse of sharpness (0-100, where 0=sharp, 100=blurry)
- Plate size: width, height, area, aspect ratio
- Edge density: percentage of edge pixels

All metrics are normalized to 0-1 or 0-100 ranges for consistency.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class PlateImageAnalyzer:
    """Analyze cropped plate images for adaptive preprocessing routing."""
    
    @staticmethod
    def analyze_plate_image(plate_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze a cropped plate image and extract characteristics.
        
        Args:
            plate_image: Cropped plate image (BGR format, from OpenCV)
            
        Returns:
            Dictionary with image profile metrics:
            {
                "brightness": 0-255,
                "contrast": 0-127 (std dev of pixel values),
                "sharpness": 0+ (variance of Laplacian),
                "blur_score": 0-100 (0=sharp, 100=blurry),
                "width": int,
                "height": int,
                "area": int,
                "aspect_ratio": float,
                "edge_density": 0-1 (fraction of edge pixels),
                "dynamic_range": 0-255 (max - min pixel intensity)
            }
        """
        
        if plate_image is None or plate_image.size == 0:
            return PlateImageAnalyzer._empty_profile()
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        height, width = gray.shape
        
        # Calculate brightness (mean intensity)
        brightness = float(np.mean(gray))
        
        # Calculate contrast (std dev of intensity)
        contrast = float(np.std(gray))
        
        # Calculate sharpness (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))
        
        # Calculate blur score (0=sharp, 100=blurry)
        # Normalize sharpness to 0-100 scale
        # Very sharp images might have sharpness > 1000
        # Very blurry images might have sharpness < 10
        # We'll use a sigmoid-like scaling
        if sharpness < 10:
            blur_score = 100.0  # Very blurry
        elif sharpness > 1000:
            blur_score = 0.0    # Very sharp
        else:
            # Linear interpolation between 10 and 1000
            blur_score = 100.0 - ((sharpness - 10) / (1000 - 10)) * 100.0
        blur_score = max(0, min(100, blur_score))  # Clamp to 0-100
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Calculate dynamic range (max - min pixel intensity)
        dynamic_range = float(np.max(gray) - np.min(gray))
        
        # Size metrics
        area = width * height
        aspect_ratio = width / height if height > 0 else 0.0
        
        return {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
            "blur_score": round(blur_score, 2),
            "width": int(width),
            "height": int(height),
            "area": int(area),
            "aspect_ratio": round(aspect_ratio, 3),
            "edge_density": round(edge_density, 3),
            "dynamic_range": round(dynamic_range, 2)
        }
    
    @staticmethod
    def _empty_profile() -> Dict[str, float]:
        """Return empty/default image profile."""
        return {
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "blur_score": 100.0,
            "width": 0,
            "height": 0,
            "area": 0,
            "aspect_ratio": 0.0,
            "edge_density": 0.0,
            "dynamic_range": 0.0
        }
    
    @staticmethod
    def normalize_profile(profile: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize profile metrics to 0-1 range (except width/height/area/aspect_ratio).
        
        Useful for machine learning or statistical analysis.
        
        Args:
            profile: Image profile from analyze_plate_image()
            
        Returns:
            Profile with metrics normalized to 0-1 (except structural metrics)
        """
        normalized = profile.copy()
        
        # Normalize brightness to 0-1 (divide by 255)
        normalized["brightness"] = profile["brightness"] / 255.0
        
        # Normalize contrast to 0-1 (divide by 127, max std dev)
        normalized["contrast"] = min(1.0, profile["contrast"] / 127.0)
        
        # Sharpness is already handled by blur_score
        # Normalize sharpness to 0-1 (logarithmic scale)
        if profile["sharpness"] > 0:
            normalized["sharpness"] = np.log10(profile["sharpness"] + 1) / 4.0  # ~0-1
        else:
            normalized["sharpness"] = 0.0
        
        # Blur score is already 0-100, normalize to 0-1
        normalized["blur_score"] = profile["blur_score"] / 100.0
        
        # Dynamic range is already 0-255
        normalized["dynamic_range"] = profile["dynamic_range"] / 255.0
        
        # edge_density is already 0-1
        
        return normalized
    
    @staticmethod
    def get_profile_summary(profile: Dict[str, float]) -> str:
        """
        Get human-readable summary of image profile.
        
        Args:
            profile: Image profile from analyze_plate_image()
            
        Returns:
            Formatted string with profile details
        """
        summary = f"""
Image Profile:
  Brightness:      {profile['brightness']:.1f}/255 (mean intensity)
  Contrast:        {profile['contrast']:.1f} (std dev)
  Sharpness:       {profile['sharpness']:.1f} (Laplacian variance)
  Blur Score:      {profile['blur_score']:.1f}/100 (0=sharp, 100=blurry)
  Size:            {profile['width']}x{profile['height']} ({profile['area']} pixels)
  Aspect Ratio:    {profile['aspect_ratio']:.3f}
  Edge Density:    {profile['edge_density']:.1%}
  Dynamic Range:   {profile['dynamic_range']:.1f}/255
        """.strip()
        return summary
    
    @staticmethod
    def classify_lighting(brightness: float, contrast: float) -> str:
        """
        Classify image lighting conditions.
        
        Args:
            brightness: Mean pixel intensity (0-255)
            contrast: Standard deviation of pixel intensity
            
        Returns:
            Lighting condition description
        """
        if brightness < 50:
            if contrast < 30:
                return "VERY_DARK_LOW_CONTRAST"
            else:
                return "VERY_DARK_HIGH_CONTRAST"
        elif brightness < 100:
            if contrast < 30:
                return "DARK_LOW_CONTRAST"
            else:
                return "DARK_HIGH_CONTRAST"
        elif brightness < 150:
            if contrast < 30:
                return "MEDIUM_LOW_CONTRAST"
            else:
                return "MEDIUM_HIGH_CONTRAST"
        elif brightness < 200:
            if contrast < 30:
                return "BRIGHT_LOW_CONTRAST"
            else:
                return "BRIGHT_HIGH_CONTRAST"
        else:
            if contrast < 30:
                return "VERY_BRIGHT_LOW_CONTRAST"
            else:
                return "VERY_BRIGHT_HIGH_CONTRAST"
    
    @staticmethod
    def classify_focus(blur_score: float) -> str:
        """
        Classify image focus/sharpness.
        
        Args:
            blur_score: Blur score (0=sharp, 100=blurry)
            
        Returns:
            Focus condition description
        """
        if blur_score < 20:
            return "VERY_SHARP"
        elif blur_score < 40:
            return "SHARP"
        elif blur_score < 60:
            return "MEDIUM"
        elif blur_score < 80:
            return "BLURRY"
        else:
            return "VERY_BLURRY"


# Module-level convenience functions

def analyze_plate_image(plate_image: np.ndarray) -> Dict[str, float]:
    """Convenience function: analyze plate image."""
    return PlateImageAnalyzer.analyze_plate_image(plate_image)


def normalize_profile(profile: Dict[str, float]) -> Dict[str, float]:
    """Convenience function: normalize profile."""
    return PlateImageAnalyzer.normalize_profile(profile)


def get_profile_summary(profile: Dict[str, float]) -> str:
    """Convenience function: get profile summary."""
    return PlateImageAnalyzer.get_profile_summary(profile)


def classify_lighting(brightness: float, contrast: float) -> str:
    """Convenience function: classify lighting."""
    return PlateImageAnalyzer.classify_lighting(brightness, contrast)


def classify_focus(blur_score: float) -> str:
    """Convenience function: classify focus."""
    return PlateImageAnalyzer.classify_focus(blur_score)
