"""
Phase 2.2: Confidence Thresholding & Variant Selection

This module implements:
1. Configurable confidence threshold (default: 0.85)
2. Variant selection logic (choose highest-confidence result)
3. Result status codes (ACCEPTED, LOW_CONFIDENCE, OCR_FAILED)
4. Structured result format (selected_result + all_variants)

Important:
- Does NOT implement OCR text correction (Phase 2.3)
- Does NOT implement ensemble voting (Phase 2.4+)
- Highest confidence is only the selection mechanism, not correctness proof
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
from indian_plate_validator import validate_plate
try:
    import config
except ImportError:
    config = None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ResultStatus(Enum):
    """Result status codes for OCR recognition"""
    ACCEPTED = "ACCEPTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OCR_FAILED = "OCR_FAILED"
    DETECTION_FAILED = "DETECTION_FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    LOW_RESOLUTION = "LOW_RESOLUTION"


class ThresholdConfig:
    """Configurable confidence threshold settings"""
    
    # Primary threshold: results must meet or exceed this confidence
    OCR_CONFIDENCE_THRESHOLD = 0.85
    
    # Minimum valid confidence (for internal validation)
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0


class VariantSelector:
    """
    Selects the best variant result based on confidence threshold
    and variant ranking.
    """
    
    # Order of variants (for fallback, if needed in future)
    VARIANT_ORDER = [
        "baseline",
        "clahe_sharp",
        "denoise_clahe",
        "otsu",
        "adaptive"
    ]
    
    @staticmethod
    def select_best_variant(
        comparison_result: Dict,
        threshold: float = ThresholdConfig.OCR_CONFIDENCE_THRESHOLD
    ) -> Dict:
        """
        Select the best variant result based on confidence threshold.
        
        Args:
            comparison_result: Result from compare_all_preprocessing_variants()
                {
                    "variants": [
                        {
                            "variant": "baseline",
                            "text": "...",
                            "confidence": 0.91,
                            "success": True
                        },
                        ...
                    ],
                    "best_by_confidence": "baseline",
                    "best_confidence": 0.91
                }
            threshold: Confidence threshold (default: 0.85)
        
        Returns:
            Structured result with selected variant and all variants:
            {
                "selected_result": {
                    "text": "MH12AB1234",
                    "confidence": 0.94,
                    "variant_used": "clahe_sharp",
                    "status": "ACCEPTED"
                },
                "all_variants": [
                    {
                        "variant": "baseline",
                        "text": "...",
                        "confidence": 0.79,
                        "success": True
                    },
                    ...
                ],
                "threshold_used": 0.85,
                "best_available_confidence": 0.94,
                "variants_above_threshold": 2
            }
        """
        
        # Validate input
        if not comparison_result:
            return VariantSelector._failed_result(
                "Empty comparison result",
                threshold,
                []
            )
        
        variants = comparison_result.get("variants", [])
        
        if not variants:
            return VariantSelector._failed_result(
                "No variants in comparison result",
                threshold,
                []
            )
        
        # Prefer variants that pass validate_plate
        valid_variants = []
        for v in variants:
            if v.get("success", False):
                t = v.get("text", "")
                cleaned, is_v = validate_plate(t)
                v["is_valid"] = is_v
                v["cleaned_text"] = cleaned if is_v else t
                if is_v:
                    valid_variants.append(v)

        candidates = valid_variants if valid_variants else [v for v in variants if v.get("success", False)]

        if candidates:
            # Sort by confidence descending
            candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
            best_variant = candidates[0]
            best_confidence = best_variant.get("confidence", 0.0)
        else:
            best_variant = None
            best_confidence = -1.0

        # If no successful variant found
        if best_variant is None:
            return VariantSelector._failed_result(
                "All variants failed OCR",
                threshold,
                variants
            )

        # Status determination
        acc_floor = getattr(config, 'ACCEPTANCE_CONFIDENCE_FLOOR', 0.5) if config else 0.5
        is_valid = best_variant.get("is_valid", False)
        raw_text = best_variant.get("text", "")

        if not raw_text.strip():
            status = ResultStatus.OCR_FAILED.value
        elif is_valid and best_confidence >= acc_floor:
            status = ResultStatus.ACCEPTED.value
        else:
            status = ResultStatus.NEEDS_REVIEW.value

        # Count variants above threshold
        above_threshold = sum(
            1 for v in variants
            if v.get("success", False) and v.get("confidence", 0.0) >= threshold
        )

        return {
            "selected_result": {
                "text": best_variant.get("text", ""),
                "confidence": round(best_confidence, 4),
                "variant_used": best_variant.get("variant", None),
                "status": status
            },
            "all_variants": [
                {
                    "variant": v.get("variant"),
                    "text": v.get("text", ""),
                    "confidence": round(v.get("confidence", 0.0), 4),
                    "success": v.get("success", False)
                }
                for v in variants
            ],
            "threshold_used": threshold,
            "best_available_confidence": round(best_confidence, 4),
            "variants_above_threshold": above_threshold
        }
    
    @staticmethod
    def _failed_result(
        reason: str,
        threshold: float,
        variants: List[Dict]
    ) -> Dict:
        """
        Return a failed result structure.
        
        Args:
            reason: Reason for failure
            threshold: Confidence threshold used
            variants: List of variant results
        
        Returns:
            Failed result dictionary
        """
        logger.warning(f"[Variant Selection] {reason}")
        
        return {
            "selected_result": {
                "text": "",
                "confidence": 0.0,
                "variant_used": None,
                "status": ResultStatus.OCR_FAILED.value
            },
            "all_variants": [
                {
                    "variant": v.get("variant"),
                    "text": v.get("text", ""),
                    "confidence": round(v.get("confidence", 0.0), 4),
                    "success": v.get("success", False)
                }
                for v in variants
            ],
            "threshold_used": threshold,
            "best_available_confidence": 0.0,
            "variants_above_threshold": 0,
            "error": reason
        }


class ResultFormatter:
    """Format results for API responses and logging"""
    
    @staticmethod
    def format_for_api(
        selected_result: Dict,
        all_variants: List[Dict],
        plate_crop: Optional[bytes] = None
    ) -> Dict:
        """
        Format result for API response (Flask/JSON).
        
        Args:
            selected_result: Selected variant result
            all_variants: All variant results
            plate_crop: Optional cropped plate image (not included in this dict)
        
        Returns:
            API-ready dictionary
        """
        return {
            "success": selected_result.get("status") == ResultStatus.ACCEPTED.value,
            "text": selected_result.get("text", ""),
            "confidence": selected_result.get("confidence", 0.0),
            "variant_used": selected_result.get("variant_used"),
            "status": selected_result.get("status"),
            "all_variants": all_variants
        }
    
    @staticmethod
    def format_for_logging(
        selected_result: Dict,
        best_available: float,
        threshold: float
    ) -> str:
        """
        Format result for console/file logging.
        
        Args:
            selected_result: Selected variant result
            best_available: Best confidence among all variants
            threshold: Threshold used
        
        Returns:
            Formatted log message
        """
        status = selected_result.get("status")
        text = selected_result.get("text", "")
        conf = selected_result.get("confidence", 0.0)
        variant = selected_result.get("variant_used")
        
        return (
            f"[Phase 2.2] Status={status} | Text='{text}' | "
            f"Conf={conf:.4f} ({conf*100:.1f}%) | Variant={variant} | "
            f"Threshold={threshold:.2f}"
        )


class ThresholdValidator:
    """Validates and constrains threshold values"""
    
    @staticmethod
    def validate_threshold(value: float) -> float:
        """
        Validate and constrain threshold to valid range.
        
        Args:
            value: Threshold value to validate
        
        Returns:
            Valid threshold value (clamped to [0.0, 1.0])
        """
        if not isinstance(value, (int, float)):
            logger.warning(f"Invalid threshold type: {type(value)}, using default")
            return ThresholdConfig.OCR_CONFIDENCE_THRESHOLD
        
        clamped = max(ThresholdConfig.MIN_CONFIDENCE, min(value, ThresholdConfig.MAX_CONFIDENCE))
        
        if clamped != value:
            logger.warning(f"Threshold {value} clamped to {clamped}")
        
        return clamped
    
    @staticmethod
    def validate_confidence(value: float) -> float:
        """
        Validate and constrain confidence to valid range.
        
        Args:
            value: Confidence value to validate
        
        Returns:
            Valid confidence value (clamped to [0.0, 1.0])
        """
        return max(ThresholdConfig.MIN_CONFIDENCE, min(value, ThresholdConfig.MAX_CONFIDENCE))


# Precomputed statistics for thresholding decisions (from Phase 2.1 testing)
VARIANT_STATISTICS = {
    "baseline": {
        "avg_confidence": 0.9099,
        "best_occurrences": 3,
        "success_rate": 1.0
    },
    "otsu": {
        "avg_confidence": 0.8878,
        "best_occurrences": 2,
        "success_rate": 1.0
    },
    "adaptive": {
        "avg_confidence": 0.7032,
        "best_occurrences": 1,
        "success_rate": 0.8  # 2 failures out of 10
    },
    "clahe_sharp": {
        "avg_confidence": 0.9414,
        "best_occurrences": 3,
        "success_rate": 1.0
    },
    "denoise_clahe": {
        "avg_confidence": 0.9001,
        "best_occurrences": 1,
        "success_rate": 1.0
    }
}


def get_variant_statistics() -> Dict:
    """Return variant statistics from Phase 2.1 testing"""
    return VARIANT_STATISTICS.copy()
