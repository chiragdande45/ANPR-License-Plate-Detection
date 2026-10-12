"""
Phase 2.3: Indian Number Plate Format Validation

This module validates cleaned OCR text against Indian vehicle registration plate structures.

Important:
- Does NOT reject every invalid format
- Returns detailed validation info (valid, pattern, reason)
- Supports multiple common formats
- Flexible to allow variations in real-world plates

Indian Plate Structures:
1. Standard format: AA##XX#### (2 letters, 2 digits, 2 letters, 4 digits)
   Example: MH12AB1234, DL01CD5678
   
2. With spaces: AA ## XX #### or AA##XX ####
   Example: MH 12 AB 1234, MH12 AB 1234
   
3. Old format: AA##A#### (2 letters, 2 digits, 1 letter, 4 digits)
   Example: MH12A1234
"""

import re
from typing import Dict, List, Optional, Tuple


class IndianPlatePatterns:
    """Common Indian vehicle registration plate patterns"""
    
    # Cleaned/normalized patterns (no spaces)
    PATTERNS = {
        # Standard modern format: AA##XX####
        # Examples: MH12AB1234, DL01CD5678, KA01AE1234
        "standard": {
            "pattern": r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$",
            "description": "Standard format: 2 letters, 2 digits, 2 letters, 4 digits",
            "example": "MH12AB1234",
            "is_valid_format": True,
            "length": 10
        },
        
        # Old format: AA##A####
        # Examples: MH12A1234, DL01C5678
        "old_format": {
            "pattern": r"^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$",
            "description": "Old format: 2 letters, 2 digits, 1 letter, 4 digits",
            "example": "MH12A1234",
            "is_valid_format": True,
            "length": 9
        },
        
        # District-letter format: AA#XXX#### (1 digit + 1 letter district code)
        # The RTO sub-code is written as a single digit + single letter, e.g. DL3C, MH4A, UP7C.
        # Full example: DL3CAY9324  (DL = Delhi, 3C = sub-RTO, AY = series, 9324 = number)
        # Structure: 2 state letters + 1 digit + 1 letter + 2 series letters + 4 digits = 10 chars
        "district_letter": {
            "pattern": r"^[A-Z]{2}\d[A-Z]{3}\d{4}$",
            "description": "District-letter format: 2 letters, 1 digit, 1 letter RTO, 2 series letters, 4 digits",
            "example": "DL3CAY9324",
            "is_valid_format": True,
            "length": 10
        },

        # Three-wheeler format: AA##XXX#### (sometimes 3 letters)
        # Examples: MH12ABC1234
        "three_wheeler": {
            "pattern": r"^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$",
            "description": "Three-wheeler format: 2 letters, 2 digits, 3 letters, 4 digits",
            "example": "MH12ABC1234",
            "is_valid_format": True,
            "length": 11
        },
        
        # Commercial/heavy vehicle: AA##C#### (single letter after district code)
        # Examples: MH12C1234, DL01C5678
        "commercial": {
            "pattern": r"^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$",
            "description": "Commercial/Heavy: 2 letters, 2 digits, 1 letter, 4 digits",
            "example": "MH12C1234",
            "is_valid_format": True,
            "length": 9
        },
        
        # Taxi/cab format: AA##TA#### (TA = taxi)
        # Examples: MH12TA1234, DL01TA5678
        "taxi": {
            "pattern": r"^[A-Z]{2}\d{2}(TA|TC|TN|TR)\d{4}$",
            "description": "Taxi/Cab format: 2 letters, 2 digits, TA/TC/TN/TR, 4 digits",
            "example": "MH12TA1234",
            "is_valid_format": True,
            "length": 10
        },
        
        # Generic mixed format: at least 2 letters and 4 digits
        # Allows for variations and potential OCR errors
        "generic_mixed": {
            "pattern": r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{8,12}$",
            "description": "Generic: mixed letters/digits, length 8-12",
            "example": "MH12AB1234",
            "is_valid_format": False,
            "length_range": (8, 12),
            "reason": "Matches generic format but not standard structure"
        }
    }
    
    @classmethod
    def get_pattern(cls, name: str) -> Optional[Dict]:
        """Get a specific pattern by name"""
        return cls.PATTERNS.get(name)
    
    @classmethod
    def get_all_patterns(cls) -> Dict:
        """Get all patterns"""
        return cls.PATTERNS.copy()


class IndianStateCodeValidator:
    """Validate Indian vehicle registration state codes"""
    
    # Official Indian state/territory registration codes
    STATE_CODES = {
        "AN": "Andaman and Nicobar Islands",
        "AP": "Andhra Pradesh",
        "AR": "Arunachal Pradesh",
        "AS": "Assam",
        "BR": "Bihar",
        "CH": "Chandigarh",
        "CT": "Chhattisgarh",
        "DD": "Daman and Diu",
        "DL": "Delhi",
        "DN": "Dadra and Nagar Haveli",
        "GA": "Goa",
        "GJ": "Gujarat",
        "HR": "Haryana",
        "HP": "Himachal Pradesh",
        "JK": "Jammu & Kashmir",
        "JH": "Jharkhand",
        "KA": "Karnataka",
        "KL": "Kerala",
        "LA": "Ladakh",
        "LD": "Lakshadweep",
        "MP": "Madhya Pradesh",
        "MH": "Maharashtra",
        "MN": "Manipur",
        "ML": "Meghalaya",
        "MZ": "Mizoram",
        "NL": "Nagaland",
        "OD": "Odisha",
        "OR": "Odisha (old code)",
        "PB": "Punjab",
        "PY": "Puducherry",
        "RJ": "Rajasthan",
        "SK": "Sikkim",
        "TG": "Telangana",
        "TN": "Tamil Nadu",
        "TR": "Tripura",
        "UP": "Uttar Pradesh",
        "UK": "Uttarakhand",
        "WB": "West Bengal",
    }
    
    @staticmethod
    def is_valid_state_code(code: str) -> bool:
        """Check if state code is valid"""
        if not code or len(code) != 2:
            return False
        return code.upper() in IndianStateCodeValidator.STATE_CODES
    
    @staticmethod
    def get_state_name(code: str) -> Optional[str]:
        """Get state name from code"""
        return IndianStateCodeValidator.STATE_CODES.get(code.upper())


class PlateValidator:
    """Validate cleaned OCR text as Indian vehicle registration plate"""
    
    @staticmethod
    def validate_indian_plate(text: str) -> Dict:
        """
        Validate if text matches Indian plate format.
        
        Returns detailed validation info:
        {
            "valid": bool,
            "text": str,
            "pattern_name": str or None,
            "pattern": str or None,
            "validation_status": "VALID", "INVALID_FORMAT", "INVALID_STATE_CODE", or "EMPTY",
            "reason": str,
            "state_code": str or None,
            "state_name": str or None,
            "length": int,
            "matched_patterns": [list of matched pattern names]
        }
        
        Args:
            text: Cleaned/normalized text (e.g., "MH12AB1234")
            
        Returns:
            Validation result dictionary
        """
        
        # Check for empty
        if not text or not text.strip():
            return {
                "valid": False,
                "text": "",
                "pattern_name": None,
                "pattern": None,
                "validation_status": "EMPTY",
                "reason": "Empty or whitespace-only text",
                "state_code": None,
                "state_name": None,
                "length": 0,
                "matched_patterns": []
            }
        
        text = text.strip().upper()
        length = len(text)
        matched_patterns = []
        best_match = None
        best_pattern_name = None
        
        # EARLY VALIDATION: Check state code first
        # Indian plates ALWAYS start with 2 letters (state code)
        # If first 2 characters are not both letters, reject immediately
        # If first 2 characters are letters but NOT a valid state code, reject immediately
        state_code = None
        state_name = None
        
        if length >= 2:
            potential_state = text[:2]
            # Indian plates MUST start with 2 letters
            if not potential_state.isalpha():
                # First 2 chars are not both letters - invalid plate structure
                return {
                    "valid": False,
                    "text": text,
                    "pattern_name": None,
                    "pattern": None,
                    "validation_status": "INVALID_FORMAT",
                    "reason": f"Indian plates must start with 2-letter state code, got '{potential_state}'",
                    "state_code": None,
                    "state_name": None,
                    "length": length,
                    "matched_patterns": []
                }
            
            # Both chars are letters - validate as state code
            if IndianStateCodeValidator.is_valid_state_code(potential_state):
                state_code = potential_state
                state_name = IndianStateCodeValidator.get_state_name(potential_state)
            else:
                # First 2 chars are letters but NOT a valid state code
                # This is a red flag - reject as invalid state code
                return {
                    "valid": False,
                    "text": text,
                    "pattern_name": None,
                    "pattern": None,
                    "validation_status": "INVALID_STATE_CODE",
                    "reason": (
                        f"Invalid state code: '{potential_state}'. "
                        f"Valid codes: {', '.join(sorted(IndianStateCodeValidator.STATE_CODES.keys()))}"
                    ),
                    "state_code": None,
                    "state_name": None,
                    "length": length,
                    "matched_patterns": []
                }
        
        # Try to match against all valid patterns
        patterns = IndianPlatePatterns.get_all_patterns()
        
        for pattern_name, pattern_info in patterns.items():
            pattern_regex = pattern_info.get("pattern")
            if not pattern_regex:
                continue
            
            if re.match(pattern_regex, text):
                matched_patterns.append(pattern_name)
                
                # Prefer patterns marked as valid format
                if pattern_info.get("is_valid_format", False):
                    if best_match is None:
                        best_match = pattern_info
                        best_pattern_name = pattern_name
        
        # Determine validation status
        if best_match:
            validation_status = "VALID"
            reason = f"Matches {best_pattern_name}: {best_match.get('description')}"
        elif matched_patterns:
            # Matched a pattern but not marked as valid format
            validation_status = "INVALID_FORMAT"
            reason = f"Matched pattern(s): {', '.join(matched_patterns)} but not standard format"
        else:
            validation_status = "INVALID_FORMAT"
            reason = "Does not match any known Indian plate format"
        
        return {
            "valid": validation_status == "VALID",
            "text": text,
            "pattern_name": best_pattern_name,
            "pattern": best_match.get("pattern") if best_match else None,
            "validation_status": validation_status,
            "reason": reason,
            "state_code": state_code,
            "state_name": state_name,
            "length": length,
            "matched_patterns": matched_patterns
        }
    
    @staticmethod
    def get_validation_summary(validation_result: Dict) -> str:
        """
        Get human-readable validation summary.
        
        Args:
            validation_result: Result from validate_indian_plate()
            
        Returns:
            Formatted string
        """
        status = validation_result.get("validation_status", "UNKNOWN")
        text = validation_result.get("text", "")
        reason = validation_result.get("reason", "No reason provided")
        
        if status == "VALID":
            return f"✓ VALID: {text} ({reason})"
        elif status == "INVALID_FORMAT":
            return f"✗ INVALID: {text} ({reason})"
        elif status == "EMPTY":
            return "✗ EMPTY: No text to validate"
        else:
            return f"? UNKNOWN: {reason}"


class PlateQualityScorer:
    """Score plate quality based on validation and other factors"""
    
    @staticmethod
    def score_plate_quality(
        text: str,
        confidence: float,
        validation_result: Dict,
        corrections_made: int = 0,
        correction_confidence: float = 1.0
    ) -> Dict:
        """
        Score overall plate quality.
        
        Factors:
        1. Validation status (30%)
        2. OCR confidence (30%)
        3. Number of corrections (20%)
        4. Correction confidence (20%)
        
        Args:
            text: Cleaned plate text
            confidence: OCR confidence (0-1)
            validation_result: Result from validate_indian_plate()
            corrections_made: Number of character corrections
            correction_confidence: Confidence in corrections (0-1)
            
        Returns:
            Quality score dictionary
        """
        
        # Validation component (0-30)
        validation_score = 0
        if validation_result.get("valid"):
            validation_score = 30
        elif validation_result.get("validation_status") == "INVALID_FORMAT":
            validation_score = 10
        else:
            validation_score = 5
        
        # OCR confidence component (0-30)
        ocr_score = confidence * 30
        
        # Corrections component (0-20)
        # Fewer corrections = higher score
        if corrections_made == 0:
            corrections_score = 20
        elif corrections_made == 1:
            corrections_score = 15
        elif corrections_made == 2:
            corrections_score = 10
        elif corrections_made <= 3:
            corrections_score = 5
        else:
            corrections_score = 0
        
        # Correction confidence component (0-20)
        correction_conf_score = correction_confidence * 20
        
        # Total score (0-100)
        total_score = (
            validation_score +
            ocr_score +
            corrections_score +
            correction_conf_score
        )
        
        # Normalize to 0-1 scale for consistency
        normalized_score = total_score / 100.0
        
        # Determine quality tier
        if normalized_score >= 0.85:
            quality_tier = "EXCELLENT"
        elif normalized_score >= 0.70:
            quality_tier = "GOOD"
        elif normalized_score >= 0.50:
            quality_tier = "FAIR"
        elif normalized_score >= 0.30:
            quality_tier = "POOR"
        else:
            quality_tier = "VERY_POOR"
        
        return {
            "total_score": round(normalized_score, 4),
            "quality_tier": quality_tier,
            "validation_score": round(validation_score / 30, 4),
            "ocr_confidence_score": round(ocr_score / 30, 4),
            "corrections_score": round(corrections_score / 20, 4),
            "correction_confidence_score": round(correction_conf_score / 20, 4),
            "components": {
                "validation": validation_result.get("valid"),
                "ocr_confidence": round(confidence, 4),
                "corrections_made": corrections_made,
                "correction_confidence": round(correction_confidence, 4)
            }
        }


# Module-level convenience functions

def validate_indian_plate(text: str) -> Dict:
    """Convenience function: validate plate format"""
    return PlateValidator.validate_indian_plate(text)


def get_validation_summary(validation_result: Dict) -> str:
    """Convenience function: get validation summary"""
    return PlateValidator.get_validation_summary(validation_result)


def score_plate_quality(
    text: str,
    confidence: float,
    validation_result: Dict,
    corrections_made: int = 0,
    correction_confidence: float = 1.0
) -> Dict:
    """Convenience function: score plate quality"""
    return PlateQualityScorer.score_plate_quality(
        text, confidence, validation_result, corrections_made, correction_confidence
    )
