"""
Phase 2.3: OCR Text Cleaning & Character Correction

This module implements:
1. Text normalization (remove spaces, symbols, uppercase)
2. Position-aware OCR character correction (0/O, 1/I, 5/S, 8/B, 2/Z)
3. Safe cleaning without aggressive guessing

Important:
- Preserves raw OCR results
- Only corrects when justified by position/context
- Does NOT force every result into a valid plate format
- Modular and easy to modify
"""

import re
from typing import Dict, Tuple, Optional


class TextNormalizer:
    """Normalize OCR text for license plate processing"""
    
    @staticmethod
    def normalize_ocr_text(text: str) -> str:
        """
        Normalize OCR text for license plate processing.
        
        Steps:
        1. Convert to uppercase
        2. Remove leading/trailing whitespace
        3. Remove unnecessary internal spaces
        4. Remove unwanted symbols
        5. Preserve alphanumeric characters only
        
        Args:
            text: Raw OCR text (may contain spaces, symbols, mixed case)
            
        Returns:
            Normalized text (uppercase, no spaces/symbols)
            
        Examples:
            "hr 26 da 5443" → "HR26DA5443"
            "MH 12 AB 1234" → "MH12AB1234"
            "HR-26-DA-5443" → "HR26DA5443"
            "MH@12#AB$1234" → "MH12AB1234"
        """
        if not text:
            return ""
        
        # Step 1: Convert to uppercase
        text = text.upper()
        
        # Step 2: Remove leading/trailing whitespace
        text = text.strip()
        
        # Step 3-4: Remove spaces and unwanted symbols, keep alphanumeric only
        # This preserves: A-Z, 0-9
        # This removes: spaces, hyphens, underscores, @, #, $, %, etc.
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text
    
    @staticmethod
    def remove_common_symbols(text: str) -> str:
        """
        Remove common OCR artifacts and symbols.
        
        Removes:
        - Spaces
        - Hyphens and underscores
        - Special symbols (@, #, $, %, etc.)
        - Common OCR noise characters
        
        Args:
            text: Text to clean
            
        Returns:
            Text without symbols
        """
        if not text:
            return ""
        
        # Keep only alphanumeric characters
        return re.sub(r'[^A-Za-z0-9]', '', text)
    
    @staticmethod
    def split_into_segments(text: str) -> list:
        """
        Split normalized text into logical segments for analysis.
        
        Indian plates typically have structure like:
        - State code: 2 letters
        - District code: 2 digits
        - Plate type: 1-2 letters
        - Serial: 4 digits
        
        But we keep it flexible for various formats.
        
        Args:
            text: Normalized plate text
            
        Returns:
            List of character groups
        """
        if not text:
            return []
        
        # Split into consecutive groups of same type (letters or digits)
        segments = []
        current_segment = ""
        current_type = None  # 'letter' or 'digit'
        
        for char in text:
            char_type = 'letter' if char.isalpha() else 'digit'
            
            if current_type is None:
                current_type = char_type
                current_segment = char
            elif current_type == char_type:
                current_segment += char
            else:
                segments.append(current_segment)
                current_type = char_type
                current_segment = char
        
        if current_segment:
            segments.append(current_segment)
        
        return segments


class OCRCharacterCorrector:
    """
    Position-aware OCR character correction.
    
    Corrects common OCR confusions based on position and context.
    """
    
    # Common OCR confusions (bidirectional)
    CONFUSION_PAIRS = {
        '0': 'O',  # Zero ↔ Letter O
        'O': '0',
        '1': 'I',  # One ↔ Letter I
        'I': '1',
        '5': 'S',  # Five ↔ Letter S
        'S': '5',
        '8': 'B',  # Eight ↔ Letter B
        'B': '8',
        '2': 'Z',  # Two ↔ Letter Z
        'Z': '2',
    }
    
    @staticmethod
    def correct_ocr_characters(text: str) -> str:
        """
        Apply position-aware OCR character correction.
        
        Strategy:
        - Analyze text structure (segments of letters/digits)
        - For each segment, determine expected type
        - Correct characters that don't match expected type
        
        Example:
            "MH12O81234" with context suggests:
            - MH: letters ✓
            - 12: digits ✓
            - O8: mixed → position-specific correction
            - 1234: digits ✓
        
        Args:
            text: Normalized text (from normalize_ocr_text)
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Split into segments (consecutive letters/digits)
        segments = TextNormalizer.split_into_segments(text)
        
        corrected_segments = []
        for segment in segments:
            if segment:
                corrected = OCRCharacterCorrector._correct_segment(segment)
                corrected_segments.append(corrected)
        
        return ''.join(corrected_segments)
    
    @staticmethod
    def _correct_segment(segment: str) -> str:
        """
        Correct a single segment based on its expected type.
        
        If segment is predominantly letters, correct digit-like chars to letters.
        If segment is predominantly digits, correct letter-like chars to digits.
        
        Args:
            segment: Single segment (e.g., "O81", "MH12")
            
        Returns:
            Corrected segment
        """
        if not segment or len(segment) == 0:
            return segment
        
        # Determine expected type from segment
        # Count letters vs digits
        letter_count = sum(1 for c in segment if c.isalpha() and c not in '01589')
        digit_count = sum(1 for c in segment if c.isdigit() and c not in 'OBISZ')
        
        # Add confusion characters to appropriate counts
        for c in segment:
            if c in '0O':
                # Could be digit or letter, count both
                pass
            elif c in '1I':
                pass
            elif c in '5S':
                pass
            elif c in '8B':
                pass
            elif c in '2Z':
                pass
        
        # Determine segment type based on majority
        if letter_count > digit_count:
            expected_type = 'letter'
        elif digit_count > letter_count:
            expected_type = 'digit'
        else:
            # Tie or mixed - don't correct
            return segment
        
        # Apply corrections based on expected type
        corrected = ""
        for char in segment:
            if expected_type == 'letter' and char.isdigit():
                # Digit in letter segment - try to correct
                corrected += OCRCharacterCorrector._digit_to_letter(char)
            elif expected_type == 'digit' and char.isalpha():
                # Letter in digit segment - try to correct
                corrected += OCRCharacterCorrector._letter_to_digit(char)
            else:
                # No correction needed
                corrected += char
        
        return corrected
    
    @staticmethod
    def _digit_to_letter(char: str) -> str:
        """Convert digit-like character to letter."""
        conversions = {
            '0': 'O',  # Zero → O
            '1': 'I',  # One → I
            '5': 'S',  # Five → S
            '8': 'B',  # Eight → B
            '2': 'Z',  # Two → Z
        }
        return conversions.get(char, char)
    
    @staticmethod
    def _letter_to_digit(char: str) -> str:
        """Convert letter-like character to digit."""
        conversions = {
            'O': '0',  # O → Zero
            'I': '1',  # I → One
            'S': '5',  # S → Five
            'B': '8',  # B → Eight
            'Z': '2',  # Z → Two
        }
        return conversions.get(char, char)
    
    @staticmethod
    def get_correction_confidence(original: str, corrected: str) -> float:
        """
        Estimate confidence in corrections made.
        
        Higher score = more confident
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Confidence score 0.0-1.0
        """
        if original == corrected:
            # No corrections made - high confidence (no false corrections)
            return 1.0
        
        # Count changes
        changes = sum(1 for a, b in zip(original, corrected) if a != b)
        
        # Fewer changes = higher confidence
        # Max 2-3 corrections is still reasonable
        if changes > 3:
            return 0.3  # Many changes - low confidence
        elif changes > 2:
            return 0.5  # Several changes - moderate confidence
        elif changes > 1:
            return 0.7  # Few changes - good confidence
        else:
            return 0.9  # Single change - high confidence
        
        return 0.0


class TextCleaningPipeline:
    """
    Complete text cleaning pipeline.
    
    Pipeline:
    1. Normalize (uppercase, remove spaces/symbols)
    2. Character correction (position-aware)
    3. Confidence scoring
    """
    
    @staticmethod
    def clean_ocr_text(raw_text: str) -> Dict:
        """
        Clean and correct OCR text.
        
        Args:
            raw_text: Raw OCR text from PaddleOCR
            
        Returns:
            Dictionary with:
            {
                "raw_text": original,
                "cleaned_text": normalized,
                "corrected_text": corrected,
                "corrections_made": count,
                "correction_confidence": score
            }
        """
        if not raw_text or not raw_text.strip():
            return {
                "raw_text": raw_text or "",
                "cleaned_text": "",
                "corrected_text": "",
                "corrections_made": 0,
                "correction_confidence": 1.0
            }
        
        # Step 1: Normalize
        cleaned = TextNormalizer.normalize_ocr_text(raw_text)
        
        # Step 2: Correct characters
        corrected = OCRCharacterCorrector.correct_ocr_characters(cleaned)
        
        # Step 3: Count corrections and estimate confidence
        corrections_made = sum(1 for a, b in zip(cleaned, corrected) if a != b)
        correction_confidence = OCRCharacterCorrector.get_correction_confidence(
            cleaned, corrected
        )
        
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "corrected_text": corrected,
            "corrections_made": corrections_made,
            "correction_confidence": round(correction_confidence, 4)
        }


# Module-level convenience functions

def normalize_ocr_text(text: str) -> str:
    """Convenience function: normalize text"""
    return TextNormalizer.normalize_ocr_text(text)


def correct_ocr_characters(text: str) -> str:
    """Convenience function: correct characters"""
    return OCRCharacterCorrector.correct_ocr_characters(text)


def clean_ocr_text(raw_text: str) -> Dict:
    """Convenience function: full cleaning pipeline"""
    return TextCleaningPipeline.clean_ocr_text(raw_text)
