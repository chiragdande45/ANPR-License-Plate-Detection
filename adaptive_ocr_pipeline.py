"""
Phase 2.4: Adaptive OCR Pipeline

Main pipeline that orchestrates:
1. Image analysis
2. Adaptive variant ranking
3. Fallback OCR with validation
4. Candidate ranking and selection

Replaces the simple "run all 5 variants, pick best confidence" approach
with intelligent routing that respects validation and quality scores.
"""

import sys
sys.path.insert(0, r'c:\Projects\ANPR-Project_Code_File\ANPR')

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from plate_image_analyzer import analyze_plate_image
from adaptive_router import VariantRanker, CandidateRanker, AdaptiveOCRPipeline


class AdaptiveOCRProcessor:
    """Main adaptive OCR processor."""
    
    def __init__(self, preprocessing_module=None, ocr_module=None, validator_module=None):
        """
        Initialize adaptive OCR processor.
        
        Args:
            preprocessing_module: Module with preprocessing methods
            ocr_module: Module with OCR methods
            validator_module: Module with validation methods
        """
        self.preprocessing = preprocessing_module
        self.ocr_module = ocr_module
        self.validator = validator_module
    
    def process_plate_adaptive(
        self,
        plate_image: np.ndarray,
        max_variants: Optional[int] = None,
        confidence_threshold: float = 0.85,
        require_validation: bool = True,
        debug: bool = False
    ) -> Dict:
        """
        Process plate using adaptive variant routing.
        
        Pipeline:
        1. Analyze plate image (brightness, contrast, blur, etc.)
        2. Rank preprocessing variants based on image characteristics
        3. Try variants in order until:
           - Confidence threshold met, AND
           - Validation passed (if required)
        4. Rank candidates by confidence + validation + quality
        5. Return best candidate
        
        Args:
            plate_image: Cropped plate image
            max_variants: Max variants to try (None = all 5)
            confidence_threshold: Min confidence to accept (default 0.85)
            require_validation: Require validation pass? (default True)
            debug: Print debug info?
            
        Returns:
            Result dict with:
            - success: bool (confidence threshold met + validation passed)
            - text: Best OCR text
            - confidence: Best confidence
            - variant_used: Which variant produced best result
            - variants_tried: Number of variants executed
            - image_profile: Image characteristics
            - strategy: Strategy used
            - all_candidates: All candidates tried
            - processing_time_ms: Execution time
        """
        
        start_time = time.time()
        
        # Step 1: Analyze plate image
        image_profile = analyze_plate_image(plate_image)
        if debug:
            print(f"\n[ADAPTIVE] Image profile: brightness={image_profile['brightness']:.1f}, "
                  f"contrast={image_profile['contrast']:.1f}, blur={image_profile['blur_score']:.1f}")
        
        # Step 2: Create OCR strategy
        strategy = AdaptiveOCRPipeline.create_ocr_strategy(image_profile)
        if debug:
            print(f"[ADAPTIVE] Strategy: {strategy['conditions']} conditions")
            print(f"[ADAPTIVE] Variant order: {' → '.join(strategy['variant_ranking'][:3])}")
        
        # Step 3: Get variants to try
        variant_ranking = strategy['variant_ranking']
        if max_variants:
            variant_ranking = variant_ranking[:max_variants]
        
        # Step 4: Try variants in order
        candidates = []
        variants_tried = 0
        best_candidate = None
        best_score = -1
        
        for variant_name in variant_ranking:
            if debug:
                print(f"[ADAPTIVE] Trying {variant_name}...")
            
            # Preprocess
            preprocessed = self._preprocess(plate_image, variant_name)
            if preprocessed is None:
                if debug:
                    print(f"[ADAPTIVE]   Preprocessing failed")
                continue
            
            # OCR
            ocr_result = self._ocr(preprocessed)
            if ocr_result is None:
                if debug:
                    print(f"[ADAPTIVE]   OCR failed")
                continue
            
            raw_text = ocr_result.get('text', '')
            confidence = ocr_result.get('confidence', 0)
            
            # Clean + Validate
            cleaned = self._clean_text(raw_text)
            validation_result = self._validate(cleaned)
            quality_score = self._score_quality(cleaned, confidence, validation_result)
            
            variants_tried += 1
            
            # Create candidate
            candidate = {
                'variant': variant_name,
                'raw_text': raw_text,
                'cleaned_text': cleaned,
                'confidence': confidence,
                'validation_valid': validation_result.get('valid', False),
                'validation_status': validation_result.get('validation_status', 'UNKNOWN'),
                'quality_score': quality_score,
                'success': confidence >= confidence_threshold and (
                    validation_result.get('valid', False) if require_validation else True
                )
            }
            candidates.append(candidate)
            
            if debug:
                print(f"[ADAPTIVE]   {variant_name}: conf={confidence:.4f}, valid={candidate['validation_valid']}, quality={quality_score:.4f}")
            
            # Check if acceptable
            if candidate['success']:
                if debug:
                    print(f"[ADAPTIVE]   ACCEPTED - stopping search")
                best_candidate = candidate
                break
            
            # Track best (even if not yet successful)
            # Composite score: confidence + validation + quality
            composite = confidence * 0.4 + (1.0 if candidate['validation_valid'] else 0) * 0.35 + quality_score * 0.25
            if composite > best_score:
                best_score = composite
                best_candidate = candidate
        
        # Step 5: If no variant succeeded, rank all and pick best
        if best_candidate is None:
            if debug:
                print(f"[ADAPTIVE] No variant met criteria, picking best composite...")
            
            ranked = CandidateRanker.rank_ocr_candidates(candidates)
            if ranked:
                best_candidate, _ = ranked[0]
        
        # Prepare result
        if best_candidate is None:
            # Fallback (shouldn't happen)
            best_candidate = {
                'variant': 'none',
                'raw_text': '',
                'cleaned_text': '',
                'confidence': 0,
                'validation_valid': False,
                'validation_status': 'OCR_FAILED',
                'quality_score': 0,
                'success': False
            }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        result = {
            'success': best_candidate['success'],
            'text': best_candidate.get('cleaned_text', ''),
            'raw_text': best_candidate.get('raw_text', ''),
            'confidence': best_candidate.get('confidence', 0),
            'variant_used': best_candidate.get('variant', ''),
            'validation_status': best_candidate.get('validation_status', ''),
            'quality_score': best_candidate.get('quality_score', 0),
            'variants_tried': variants_tried,
            'image_profile': image_profile,
            'strategy': strategy,
            'all_candidates': candidates,
            'processing_time_ms': round(processing_time_ms, 2)
        }
        
        return result
    
    def _preprocess(self, plate_image: np.ndarray, variant_name: str) -> Optional[np.ndarray]:
        """Apply preprocessing variant."""
        if self.preprocessing is None:
            return plate_image
        
        try:
            method = getattr(self.preprocessing, f'apply_{variant_name}', None)
            if method:
                return method(plate_image)
            return plate_image
        except Exception as e:
            print(f"  Preprocessing error: {e}")
            return None
    
    def _ocr(self, preprocessed: np.ndarray) -> Optional[Dict]:
        """Run OCR on preprocessed image."""
        if self.ocr_module is None:
            return {'text': '', 'confidence': 0}
        
        try:
            result = self.ocr_module.recognize_text(preprocessed)
            return result
        except Exception as e:
            print(f"  OCR error: {e}")
            return None
    
    def _clean_text(self, raw_text: str) -> str:
        """Clean OCR text."""
        if raw_text is None:
            return ""
        # Basic cleaning - uppercase, remove spaces/symbols
        return raw_text.upper().replace(' ', '').replace('-', '')
    
    def _validate(self, text: str) -> Dict:
        """Validate plate format."""
        if self.validator is None:
            return {'valid': False, 'validation_status': 'NO_VALIDATOR'}
        
        try:
            return self.validator.validate_indian_plate(text)
        except Exception as e:
            print(f"  Validation error: {e}")
            return {'valid': False, 'validation_status': 'VALIDATION_ERROR'}
    
    def _score_quality(self, text: str, confidence: float, validation_result: Dict) -> float:
        """Score plate quality."""
        # Simple quality score: 0.5 base + 0.3 from confidence + 0.2 from validation
        validation_score = 1.0 if validation_result.get('valid', False) else 0.0
        quality = 0.5 + (confidence * 0.3) + (validation_score * 0.2)
        return min(1.0, quality)


# Module-level convenience function

def process_plate_adaptive(plate_image: np.ndarray, **kwargs) -> Dict:
    """
    Convenience function: process plate with adaptive routing.
    
    Requires external modules to be set up via AdaptiveOCRProcessor.
    For standalone use, just analyzes image and returns strategy.
    """
    processor = AdaptiveOCRProcessor()
    return processor.process_plate_adaptive(plate_image, **kwargs)
