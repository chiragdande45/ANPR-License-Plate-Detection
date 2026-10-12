"""
Phase 2.4: Adaptive Preprocessing Variant Routing

Intelligently selects and ranks preprocessing variants based on
image characteristics, combined with OCR confidence and validation.

Architecture:
- Image Analysis: Extract image profile (brightness, contrast, blur, etc.)
- Variant Ranking: Rank preprocessing variants based on image profile
- Fallback Pipeline: Try variants in order, validate results
- Candidate Ranking: Rank OCR candidates by confidence + validation + quality
"""

import sys
sys.path.insert(0, r'c:\Projects\ANPR-Project_Code_File\ANPR')

import numpy as np
from typing import Dict, List, Tuple, Optional
from plate_image_analyzer import analyze_plate_image


class VariantRanker:
    """Ranks preprocessing variants based on image characteristics."""
    
    # Variant names in priority order (from correlation analysis)
    VARIANTS_BY_CONFIDENCE = [
        'clahe_sharp',    # 0.9414 avg confidence, 100% success
        'baseline',       # 0.9099 avg confidence, 100% success
        'denoise_clahe',  # 0.9001 avg confidence, 100% success
        'otsu',           # 0.8878 avg confidence, 100% success
        'adaptive'        # 0.7033 avg confidence, 80% success (fallback)
    ]
    
    @staticmethod
    def rank_preprocessing_variants(image_profile: Dict[str, float]) -> List[str]:
        """
        Rank preprocessing variants based on image profile.
        
        Strategy based on analysis:
        - clahe_sharp: Best overall, recommended for most cases
        - otsu: Wins on high-brightness, high-contrast images
        - baseline: Good for well-lit, standard conditions
        - denoise_clahe: Backup for noisy images
        - adaptive: Last resort due to variable performance
        
        Args:
            image_profile: Image profile from analyze_plate_image()
            
        Returns:
            List of variant names in priority order
        """
        
        brightness = image_profile.get('brightness', 128)
        contrast = image_profile.get('contrast', 64)
        blur_score = image_profile.get('blur_score', 50)
        edge_density = image_profile.get('edge_density', 0.1)
        
        # Initialize ranking
        ranking = []
        scores = {}
        
        # --- SCORING LOGIC ---
        # Based on correlation analysis findings
        
        # 1. CLAHE_SHARP: Default strong performer
        #    Best for various lighting conditions
        #    Handles both sharp and moderately blurry
        scores['clahe_sharp'] = 100
        
        # 2. OTSU: Wins on bright + high-contrast images
        #    - Brightness >= 170 → bonus points
        #    - Contrast >= 70 → bonus points
        scores['otsu'] = 85
        if brightness >= 170:
            scores['otsu'] += 10
        if contrast >= 70:
            scores['otsu'] += 10
        
        # 3. BASELINE: Reliable for normal lighting
        #    - 100-200 brightness → bonus
        #    - Lower blur score → bonus
        scores['baseline'] = 80
        if 100 <= brightness <= 200:
            scores['baseline'] += 10
        if blur_score < 40:
            scores['baseline'] += 5
        
        # 4. DENOISE_CLAHE: Good for noisy/edgy images
        #    - High edge density → bonus
        #    - High contrast → bonus
        scores['denoise_clahe'] = 70
        if edge_density > 0.15:
            scores['denoise_clahe'] += 10
        if contrast > 70:
            scores['denoise_clahe'] += 5
        
        # 5. ADAPTIVE: Variable, use last
        #    - Only prioritize if very specific conditions
        #    - Generally keep low priority
        scores['adaptive'] = 40
        if blur_score > 70:  # Very blurry
            scores['adaptive'] += 20
        
        # Sort by score (descending)
        sorted_variants = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranking = [v for v, score in sorted_variants]
        
        return ranking
    
    @staticmethod
    def get_ranking_explanation(image_profile: Dict[str, float]) -> str:
        """
        Get explanation of variant ranking decision.
        
        Args:
            image_profile: Image profile from analyze_plate_image()
            
        Returns:
            Human-readable explanation
        """
        ranking = VariantRanker.rank_preprocessing_variants(image_profile)
        
        brightness = image_profile.get('brightness', 128)
        contrast = image_profile.get('contrast', 64)
        blur_score = image_profile.get('blur_score', 50)
        
        explanation = f"""
Variant Ranking:
  Brightness: {brightness:.1f}, Contrast: {contrast:.1f}, Blur: {blur_score:.1f}
  
Ranking:
  1. {ranking[0]}
  2. {ranking[1]}
  3. {ranking[2]}
  4. {ranking[3]}
  5. {ranking[4]}

Reasoning:
  - If high contrast ({contrast:.1f} > 70): otsu may win
  - If bright ({brightness:.1f} > 170): otsu preferred
  - clahe_sharp: Safe choice for most conditions
  - baseline: Good for normal lighting
  - adaptive: Only as fallback
        """.strip()
        
        return explanation


class CandidateRanker:
    """Ranks OCR candidates by confidence + validation + quality."""
    
    @staticmethod
    def rank_ocr_candidates(
        candidates: List[Dict],
        confidence_weight: float = 0.4,
        validation_weight: float = 0.35,
        quality_weight: float = 0.25
    ) -> List[Tuple[Dict, float]]:
        """
        Rank OCR candidates by composite score.
        
        Composite score combines:
        - OCR confidence (40%): Higher confidence is better
        - Validation (35%): Valid format preferred
        - Quality score (25%): From Phase 2.3 quality scorer
        
        Args:
            candidates: List of candidate results with fields:
                - confidence: OCR confidence (0-1)
                - validation_valid: Is validation valid? (bool)
                - quality_score: Quality score from Phase 2.3 (0-1)
                - text: OCR text (for reference)
                
            confidence_weight: Weight for OCR confidence (default 0.4)
            validation_weight: Weight for validation (default 0.35)
            quality_weight: Weight for quality score (default 0.25)
            
        Returns:
            List of (candidate, composite_score) tuples sorted by score descending
        """
        
        ranked = []
        
        for candidate in candidates:
            # Extract metrics
            conf = candidate.get('confidence', 0)
            valid = 1.0 if candidate.get('validation_valid', False) else 0.0
            quality = candidate.get('quality_score', 0)
            
            # Composite score
            composite = (
                conf * confidence_weight +
                valid * validation_weight +
                quality * quality_weight
            )
            
            ranked.append((candidate, composite))
        
        # Sort by composite score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    @staticmethod
    def get_ranking_explanation(
        ranked_candidates: List[Tuple[Dict, float]]
    ) -> str:
        """
        Get explanation of candidate ranking.
        
        Args:
            ranked_candidates: Output from rank_ocr_candidates()
            
        Returns:
            Human-readable explanation
        """
        explanation = "OCR Candidate Ranking:\n"
        
        for i, (candidate, score) in enumerate(ranked_candidates[:3], 1):
            text = candidate.get('text', 'N/A')
            conf = candidate.get('confidence', 0)
            valid = candidate.get('validation_valid', False)
            quality = candidate.get('quality_score', 0)
            variant = candidate.get('variant', 'N/A')
            
            valid_str = "VALID" if valid else "INVALID"
            explanation += f"\n  {i}. {variant:15s} | Score={score:.4f}"
            explanation += f"\n     Text: '{text}'"
            explanation += f"\n     Confidence: {conf:.4f}, Validation: {valid_str}, Quality: {quality:.4f}"
        
        return explanation


class AdaptiveOCRPipeline:
    """Main adaptive OCR pipeline combining all components."""
    
    @staticmethod
    def create_ocr_strategy(image_profile: Dict[str, float]) -> Dict:
        """
        Create OCR strategy based on image profile.
        
        Args:
            image_profile: Image profile from analyze_plate_image()
            
        Returns:
            Strategy dict with:
            - variant_ranking: List of variants in priority order
            - reasoning: Explanation of strategy
            - confidence_threshold: Confidence threshold for acceptance
        """
        
        # Rank variants
        variant_ranking = VariantRanker.rank_preprocessing_variants(image_profile)
        
        # Determine confidence threshold based on image quality
        brightness = image_profile.get('brightness', 128)
        blur_score = image_profile.get('blur_score', 50)
        
        # Lower threshold for good conditions, higher for poor
        if blur_score < 30 and 80 <= brightness <= 200:
            # Good conditions: require 0.85+ confidence
            confidence_threshold = 0.85
            conditions = "GOOD"
        elif blur_score < 50 or (50 <= brightness <= 220):
            # Medium conditions: 0.80+ confidence
            confidence_threshold = 0.80
            conditions = "MEDIUM"
        else:
            # Poor conditions: 0.75+ confidence (more permissive)
            confidence_threshold = 0.75
            conditions = "POOR"
        
        reasoning = f"""
Image conditions: {conditions}
  Brightness: {brightness:.1f}, Blur: {blur_score:.1f}
  
Strategy:
  1. Try variants in this order: {' → '.join(variant_ranking[:3])}
  2. Accept when confidence >= {confidence_threshold}
  3. Fallback to remaining variants if needed
  4. Final validation: check plate format
        """.strip()
        
        return {
            'variant_ranking': variant_ranking,
            'confidence_threshold': confidence_threshold,
            'reasoning': reasoning,
            'conditions': conditions
        }


# Module-level convenience functions

def rank_preprocessing_variants(image_profile: Dict[str, float]) -> List[str]:
    """Convenience function: rank variants."""
    return VariantRanker.rank_preprocessing_variants(image_profile)


def rank_ocr_candidates(candidates: List[Dict]) -> List[Tuple[Dict, float]]:
    """Convenience function: rank candidates."""
    return CandidateRanker.rank_ocr_candidates(candidates)


def create_ocr_strategy(image_profile: Dict[str, float]) -> Dict:
    """Convenience function: create OCR strategy."""
    return AdaptiveOCRPipeline.create_ocr_strategy(image_profile)
