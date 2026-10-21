"""
PaddleOCR Module for License Plate Recognition

Phase 2.1: Multiple preprocessing variants for comparison
Phase 2.2: Confidence thresholding + variant selection
Phase 2.3: OCR text cleaning + Indian plate validation
"""

import os
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ocr_preprocessing import PlatePreprocessor
from threshold_and_selection import (
    VariantSelector,
    ThresholdConfig,
    ResultFormatter,
    ThresholdValidator,
    ResultStatus
)
from ocr_cleaning import TextCleaningPipeline
from indian_plate_validator import PlateValidator, PlateQualityScorer, validate_plate
try:
    import config
except ImportError:
    config = None
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
            dt_polys = ocr_result.get('dt_polys', [])
            
            # Post-filter: Discard detected text boxes whose height is < 40% of the tallest text box
            # and filter to [A-Z0-9] only
            filtered_texts = []
            filtered_scores = []
            if dt_polys and len(dt_polys) == len(rec_texts):
                heights = []
                for poly in dt_polys:
                    pts = np.array(poly)
                    h = pts[:, 1].max() - pts[:, 1].min()
                    heights.append(h)
                max_h = max(heights) if heights else 0
                for t, s, h in zip(rec_texts, rec_scores, heights):
                    if max_h > 0 and h < 0.40 * max_h:
                        continue  # discard small text like IND
                    clean_t = re.sub(r'[^A-Z0-9]', '', str(t).upper())
                    if clean_t:
                        filtered_texts.append(clean_t)
                        filtered_scores.append(s)
            else:
                for t, s in zip(rec_texts, rec_scores):
                    clean_t = re.sub(r'[^A-Z0-9]', '', str(t).upper())
                    if clean_t:
                        filtered_texts.append(clean_t)
                        filtered_scores.append(s)
            
            recognized_text = ''.join(filtered_texts)
            avg_confidence = float(np.mean(filtered_scores)) if filtered_scores else 0.0
            success = len(recognized_text.strip()) > 0
            
            return {
                "success": success,
                "text": recognized_text.strip(),
                "confidence": round(avg_confidence, 4)
            }
        
        except Exception as e:
            print(f"[OCR ERROR] Recognition failed: {e}")
            return {"success": False, "text": "", "confidence": 0.0, "error": str(e)}
    
    def compare_all_preprocessing_variants(
        self,
        plate_image,
        early_stopping: bool = True,
        early_stop_threshold: float = None,
    ):
        """
        PHASE 2.1 — Compare all preprocessing variants

        Apply all 5 preprocessing methods to the same cropped plate image
        and run OCR on each, collecting results for comparison.

        Args:
            plate_image: Cropped plate region (BGR)
            early_stopping: If True, stop after the first variant that meets
                            early_stop_threshold AND passes plate validation.
                            Default is False (original behaviour — all 5 run).
            early_stop_threshold: Confidence threshold for early stopping.
                            If None, uses ThresholdConfig.OCR_CONFIDENCE_THRESHOLD.
                            Only used when early_stopping=True.

        Returns:
            Dictionary with comparison results:
            {
                "variants": [
                    {
                        "method": "baseline",
                        "description": "...",
                        "text": "MH12AB1234",
                        "confidence": 0.91,
                        "success": True
                    },
                    ...
                ],
                "best_by_confidence": "baseline",
                "best_confidence": 0.91,
                # Early-stopping metadata (always present):
                "ocr_calls_made": int,
                "early_stopped": bool,
                "early_stop_reason": str or None,
            }
        """
        if self.ocr is None:
            return {"error": "OCR not initialized", "variants": [],
                    "ocr_calls_made": 0, "early_stopped": False,
                    "early_stop_reason": None}

        # Resolve early-stop threshold (only used when early_stopping=True)
        if early_stopping and early_stop_threshold is None:
            early_stop_threshold = ThresholdConfig.OCR_CONFIDENCE_THRESHOLD

        try:
            # Get all preprocessing variants
            all_variants = PlatePreprocessor.get_all_variants(plate_image)
            descriptions = PlatePreprocessor.get_variant_descriptions()

            results = []
            best_confidence = 0.0
            best_method = None
            ocr_calls_made = 0
            early_stopped = False
            early_stop_reason = None

            # Run OCR on each variant
            for method_name, processed_image in all_variants.items():
                ocr_result = self.recognize_text(processed_image)
                ocr_calls_made += 1

                conf = ocr_result.get("confidence", 0.0)
                text = ocr_result.get("text", "")
                success = ocr_result.get("success", False)

                result_entry = {
                    "method": method_name,       # kept for backward compat
                    "variant": method_name,      # canonical key expected by VariantSelector
                    "description": descriptions.get(method_name, "Unknown"),
                    "text": text,
                    "confidence": conf,
                    "success": success,
                }
                results.append(result_entry)

                # Track best confidence (used for final selection in baseline mode)
                if conf > best_confidence:
                    best_confidence = conf
                    best_method = method_name

                # ── EARLY STOPPING CHECK ─────────────────────────────────────
                # Only triggered when early_stopping=True.
                # Stop early only on a valid plate!
                if early_stopping and success:
                    clean_cand, is_cand_valid = validate_plate(text)
                    if is_cand_valid and conf >= early_stop_threshold:
                        early_stopped = True
                        early_stop_reason = (
                            f"variant={method_name} conf={conf:.4f}>={early_stop_threshold:.4f}"
                            f" validation=VALID after {ocr_calls_made} call(s)"
                        )
                        break

            return {
                "variants": results,
                "best_by_confidence": best_method,
                "best_confidence": round(best_confidence, 4),
                "ocr_calls_made": ocr_calls_made,
                "early_stopped": early_stopped,
                "early_stop_reason": early_stop_reason,
            }

        except Exception as e:
            print(f"[OCR ERROR] Variant comparison failed: {e}")
            return {
                "error": str(e),
                "variants": [],
                "ocr_calls_made": 0,
                "early_stopped": False,
                "early_stop_reason": None,
            }
    
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
        
        PHASE 1 COMPATIBILITY: Original single-variant pipeline
        
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
    
    def process_plate_with_cleaning_and_validation(
        self,
        image,
        bbox,
        return_crop=False,
        threshold=None,
        return_all_variants=True,
        early_stopping=True,
    ):
        """
        PHASE 2.3: Process plate with variant selection + cleaning + validation

        Pipeline:
        1. Extract plate region
        2. Run all 5 preprocessing variants (or fewer if early_stopping=True)
        3. Collect OCR results from each variant
        4. Select variant with highest confidence
        5. Apply confidence threshold (default: 0.85)
        6. CLEAN OCR text (normalize + character correction)
        7. VALIDATE against Indian plate format
        8. Return structured result with raw, cleaned, final text

        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            return_crop: If True, also return the cropped plate image
            threshold: Confidence threshold (default: 0.85 from config)
            return_all_variants: If True, include all variant results in response
            early_stopping: If True, stop after the first variant that meets
                            the confidence threshold AND passes plate validation.
                            Default is False (original behaviour — all 5 variants run).

        Returns:
            Dictionary with:
            {
                "success": bool,
                "raw_text": str,
                "cleaned_text": str,
                "final_text": str,
                "confidence": float,
                "variant_used": str,
                "status": str (ACCEPTED, LOW_CONFIDENCE, OCR_FAILED, DETECTION_FAILED),
                "validation_status": str,
                "validation_valid": bool,
                "state_code": str or None,
                "quality_score": float,
                "quality_tier": str,
                "corrections_made": int,
                "correction_confidence": float,
                "is_valid_format": bool,
                "all_variants": [list],
                "cropped_image": optional,
                # Early-stopping metadata (always present):
                "ocr_calls_made": int,
                "early_stopped": bool,
                "early_stop_reason": str or None,
            }
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract plate region
        plate_region = image[y1:y2, x1:x2]
        
        if plate_region.size == 0:
            return {
                "success": False,
                "raw_text": "",
                "cleaned_text": "",
                "final_text": "",
                "confidence": 0.0,
                "variant_used": None,
                "status": ResultStatus.DETECTION_FAILED.value,
                "validation_status": "EMPTY",
                "validation_valid": False,
                "state_code": None,
                "quality_score": 0.0,
                "quality_tier": "VERY_POOR",
                "corrections_made": 0,
                "correction_confidence": 0.0,
                "is_valid_format": False,
                "all_variants": [],
                "error": "Could not extract plate region",
                "cropped_image": None
            }
        
        # Store original crop for UI display
        original_crop = plate_region.copy()
        
        # Add synthetic grey padding (from config.PADDING_PERCENT) to prevent edge-character cutoff
        pad_frac = getattr(config, 'PADDING_PERCENT', 0.09) if config else 0.09
        pad_x = max(2, int(plate_region.shape[1] * pad_frac))
        pad_y = max(2, int(plate_region.shape[0] * pad_frac))
        plate_region = cv2.copyMakeBorder(
            plate_region, pad_y, pad_y, pad_x, pad_x, 
            cv2.BORDER_CONSTANT, value=[127, 127, 127]
        )
        
        # Validate threshold
        if threshold is None:
            threshold = ThresholdConfig.OCR_CONFIDENCE_THRESHOLD
        else:
            threshold = ThresholdValidator.validate_threshold(threshold)
        
        # Get all preprocessing variants and run OCR on each
        comparison = self.compare_all_preprocessing_variants(
            plate_region,
            early_stopping=early_stopping,
            early_stop_threshold=threshold,
        )
        
        # Check for errors in comparison
        if "error" in comparison:
            return {
                "success": False,
                "raw_text": "",
                "cleaned_text": "",
                "final_text": "",
                "confidence": 0.0,
                "variant_used": None,
                "status": ResultStatus.OCR_FAILED.value,
                "validation_status": "EMPTY",
                "validation_valid": False,
                "state_code": None,
                "quality_score": 0.0,
                "quality_tier": "VERY_POOR",
                "corrections_made": 0,
                "correction_confidence": 0.0,
                "is_valid_format": False,
                "all_variants": comparison.get("variants", []),
                "error": comparison.get("error"),
                "cropped_image": original_crop if return_crop else None,
                "ocr_calls_made": comparison.get("ocr_calls_made", 0),
                "early_stopped": False,
                "early_stop_reason": None,
            }
        
        # Select best variant based on threshold (prefers valid plates)
        selection = VariantSelector.select_best_variant(comparison, threshold)
        
        selected_result = selection.get("selected_result", {})
        all_variants = selection.get("all_variants", [])
        
        raw_text = selected_result.get("text", "")
        ocr_confidence = selected_result.get("confidence", 0.0)
        variant_used = selected_result.get("variant_used")
        
        # PHASE 2.3: Clean and validate text
        cleaning_result = TextCleaningPipeline.clean_ocr_text(raw_text)
        cleaned_text = cleaning_result.get("corrected_text", "")
        corrections_made = cleaning_result.get("corrections_made", 0)
        correction_confidence = cleaning_result.get("correction_confidence", 1.0)
        
        # Validate with validate_plate
        clean_cand, is_valid_plate = validate_plate(cleaned_text)
        if not is_valid_plate:
            clean_cand, is_valid_plate = validate_plate(raw_text)
        
        if is_valid_plate and clean_cand:
            final_text = clean_cand
            validation_valid = True
            validation_status = "VALID"
            state_code = clean_cand[:2]
        else:
            final_text = cleaned_text if cleaned_text else raw_text
            validation_valid = False
            validation_status = "INVALID_FORMAT"
            state_code = final_text[:2] if len(final_text) >= 2 and final_text[:2].isalpha() else None
        
        # Validate against Indian plate format for quality scorer
        validation_result = PlateValidator.validate_indian_plate(final_text)
        
        # Score overall quality
        quality_result = PlateQualityScorer.score_plate_quality(
            final_text,
            ocr_confidence,
            validation_result,
            corrections_made,
            correction_confidence
        )

        # Status order:
        # 1. empty text -> OCR_FAILED
        # 2. original crop width < 80px -> LOW_RESOLUTION (still run OCR, never ACCEPTED)
        # 3. valid format and state code -> ACCEPTED
        # 4. otherwise -> NEEDS_REVIEW
        orig_crop_w = original_crop.shape[1]
        min_w_low_res = getattr(config, 'MIN_PLATE_WIDTH_FOR_LOW_RES', 80) if config else 80
        acc_floor = getattr(config, 'ACCEPTANCE_CONFIDENCE_FLOOR', 0.5) if config else 0.5

        if not raw_text.strip() and not final_text.strip():
            final_status = ResultStatus.OCR_FAILED.value
        elif orig_crop_w < min_w_low_res:
            final_status = ResultStatus.LOW_RESOLUTION.value
        elif validation_valid and ocr_confidence >= acc_floor:
            final_status = ResultStatus.ACCEPTED.value
        else:
            final_status = ResultStatus.NEEDS_REVIEW.value
        
        # Build result
        result = {
            "success": final_status == ResultStatus.ACCEPTED.value,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "final_text": final_text,
            "text": final_text,
            "confidence": ocr_confidence,
            "variant_used": variant_used,
            "status": final_status,
            "validation_status": validation_status,
            "validation_valid": validation_valid,
            "state_code": state_code,
            "quality_score": quality_result.get("total_score", 0.0),
            "quality_tier": quality_result.get("quality_tier", "VERY_POOR"),
            "corrections_made": corrections_made,
            "correction_confidence": correction_confidence,
            "is_valid_format": validation_valid,
            "threshold_used": threshold,
            "variants_above_threshold": selection.get("variants_above_threshold", 0),
            "ocr_calls_made": comparison.get("ocr_calls_made", 5),
            "early_stopped": comparison.get("early_stopped", False),
            "early_stop_reason": comparison.get("early_stop_reason", None),
        }
        
        # Include all variants if requested
        if return_all_variants:
            result["all_variants"] = all_variants
        
        # Include cropped image if requested
        if return_crop:
            result["cropped_image"] = original_crop
        
        # ── SMALL-CROP RECOVERY ──────────────────────────────────────────────
        # When the normal pipeline did NOT produce a valid plate, attempt one
        # additional OCR pass with stronger preprocessing (pad + 4× + CLAHE).
        # Triggered only when:
        #   - validation_valid is False, OR
        #   - recognized text is suspiciously short (< 5 chars)
        # Does NOT weaken validation — recovery result is only accepted when
        # validation_valid=True on the recovered text.
        if not validation_valid or len(cleaned_text) < 5:
            recovery = self._recover_small_crop(
                plate_region,
                threshold,
                base_variant=variant_used,
                base_calls=comparison.get("ocr_calls_made", 5)
            )
            if recovery is not None:
                result.update(recovery)
                if return_all_variants and "all_variants" not in result:
                    result["all_variants"] = all_variants
                if return_crop:
                    result["cropped_image"] = original_crop

        return result

    def _recover_small_crop(self, plate_region, threshold, base_variant=None, base_calls=5):
        """
        Recovery pass for small or partial-read crops.
        Applies: 10% padding + 4x Lanczos upscale + CLAHE.
        Returns a partial result dict only when recovery produces a VALID plate;
        otherwise returns None (caller keeps original result unchanged).
        """
        try:
            ph = max(4, int(plate_region.shape[0] * 0.10))
            pw = max(4, int(plate_region.shape[1] * 0.10))
            padded = cv2.copyMakeBorder(
                plate_region, ph, ph, pw, pw, cv2.BORDER_REPLICATE
            )
            nh = padded.shape[0] * 4
            nw = padded.shape[1] * 4
            upscaled = cv2.resize(padded, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

            lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
            recovered_img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]),
                                         cv2.COLOR_LAB2BGR)

            ocr_r   = self.recognize_text(recovered_img)
            rec_txt = ocr_r.get("text", "")
            rec_cf  = ocr_r.get("confidence", 0.0)

            cleaning = TextCleaningPipeline.clean_ocr_text(rec_txt)
            cleaned  = cleaning.get("corrected_text", "")
            val      = PlateValidator.validate_indian_plate(cleaned)

            if not val.get("valid", False):
                return None

            corr_made = cleaning.get("corrections_made", 0)
            corr_conf = cleaning.get("correction_confidence", 1.0)
            quality   = PlateQualityScorer.score_plate_quality(
                cleaned, rec_cf, val, corr_made, corr_conf
            )

            return {
                "success":               rec_cf >= threshold,
                "raw_text":              rec_txt,
                "cleaned_text":          cleaned,
                "final_text":            cleaned,
                "confidence":            round(float(rec_cf), 4),
                "variant_used":          base_variant or "baseline",
                "recovery_applied":      True,
                "recovery_variant":      "recovery_pad4x_clahe",
                "status":                ResultStatus.ACCEPTED.value
                                         if rec_cf >= threshold
                                         else ResultStatus.LOW_CONFIDENCE.value,
                "validation_status":     val["validation_status"],
                "validation_valid":      True,
                "state_code":            val.get("state_code"),
                "quality_score":         quality.get("total_score", 0.0),
                "quality_tier":          quality.get("quality_tier", "GOOD"),
                "corrections_made":      corr_made,
                "correction_confidence": corr_conf,
                "is_valid_format":       True,
                "ocr_calls_made":        base_calls + 1,
                "early_stopped":         False,
                "early_stop_reason":     None,
            }

        except Exception as e:
            print(f"[OCR RECOVERY] Failed: {e}")
            return None
