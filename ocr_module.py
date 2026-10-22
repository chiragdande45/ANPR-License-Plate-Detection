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
        """Initialize PaddleOCR with config detector parameters"""
        try:
            print("[OCR] Initializing PaddleOCR...")
            det_thresh = getattr(config, 'DET_DB_THRESH', 0.2) if config else 0.2
            det_box_thresh = getattr(config, 'DET_DB_BOX_THRESH', 0.4) if config else 0.4
            det_unclip = getattr(config, 'DET_DB_UNCLIP_RATIO', 2.0) if config else 2.0
            # Disable mkldnn to avoid tensor conversion errors on Windows CPU
            # See: https://github.com/PaddlePaddle/PaddleOCR/discussions/13052
            self.ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                enable_mkldnn=False,
                det_db_thresh=det_thresh,
                det_db_box_thresh=det_box_thresh,
                det_db_unclip_ratio=det_unclip
            )
            print("[OCR] PaddleOCR initialized successfully")
        except Exception as e:
            print(f"[OCR ERROR] Failed to initialize PaddleOCR: {e}")
            self.ocr = None

    @staticmethod
    def estimate_plate_tilt(crop):
        """
        Estimate plate tilt using Hough lines on plate edges / text lines
        and cv2.minAreaRect on dark text pixels.
        Returns angle in degrees to rotate by to level the plate.
        """
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        
        # 1. Hough lines on Canny edges
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
        hough_angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0:
                    ang = float(np.degrees(np.arctan2(dy, dx)))
                    while ang > 45: ang -= 90
                    while ang < -45: ang += 90
                    hough_angles.append(ang)
                    
        # 2. Contours of dark text pixels
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bin_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_angles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 50 < area < 10000:
                _, (w, h), a = cv2.minAreaRect(c)
                ang = a + 90 if w < h else a
                while ang > 45: ang -= 90
                while ang < -45: ang += 90
                cnt_angles.append(ang)
                
        all_angles = []
        if hough_angles:
            all_angles.extend(hough_angles)
        if cnt_angles:
            all_angles.extend(cnt_angles)
            
        if all_angles:
            return float(np.median(all_angles))
        return 0.0

    @classmethod
    def deskew_plate(cls, crop):
        """
        Estimate plate tilt and rotate crop to level, keeping the white border
        so nothing is clipped.
        """
        if crop is None or crop.size == 0:
            return crop
        ang = cls.estimate_plate_tilt(crop)
        if abs(ang) < 2.0:
            return crop
        h, w = crop.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2.0) - w / 2.0
        M[1, 2] += (nH / 2.0) - h / 2.0
        deskewed = cv2.warpAffine(crop, M, (nW, nH), borderValue=(255, 255, 255))
        print(f"[DESKEW] Plate tilted by {ang:.2f} deg, rotated to level.")
        return deskewed
    
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
    
    def recognize_text(self, plate_image, variant_name=None):
        """
        Recognize text in plate using PaddleOCR
        
        Args:
            plate_image: Plate region image
            variant_name: Optional name of preprocessing variant for logging
            
        Returns:
            Dictionary with recognition results
        """
        if self.ocr is None:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": "OCR not initialized",
                "has_temp": False,
                "raw_texts": [],
            }
        
        try:
            # Run OCR directly on image
            result = self.ocr.ocr(plate_image)
            
            if not result or len(result) == 0:
                v_label = variant_name if variant_name else "recognize_text"
                print(f"[OCR FILTER] [{v_label}] Raw lines: 0 | After height: 0 | After alnum: 0 | After drop: 0 | Final: ''")
                return {"success": False, "text": "", "confidence": 0.0, "has_temp": False, "raw_texts": []}
            
            ocr_result = result[0]
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [])
            dt_polys = ocr_result.get('dt_polys', [])
            
            # Flag if TEMP appears in any raw OCR line
            has_temp = any('TEMP' in str(t).upper() for t in rec_texts)
            raw_line_count = len(rec_texts)

            # 1. Height filter: Discard text boxes whose height is < 40% of the tallest text box
            lines_after_height = []
            if dt_polys and len(dt_polys) == len(rec_texts):
                heights = []
                for poly in dt_polys:
                    pts = np.array(poly)
                    h = pts[:, 1].max() - pts[:, 1].min()
                    heights.append(h)
                max_h = max(heights) if heights else 0
                for poly, t, s, h in zip(dt_polys, rec_texts, rec_scores, heights):
                    if max_h > 0 and h < 0.40 * max_h:
                        continue  # discard small text
                    pts = np.array(poly)
                    lines_after_height.append({
                        "raw": str(t),
                        "score": s,
                        "y": float(pts[:, 1].min()),
                        "x": float(pts[:, 0].min()),
                    })
            else:
                for idx, (t, s) in enumerate(zip(rec_texts, rec_scores)):
                    lines_after_height.append({
                        "raw": str(t),
                        "score": s,
                        "y": float(idx),
                        "x": 0.0,
                    })

            # 2. Filter lines to [A-Z0-9]
            lines_after_alnum = []
            for item in lines_after_height:
                clean_t = re.sub(r'[^A-Z0-9]', '', item["raw"].upper())
                if clean_t:
                    item_copy = dict(item)
                    item_copy["text"] = clean_t
                    lines_after_alnum.append(item_copy)

            # 3. Drop lines whose text is exactly "IND" or "TEMP"
            kept_lines = []
            for item in lines_after_alnum:
                if item["text"] in ("IND", "TEMP"):
                    continue
                kept_lines.append(item)

            # 4. Default join: sorted by y, then x
            kept_lines.sort(key=lambda item: (item['y'], item['x']))
            default_order = [item['text'] for item in kept_lines]
            default_text = ''.join(default_order)

            # 5. Check validation and line permutations (max 5 lines)
            clean_cand, is_default_valid = validate_plate(default_text)
            winning_order = default_order
            recognized_text = default_text
            order_source = "default"

            if not is_default_valid and 0 < len(kept_lines) <= 5:
                import itertools
                for perm in itertools.permutations(kept_lines):
                    cand_text = ''.join(item['text'] for item in perm)
                    _, cand_valid = validate_plate(cand_text)
                    if cand_valid:
                        winning_order = [item['text'] for item in perm]
                        recognized_text = cand_text
                        order_source = "permutation"
                        break

            if kept_lines:
                print(f"[OCR] Line order won: {winning_order} -> '{recognized_text}' ({order_source})")

            # Logging filter counts for diagnosis
            v_label = variant_name if variant_name else "recognize_text"
            print(f"[OCR FILTER] [{v_label}] Raw lines: {raw_line_count} {[str(t) for t in rec_texts]} "
                  f"| After height: {len(lines_after_height)} {[l['raw'] for l in lines_after_height]} "
                  f"| After alnum: {len(lines_after_alnum)} {[l['text'] for l in lines_after_alnum]} "
                  f"| After IND/TEMP drop: {len(kept_lines)} {[l['text'] for l in kept_lines]} "
                  f"| Final text: '{recognized_text}'")

            avg_confidence = float(np.mean([item['score'] for item in kept_lines])) if kept_lines else 0.0
            success = len(recognized_text.strip()) > 0
            
            return {
                "success": success,
                "text": recognized_text.strip(),
                "confidence": round(avg_confidence, 4),
                "has_temp": has_temp,
                "raw_texts": [str(t) for t in rec_texts],
                "winning_order": winning_order,
                "raw_lines_count": raw_line_count,
            }
        
        except Exception as e:
            print(f"[OCR ERROR] Recognition failed: {e}")
            return {"success": False, "text": "", "confidence": 0.0, "error": str(e), "has_temp": False, "raw_texts": []}
    
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
            # Variant definitions
            primary_variant_names = ["baseline", "clahe_sharp", "denoise_clahe"]
            fallback_variant_names = ["otsu", "adaptive"]

            descriptions = PlatePreprocessor.get_variant_descriptions()

            fn_map = {
                "baseline": PlatePreprocessor.variant_baseline,
                "clahe_sharp": PlatePreprocessor.variant_clahe_sharpening,
                "denoise_clahe": PlatePreprocessor.variant_denoise_clahe,
                "otsu": PlatePreprocessor.variant_otsu_threshold,
                "adaptive": PlatePreprocessor.variant_adaptive_threshold,
            }

            results = []
            best_confidence = 0.0
            best_method = None
            ocr_calls_made = 0
            early_stopped = False
            early_stop_reason = None
            has_temp = False

            def run_single_variant(method_name):
                nonlocal best_confidence, best_method, ocr_calls_made, has_temp
                fn = fn_map.get(method_name)
                processed_image = fn(plate_image) if fn else plate_image

                ocr_result = self.recognize_text(processed_image, variant_name=method_name)
                ocr_calls_made += 1

                conf = ocr_result.get("confidence", 0.0)
                text = ocr_result.get("text", "")
                success = ocr_result.get("success", False)
                v_has_temp = ocr_result.get("has_temp", False)
                if v_has_temp:
                    has_temp = True

                clean_cand, is_v = validate_plate(text)

                result_entry = {
                    "method": method_name,
                    "variant": method_name,
                    "description": descriptions.get(method_name, "Unknown"),
                    "text": text,
                    "confidence": conf,
                    "success": success,
                    "has_temp": v_has_temp,
                    "raw_texts": ocr_result.get("raw_texts", []),
                    "is_valid": is_v,
                    "cleaned_text": clean_cand if is_v else text,
                }
                results.append(result_entry)

                if conf > best_confidence:
                    best_confidence = conf
                    best_method = method_name

                return result_entry

            def check_agreement():
                groups = {}
                for r in results:
                    if r.get("is_valid", False):
                        cand = r.get("cleaned_text")
                        groups.setdefault(cand, []).append(r["variant"])
                for cand, vars_list in groups.items():
                    if len(vars_list) >= 2:
                        return True, cand, vars_list
                return False, None, []

            # 1. Run at most 3 variants (baseline, clahe_sharp, denoise_clahe)
            for m in primary_variant_names:
                run_single_variant(m)
                agreed, agreed_text, agreed_vars = check_agreement()
                if early_stopping and agreed:
                    early_stopped = True
                    early_stop_reason = f"2 variants {agreed_vars} agreed on '{agreed_text}'"
                    break

            # 2. Run otsu / adaptive only if none passes validate_plate
            if not early_stopped:
                has_any_valid = any(r.get("is_valid", False) for r in results)
                if not has_any_valid:
                    print("[OCR] No primary variant passed validation; running fallback variants (otsu, adaptive)...")
                    for m in fallback_variant_names:
                        run_single_variant(m)
                        agreed, agreed_text, agreed_vars = check_agreement()
                        if early_stopping and agreed:
                            early_stopped = True
                            early_stop_reason = f"2 variants {agreed_vars} agreed on '{agreed_text}'"
                            break

            return {
                "variants": results,
                "best_by_confidence": best_method,
                "best_confidence": round(best_confidence, 4),
                "ocr_calls_made": ocr_calls_made,
                "early_stopped": early_stopped,
                "early_stop_reason": early_stop_reason,
                "has_temp": has_temp,
            }

        except Exception as e:
            print(f"[OCR ERROR] Variant comparison failed: {e}")
            return {
                "error": str(e),
                "variants": [],
                "ocr_calls_made": 0,
                "early_stopped": False,
                "early_stop_reason": None,
                "has_temp": False,
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

        # Before OCR, estimate plate tilt, rotate crop to level, and keep white border
        plate_region = self.deskew_plate(plate_region)
        
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
                "reason": "comparison error",
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
        
        # Select best variant based on threshold (prefers valid plates and cross-variant agreement)
        selection = VariantSelector.select_best_variant(comparison, threshold)
        
        selected_result = selection.get("selected_result", {})
        all_variants = selection.get("all_variants", [])
        
        raw_text = selected_result.get("text", "")
        ocr_confidence = selected_result.get("confidence", 0.0)
        variant_used = selected_result.get("variant_used")
        final_status = selected_result.get("status", ResultStatus.NEEDS_REVIEW.value)
        status_reason = selected_result.get("reason", "")
        has_temp = selected_result.get("has_temp", False)
        
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

        orig_crop_w = original_crop.shape[1]
        min_w_low_res = getattr(config, 'MIN_PLATE_WIDTH_FOR_LOW_RES', 80) if config else 80

        if not raw_text.strip() and not final_text.strip():
            final_status = ResultStatus.OCR_FAILED.value
            status_reason = "empty text"
        elif orig_crop_w < min_w_low_res:
            final_status = ResultStatus.LOW_RESOLUTION.value
            status_reason = "low resolution"
        
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
            "reason": status_reason,
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
        text_before_recovery = final_text
        status_before_recovery = final_status
        if not validation_valid or len(cleaned_text) < 5:
            print(f"\n[OCR RECOVERY] Attempting recovery for plate with text: '{text_before_recovery}' (status: {status_before_recovery})")
            recovery = self._recover_small_crop(
                plate_region,
                threshold,
                base_variant=variant_used,
                base_calls=comparison.get("ocr_calls_made", 5)
            )
            if recovery is not None:
                text_after_recovery = recovery.get("final_text", "")
                if has_temp and recovery.get("status") == ResultStatus.ACCEPTED.value:
                    recovery["status"] = ResultStatus.NEEDS_REVIEW.value
                    recovery["reason"] = "contains TEMP"
                status_after_recovery = recovery.get("status", "")
                print(f"[OCR RECOVERY] Plate text BEFORE recovery: '{text_before_recovery}' -> AFTER recovery: '{text_after_recovery}'")
                print(f"[OCR RECOVERY] What recovery changes: text updated from '{text_before_recovery}' to '{text_after_recovery}', status from '{status_before_recovery}' to '{status_after_recovery}', validation={recovery.get('validation_status')}")
                result.update(recovery)
                result["reason"] = recovery.get("reason", "recovered valid plate")
                if return_all_variants and "all_variants" not in result:
                    result["all_variants"] = all_variants
                if return_crop:
                    result["cropped_image"] = original_crop
            else:
                print(f"[OCR RECOVERY] Plate text BEFORE recovery: '{text_before_recovery}' -> AFTER recovery: '{text_before_recovery}' (recovery failed to produce valid plate; no changes)")

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
