"""
Phase 2.5: Multi-Plate Processing Orchestrator

Processes multiple detected plates independently using the Phase 2.4 adaptive OCR pipeline.

Architecture:
1. detect_all_plates() → List of detections
2. For each detection:
   - Crop plate from image
   - Apply Phase 2.4 adaptive OCR
   - Collect result
3. Order plates deterministically (top→bottom, left→right)
4. Return results with metadata

Does NOT create new preprocessing - uses existing Phase 2.4 pipeline.
"""

import sys
sys.path.insert(0, r'c:\Projects\ANPR-Project_Code_File\ANPR')

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from core import detect_all_plates
from ocr_module import PlateOCR
try:
    import config
except ImportError:
    config = None


class MultiPlateProcessor:
    """Orchestrates detection and independent OCR processing of multiple plates."""
    
    # NMS/Deduplication settings
    IOU_THRESHOLD_DUPLICATE = 0.3  # If IoU > this, consider boxes duplicates
    
    def __init__(self, model=None, ocr_module=None):
        """
        Initialize multi-plate processor.
        
        Args:
            model: YOLO model (ultralytics YOLO object)
            ocr_module: PlateOCR instance (if None, will be created)
        """
        self.model = model
        self.ocr = ocr_module or PlateOCR()
    
    def process_image_multi_plate(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        ocr_threshold: float = 0.85,
        require_validation: bool = True,
        return_all_variants: bool = False,
        return_crops: bool = False,
        early_stopping: bool = True,
        debug: bool = False
    ) -> Dict:
        """
        Process image for multiple plates.
        
        Pipeline:
        1. Detect all plates (YOLO)
        2. Sort deterministically (top→bottom, left→right)
        3. Assign plate IDs
        4. For each plate: crop and OCR independently
        5. Return all results
        
        Args:
            image_bgr: Input image (BGR format from OpenCV)
            conf_threshold: YOLO detection confidence threshold (0-1)
            ocr_threshold: OCR confidence threshold (0-1)
            require_validation: Require plate format validation
            debug: Print debug messages
            
        Returns:
            Dict with:
            {
                "success": bool,
                "plate_count": int,
                "plates": [
                    {
                        "plate_id": int,
                        "detection_confidence": float,
                        "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
                        "crop": np.ndarray or None,
                        "ocr": {...phase 2.4 result...}
                    },
                    ...
                ],
                "processing_time_ms": float,
                "detection_time_ms": float,
                "ocr_time_ms": float
            }
        """
        
        start_time = time.time()

        conf_val = getattr(config, 'DEFAULT_DETECTION_CONF', conf_threshold) if config else conf_threshold
        iou_val = getattr(config, 'IOU_THRESHOLD', 0.5) if config else 0.5
        agnostic_val = getattr(config, 'AGNOSTIC_NMS', True) if config else True
        print(f"PIPELINE v2 ACTIVE: DEFAULT_DETECTION_CONF={conf_val}, IOU_THRESHOLD={iou_val}, AGNOSTIC_NMS={agnostic_val}")
        
        # STEP 1: Detect all plates
        if debug:
            print(f"[MultiPlate] Processing image {image_bgr.shape}")
        
        detection_start = time.time()
        
        if self.model is None:
            if debug:
                print("[MultiPlate] ERROR: Model not initialized")
            return {
                "success": False,
                "plate_count": 0,
                "plates": [],
                "error": "Model not initialized",
                "processing_time_ms": 0,
                "detection_time_ms": 0,
                "ocr_time_ms": 0
            }
        
        detection_result = detect_all_plates(
            self.model,
            image_bgr,
            conf_threshold=conf_threshold
        )
        
        detection_time = (time.time() - detection_start) * 1000
        
        detections = detection_result.get("detections", [])
        
        if not detections:
            if debug:
                print("[MultiPlate] No plates detected")
            return {
                "success": False,
                "plate_count": 0,
                "plates": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
                "detection_time_ms": detection_time,
                "ocr_time_ms": 0
            }
        
        if debug:
            print(f"[MultiPlate] Detected {len(detections)} plate(s)")
        
        # STEP 2: Remove overlapping/duplicate detections using IoU + containment suppression
        deduplicated_detections = self._remove_duplicate_detections(detections)
        
        # STEP 2.5: Sort deterministically (top→bottom, left→right)
        sorted_detections = self._sort_plates_deterministically(deduplicated_detections)
        
        if debug:
            if len(deduplicated_detections) < len(detections):
                print(f"[MultiPlate] Removed {len(detections) - len(deduplicated_detections)} "
                      f"duplicate detection(s)")
        
        # STEP 3: Process each plate
        ocr_start = time.time()
        plates = []
        
        for plate_id, detection in enumerate(sorted_detections, 1):
            plate_result = self._process_single_plate(
                image_bgr,
                detection,
                plate_id,
                ocr_threshold,
                require_validation,
                return_all_variants,
                return_crops,
                early_stopping,
                debug
            )
            plates.append(plate_result)
        
        ocr_time = (time.time() - ocr_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "plate_count": len(plates),
            "plates": plates,
            "processing_time_ms": round(total_time, 2),
            "detection_time_ms": round(detection_time, 2),
            "ocr_time_ms": round(ocr_time, 2),
            "avg_time_per_plate_ms": round(ocr_time / len(plates), 2) if plates else 0
        }
    
    def _sort_plates_deterministically(self, detections: List[Dict]) -> List[Dict]:
        """
        Sort plates deterministically: top→bottom, then left→right.
        
        Args:
            detections: List of detection dicts with x1, y1, x2, y2
            
        Returns:
            Sorted list of detections
        """
        # Sort by: y1 (top position) first, then x1 (left position)
        sorted_dets = sorted(
            detections,
            key=lambda d: (d['y1'], d['x1'])
        )
        return sorted_dets
    
    def _compute_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.
        
        Args:
            box1, box2: Dict with x1, y1, x2, y2
            
        Returns:
            IoU score (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
        x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _remove_duplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove overlapping/duplicate detections using IoU threshold and containment suppression.
        
        Strategy:
        - Sort candidates by confidence descending
        - Higher confidence box is prioritized
        - Discard if IoU > threshold
        - Discard if >80% of lower-confidence box area lies inside higher-confidence box (or vice versa)
        """
        if not detections:
            return []
        
        sorted_by_conf = sorted(detections, key=lambda d: d.get('confidence', 0.0), reverse=True)
        kept = []
        iou_thresh = getattr(config, 'IOU_THRESHOLD', self.IOU_THRESHOLD_DUPLICATE) if config else self.IOU_THRESHOLD_DUPLICATE
        
        for det in sorted_by_conf:
            drop = False
            for k in kept:
                # 1. IoU check
                if self._compute_iou(det, k) > iou_thresh:
                    drop = True
                    break
                
                # 2. Containment suppression
                xi1 = max(det['x1'], k['x1'])
                yi1 = max(det['y1'], k['y1'])
                xi2 = min(det['x2'], k['x2'])
                yi2 = min(det['y2'], k['y2'])
                inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                det_area = max(1e-6, (det['x2'] - det['x1']) * (det['y2'] - det['y1']))
                k_area = max(1e-6, (k['x2'] - k['x1']) * (k['y2'] - k['y1']))
                
                if (inter_area / det_area > 0.80) or (inter_area / k_area > 0.80):
                    drop = True
                    break
            
            if not drop:
                kept.append(det)
        
        return kept
    
    def _process_single_plate(
        self,
        image_bgr: np.ndarray,
        detection: Dict,
        plate_id: int,
        ocr_threshold: float,
        require_validation: bool,
        return_all_variants: bool,
        return_crops: bool,
        early_stopping: bool,
        debug: bool
    ) -> Dict:
        """
        Process a single detected plate.
        
        Args:
            image_bgr: Full image
            detection: Detection dict from detect_all_plates()
            plate_id: Assigned plate ID (1, 2, 3, ...)
            ocr_threshold: OCR confidence threshold
            require_validation: Require plate format validation
            debug: Print debug messages
            
        Returns:
            Dict with plate result including OCR output
        """
        
        x1 = int(detection['x1'])
        y1 = int(detection['y1'])
        x2 = int(detection['x2'])
        y2 = int(detection['y2'])
        
        if debug:
            print(f"[MultiPlate] Processing plate {plate_id}: bbox=({x1},{y1},{x2},{y2})")
        
        # Expand bounding box using config.PADDING_PERCENT for better OCR (e.g. 10%)
        h, w = image_bgr.shape[:2]
        pad_ratio = getattr(config, 'PADDING_PERCENT', 0.10) if config else 0.10
        pad_x = max(4, int((x2 - x1) * pad_ratio))
        pad_y = max(4, int((y2 - y1) * pad_ratio))
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        
        # Crop plate
        try:
            crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if crop.size == 0:
                if debug:
                    print(f"[MultiPlate] ERROR: Empty crop for plate {plate_id}")
                crop = None
        except Exception as e:
            if debug:
                print(f"[MultiPlate] ERROR cropping plate {plate_id}: {e}")
            crop = None
        
        # Run OCR (Phase 2.3 pipeline with Phase 2.4 adaptive routing)
        try:
            ocr_result = self.ocr.process_plate_with_cleaning_and_validation(
                image_bgr,
                [crop_x1, crop_y1, crop_x2, crop_y2],
                return_crop=return_crops,
                threshold=ocr_threshold,
                return_all_variants=return_all_variants,
                early_stopping=early_stopping,
            )
        except Exception as e:
            if debug:
                print(f"[MultiPlate] ERROR processing OCR for plate {plate_id}: {e}")
            ocr_result = {
                "success": False,
                "error": str(e),
                "final_text": "",
                "confidence": 0.0,
                "validation_status": "OCR_ERROR"
            }
        
        # Build plate result
        plate_result = {
            "plate_id": plate_id,
            "detection_confidence": round(float(detection['confidence']), 4),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "crop": ocr_result.get("cropped_image") if return_crops else crop,
            "ocr": {
                "success": ocr_result.get("success", False),
                "text": ocr_result.get("final_text", "") or ocr_result.get("text", ""),
                "final_text": ocr_result.get("final_text", "") or ocr_result.get("text", ""),
                "raw_text": ocr_result.get("raw_text", ""),
                "cleaned_text": ocr_result.get("cleaned_text", ""),
                "confidence": round(float(ocr_result.get("confidence", 0.0)), 4),
                "validation_status": ocr_result.get("validation_status", "UNKNOWN"),
                "validation_valid": ocr_result.get("validation_valid", False),
                "quality_score": round(float(ocr_result.get("quality_score", 0.0)), 4),
                "quality_tier": ocr_result.get("quality_tier", "UNKNOWN"),
                "variant_used": ocr_result.get("variant_used"),
                "state_code": ocr_result.get("state_code"),
                "status": ocr_result.get("status", "UNKNOWN"),
                "reason": ocr_result.get("reason", ""),
                "threshold_used": ocr_result.get("threshold_used", ocr_threshold),
                "variants_above_threshold": ocr_result.get("variants_above_threshold", 0),
                "is_valid_format": ocr_result.get("is_valid_format", False),
                "corrections_made": ocr_result.get("corrections_made", 0),
                "all_variants": ocr_result.get("all_variants", []),
                # Early-stopping metadata
                "ocr_calls_made": ocr_result.get("ocr_calls_made", 5),
                "early_stopped": ocr_result.get("early_stopped", False),
                "early_stop_reason": ocr_result.get("early_stop_reason", None),
                # Difficult-plate recovery metadata
                "recovery_applied": ocr_result.get("recovery_applied", False),
                "recovery_variant": ocr_result.get("recovery_variant", None),
            }
        }
        
        if debug:
            print(f"[MultiPlate] Plate {plate_id} → {plate_result['ocr']['text']} "
                  f"({plate_result['ocr']['confidence']*100:.1f}%)")
        
        return plate_result


# Module-level convenience function

def process_image_multi_plate(
    image_bgr: np.ndarray,
    model,
    conf_threshold: float = 0.25,
    ocr_threshold: float = 0.85,
    debug: bool = False
) -> Dict:
    """
    Convenience function: Process image for multiple plates.
    
    Args:
        image_bgr: Input image (BGR)
        model: YOLO model
        conf_threshold: Detection confidence threshold
        ocr_threshold: OCR confidence threshold
        debug: Print debug info
        
    Returns:
        Multi-plate result dict
    """
    processor = MultiPlateProcessor(model=model)
    return processor.process_image_multi_plate(
        image_bgr,
        conf_threshold=conf_threshold,
        ocr_threshold=ocr_threshold,
        debug=debug
    )
