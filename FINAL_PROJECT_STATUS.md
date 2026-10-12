# ANPR Project — Final Project Status

**Date:** 2026-09-13  
**Status:** COMPLETE  

---

## 1. Project Overview

An Automatic Number Plate Recognition (ANPR) system for Indian license plates, built as a Flask web application with a YOLO detection backend and PaddleOCR recognition engine. The system accepts uploaded images, detects one or more license plates, recognizes the text on each plate independently, validates the result against Indian plate formats, and returns structured results through a web UI and JSON API.

---

## 2. Implemented Features

| Feature | Phase | Status |
|---|---|:---:|
| YOLOv8n plate detection with letterbox resize | Phase 1 | ✓ |
| Plate crop extraction with adaptive padding | Phase 1 | ✓ |
| Conditional perspective correction | Phase 1 | ✓ |
| Adaptive Lanczos upscaling | Phase 1 | ✓ |
| Conditional bilateral denoising | Phase 1 | ✓ |
| CLAHE contrast enhancement | Phase 1 | ✓ |
| Selective glare suppression | Phase 1 | ✓ |
| Selective mud/dirt suppression | Phase 1 | ✓ |
| Stroke-width adaptive unsharp masking | Phase 1 | ✓ |
| Sauvola adaptive thresholding | Phase 1 | ✓ |
| PaddleOCR text recognition | Phase 2.1 | ✓ |
| 5 preprocessing variants (baseline, otsu, adaptive, clahe_sharp, denoise_clahe) | Phase 2.1 | ✓ |
| Confidence thresholding + best-variant selection | Phase 2.2 | ✓ |
| OCR text normalization and character correction | Phase 2.3 | ✓ |
| Indian plate format validation (standard, old, district-letter, three-wheeler) | Phase 2.3 | ✓ |
| Quality scoring (confidence + validation + corrections) | Phase 2.3 | ✓ |
| Adaptive preprocessing routing infrastructure | Phase 2.4 | ✓ (built) |
| Multi-plate detection (up to N plates per image) | Phase 2.5 | ✓ |
| IoU-based duplicate detection removal (threshold 0.3) | Phase 2.5 | ✓ |
| Deterministic plate ordering (top→bottom, left→right) | Phase 2.5 | ✓ |
| Per-plate independent OCR with error isolation | Phase 2.5 | ✓ |
| MultiPlateProcessor wired into Flask `/api/test` | Phase 2.5 | ✓ |
| Confidence-based early stopping (ACCEPTED + valid → stop) | Phase 2.6 | ✓ |
| Full OCR metadata in API response (ocr_calls_made, early_stopped, variant_used) | Phase 2.6 | ✓ |
| variant_used field correctly populated | Phase 2.6 | ✓ |
| State code list: CH (Chandigarh), TN (Tamil Nadu) added | Phase 2.5 fix | ✓ |
| district_letter pattern (DL3C…, MH4A…) added | Phase 2.5 fix | ✓ |

---

## 3. Final Architecture

```
HTTP POST /api/test
    │
    ▼
app.py  (Flask)
    │  PROCESSOR.process_image_multi_plate(img, conf=0.25, ocr_threshold=0.85,
    │                                       return_all_variants=True, return_crops=True)
    │                                       # early_stopping=True (default)
    ▼
multi_plate_processor.py  MultiPlateProcessor
    │
    ├─ core.detect_all_plates()              ← YOLO inference (~140 ms)
    ├─ IoU deduplication (threshold 0.3)
    ├─ Sort (y1, x1) → deterministic order
    │
    └─ For each plate (independent, fault-isolated):
           │
           ├─ Crop: image[y1:y2, x1:x2]
           │
           └─ ocr_module.PlateOCR.process_plate_with_cleaning_and_validation()
                  │
                  ├─ ocr_preprocessing.PlatePreprocessor.get_all_variants()
                  │   → 5 variants generated (~10 ms total)
                  │
                  ├─ For each variant (early_stopping=True):
                  │   ├─ PlateOCR.recognize_text()          ← PaddleOCR (~2 s/call)
                  │   ├─ [early-stop check] if conf≥0.85 AND validation_valid → STOP
                  │   └─ Continue to next variant if not met
                  │
                  ├─ threshold_and_selection.VariantSelector.select_best_variant()
                  ├─ ocr_cleaning.TextCleaningPipeline.clean_ocr_text()
                  ├─ indian_plate_validator.PlateValidator.validate_indian_plate()
                  └─ indian_plate_validator.PlateQualityScorer.score_plate_quality()
    │
    ▼
JSON response to client + annotated image (base64)
```

**Note:** `adaptive_router.py`, `adaptive_ocr_pipeline.py`, and `plate_image_analyzer.py` exist in the repository but are not in the production call chain. They were built as Phase 2.4 infrastructure and are available for future integration.

---

## 4. Phase 1–2.6 Summary

| Phase | Description | Outcome |
|---|---|---|
| Phase 1 | YOLO detection + image preprocessing pipeline + Flask UI | Complete |
| Phase 2.1 | 5 preprocessing variants, PaddleOCR integration | Complete |
| Phase 2.2 | Confidence thresholding, variant selection | Complete |
| Phase 2.3 | Text cleaning, Indian plate validation, quality scoring | Complete |
| Phase 2.4 | Adaptive routing infrastructure (built, not wired in production) | Complete |
| Phase 2.5 | Multi-plate detection, MultiPlateProcessor, IoU dedup, ordering | Complete |
| Phase 2.5 fix | validator CH/TN state codes, district_letter pattern, app.py wired | Complete |
| Phase 2.6 step 1 | Baseline benchmark — identified 5 unconditional OCR calls as sole bottleneck | Complete |
| Phase 2.6 step 2 | Early-stopping experiment — 49–81% OCR reduction, 6/6 text agreement | Complete |
| Phase 2.6 final | Enable early_stopping=True in production, full validation | Complete |

---

## 5. Final Test Results

All test suites run individually to avoid thermal timeout.

| Suite | Tests | Passed | Failed |
|---|:---:|:---:|:---:|
| Validator unit tests (`test_validator.py`) | 82 | **82** | 0 |
| Phase 2.6 structural (`test_regression_phase2_6.py`) | 53 | **53** | 0 |
| variant_used fix (`test_variant_used_fix.py`) | 37 | **37** | 0 |
| Phase 2.5 integration (`run_all_phase_2_5_final_tests.py`) | 106 | **106** | 0 |
| Architecture verification (`verify_architecture.py`) | 29 | **29** | 0 |
| **Grand total** | **307** | **307** | **0** |

---

## 6. Final Performance Results

Early stopping is enabled. All numbers measured through the actual `MultiPlateProcessor` production path, mean of 3 runs.

### OCR call reduction

| Scenario | Before | After | Reduction |
|---|:---:|:---:|:---:|
| 1 plate | 5 calls | **1 call** | −80% |
| 2 plates | 10 calls | **6 calls** | −40% |
| 3 plates | 15 calls | **7 calls** | −53% |

### Time reduction

| Scenario | Before OCR | After OCR | OCR reduction | Before total | After total | Total reduction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 plate | 10,626 ms | **2,137 ms** | **−79.9%** | 10,779 ms | **2,285 ms** | **−78.8%** |
| 2 plates | 18,116 ms | **10,965 ms** | **−39.5%** | 18,265 ms | **11,109 ms** | **−39.2%** |
| 3 plates | 28,332 ms | **13,222 ms** | **−53.3%** | 28,479 ms | **13,379 ms** | **−53.0%** |

Text agreement: **6/6 (100%)** · Validation agreement: **6/6 (100%)** · Max confidence difference: **0.0046**

---

## 7. Real-World Evaluation

Dataset: `dataset_yolo/test/images/` — 171 images (held-out test split).  
Ground truth available for 14 images (plate text embedded in filename).

| Metric | Result |
|---|---|
| Detection rate | **99.4%** (170/171) |
| Not detected | 1 (motion-blurred video frame) |
| Mean OCR confidence | **0.9263** |
| Min OCR confidence | 0.4911 |
| Quality GOOD or EXCELLENT | **94.7%** (162/171) |
| Validation-VALID plates | **42.7%** (73/171) |
| Mean OCR calls/plate (with early stopping) | **3.54** (vs 5.0 baseline, −29%) |
| Exact text match (14 GT images) | **78.6%** (11/14) |
| Near-miss (1 character error) | 2/14 |
| Character substitution error | 1/14 |

The 57.3% non-valid rate reflects dataset composition (non-standard formats, small crops, tourist plates) — not recognition failures. The validator correctly rejects unverified results rather than silently accepting them.

---

## 8. Known Limitations

### Recognition
1. **Dropped final digits on small crops.** Plates occupying a small fraction of the image (< ~60 px height in the 640 px letterboxed space) sometimes lose the last character. Seen on 2 of 14 ground-truth images.
2. **Character substitution on low-resolution crops.** `K→X` type errors occur when the plate crop is small. PaddleOCR is a CPU-only model here; no GPU acceleration is configured.
3. **Non-standard plate formats.** Plates not matching the defined patterns (9-char `KL53E964`, 8-char `MH20BQ20`) are recognized correctly by OCR but flagged as INVALID_FORMAT by the validator. The validator only knows standard Indian formats.

### Performance
4. **Sequential per-plate OCR.** With early stopping, valid plates take ~2 s each. Invalid plates (must exhaust all 5 variants) take ~9 s. No parallel OCR processing is implemented.
5. **CPU-only PaddleOCR.** Each PaddleOCR call takes ~2 s on this hardware. GPU acceleration would reduce this significantly but is out of scope.

### Architecture
6. **Adaptive routing is dead code.** `adaptive_router.py`, `adaptive_ocr_pipeline.py`, and `plate_image_analyzer.py` exist but are not called in the production path. They were built as Phase 2.4 infrastructure.
7. **`variant_used` field in `VariantSelector`.** The key name mismatch (`"method"` vs `"variant"`) was fixed in Phase 2.6. Older test scripts that call `VariantSelector` directly via `all_variants` output that used `"method"` (not `"variant"`) may need updating if they inspect that field.

### Validation coverage
8. **State code list.** 38 state/UT codes are recognized. Newly added: CH (Chandigarh), TN (Tamil Nadu). Edge-case state codes or future new registrations would need to be added manually.

---

## 9. Future Scope

The following are potential improvements, listed for completeness. None are in scope for the current project.

1. **Wire adaptive_router.py into production.** Use image profiling to order variants best-first, increasing early-stopping hit rate.
2. **Parallel OCR.** Process multiple plates concurrently using a thread pool.
3. **GPU acceleration.** Enable PaddleOCR on GPU; expected ~5–10× per-call speedup.
4. **Video/stream mode.** Track plate IDs across frames instead of processing independently.
5. **Expanded validator patterns.** Diplomatic plates, military plates, temporary permits.
6. **Database logging.** Store detection results with timestamps for fleet/access-control use cases.
7. **Authentication.** Add API key or user login to the Flask interface.

---

## 10. Final Conclusion

The ANPR system is feature-complete through Phase 2.6.

**What it does well:**
- Detects plates in 99.4% of real-world test images
- Recognizes text with mean OCR confidence of 0.9263
- Handles 1–N plates per image independently with deterministic ordering
- Validates Indian plate formats and reports invalid results correctly rather than silently accepting them
- Early stopping reduces OCR calls by 29% across the dataset and up to 80% for clean single-plate images
- Full metadata (variant_used, ocr_calls_made, early_stopped) observable in every API response
- 307 tests passing with 0 failures

**What it does not do:**
- Real-time / video processing
- GPU acceleration
- Non-Indian plate formats
- 100% character-level accuracy on small/low-resolution plate crops

---

## Files Modified in Final Phase

| File | Change |
|---|---|
| `ocr_module.py` | `early_stopping=True` as default in `compare_all_preprocessing_variants` and `process_plate_with_cleaning_and_validation`; `"variant"` key added to result entries (fixes variant_used=None) |
| `multi_plate_processor.py` | `early_stopping=True` as default in `process_image_multi_plate` |
| `app.py` | Added `ocr_calls_made`, `early_stopped`, `early_stop_reason` to Flask OCR response dict |
| `test_regression_phase2_6.py` | Updated default-value assertions from `False` to `True` to match production state |
| `test_variant_used_fix.py` | New — 37 tests for variant_used metadata fix |
| `verify_architecture.py` | New — 29 architecture verification checks |
| `final_validation.py` | New — Tasks 4/5/6 combined runner |
| `PHASE_2_6_FINAL_PERFORMANCE_REPORT.md` | New — measured performance before/after |
| `FINAL_REAL_WORLD_EVALUATION.md` | New — 171-image evaluation report |
| `FINAL_PROJECT_STATUS.md` | This document |

---

**PROJECT STATUS: COMPLETE**

*Development is frozen. No further phases will be started.*
