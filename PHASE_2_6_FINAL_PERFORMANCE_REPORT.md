# Phase 2.6 — Final Performance Report

**Date:** 2026-09-13  
**Production status:** Early stopping enabled (`early_stopping=True` is now the default)  
**Measurement path:** `MultiPlateProcessor.process_image_multi_plate()` — the actual production code path

---

## Methodology

- Each scenario run 3 times; mean shown
- "Before" = `early_stopping=False` (all 5 PaddleOCR calls unconditionally)
- "After"  = `early_stopping=True`  (stops on first ACCEPTED + valid result)
- Same model, same OCR engine, same confidence threshold (0.85), same images
- Detection time excluded from OCR comparison (detection is constant ~140 ms)
- All text outputs verified identical between before and after

---

## OCR Call Count

| Scenario | Before calls | After calls | Per-plate calls (after) |
|----------|:------------:|:-----------:|-------------------------|
| 1 plate  | 5            | **1**       | plate1=1 (valid → stop) |
| 2 plates | 10           | **6**       | plate1=1 (valid), plate2=5 (invalid, exhausts all) |
| 3 plates | 15           | **7**       | plate1=5 (invalid), plate2=1 (valid), plate3=1 (valid) |

Why plate 2 in the 2-plate scenario uses 5 calls: `H47Y1124` fails all format patterns — no variant produces a valid Indian plate, so early stopping never triggers. The logic is conservative: it never accepts an unverified result.

---

## Latency Table (measured, mean of 3 runs)

| Scenario | Before OCR (ms) | After OCR (ms) | OCR Reduction | Before Total (ms) | After Total (ms) | Total Reduction |
|----------|:---------------:|:--------------:|:-------------:|:-----------------:|:----------------:|:---------------:|
| 1 plate  | 10,626          | **2,137**      | **−79.9%**    | 10,779            | **2,285**        | **−78.8%**      |
| 2 plates | 18,116          | **10,965**     | **−39.5%**    | 18,265            | **11,109**       | **−39.2%**      |
| 3 plates | 28,332          | **13,222**     | **−53.3%**    | 28,479            | **13,379**       | **−53.0%**      |

---

## Per-Plate Timing (1-plate scenario, representative)

| Stage | Before (ms) | After (ms) |
|---|:---:|:---:|
| YOLO detection | 153 | 148 |
| Crop extraction | <1 | <1 |
| PaddleOCR calls | 10,626 (5 × ~2,125) | **2,137 (1 × ~2,137)** |
| Variant selection | <1 | <1 |
| Cleaning | <1 | <1 |
| Validation | <1 | <1 |
| Quality scoring | <1 | <1 |
| **Plate total** | **~10,779** | **~2,285** |

---

## Text Agreement (all scenarios, all plates)

| Plate | Before text | After text | Match |
|---|---|---|:---:|
| DL3CAY9324 (1-plate) | DL3CAY9324 | DL3CAY9324 | ✓ |
| KL10AW2814 (2p, P1) | KL10AW2814 | KL10AW2814 | ✓ |
| H47Y1124 (2p, P2) | H47Y1124 | H47Y1124 | ✓ |
| AP16BS45 (3p, P1) | AP16BS45 | AP16BS45 | ✓ |
| DL3CAY9324 (3p, P2) | DL3CAY9324 | DL3CAY9324 | ✓ |
| MH47Y1124 (3p, P3) | MH47Y1124 | MH47Y1124 | ✓ |

**Text agreement: 6/6 (100%)**  
**Validation agreement: 6/6 (100%)**  
**No valid → invalid regression detected**

---

## Confidence Delta (before vs after)

| Plate | Before conf | After conf | Diff |
|---|:---:|:---:|:---:|
| DL3CAY9324 (1-plate) | 0.9999 | 0.9953 | −0.0046 |
| KL10AW2814 (2p, P1) | 0.9979 | 0.9979 | 0.0000 |
| H47Y1124 (2p, P2) | 0.9990 | 0.9990 | 0.0000 |
| AP16BS45 (3p, P1) | 0.9810 | 0.9810 | 0.0000 |
| DL3CAY9324 (3p, P2) | 0.9950 | 0.9915 | −0.0035 |
| MH47Y1124 (3p, P3) | 0.9511 | 0.9474 | −0.0037 |

Maximum difference: **0.0046** — within normal run-to-run variance.

---

## Early Stopping Behaviour in Production (171 real-world images)

From the full dataset evaluation:

- **Avg OCR calls per plate: 3.54** (down from 5.0, −29%)
- Images where plate 1 stopped at call 1: majority (~100/171)
- Images exhausting all 5 calls: plates failing format validation (~57%)
- Per-plate average time ~7.5 s (reduced from ~11 s baseline)

---

## Conclusion

Early stopping delivers a measured **39–80% OCR time reduction** depending on how many detected plates pass Indian plate format validation. The improvement is:
- **Largest** for clean, standard-format plates (single plate, valid): **−80%**
- **Moderate** for mixed scenarios with some invalid plates: **−39 to −53%**
- **Zero** for plates that fail all format patterns (correct: no false acceptance)

Text and validation results are identical across all tested plates. The change is safe for production use.
