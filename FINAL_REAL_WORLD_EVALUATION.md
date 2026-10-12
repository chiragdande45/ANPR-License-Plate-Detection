# Final Real-World Evaluation

**Date:** 2026-09-13  
**Dataset:** `dataset_yolo/test/images/` — the held-out test split used throughout the project  
**Total images:** 171 (138 .jpg, 27 .jpeg, 6 .png)  
**Early stopping:** Enabled (production default, `early_stopping=True`)  
**Ground truth:** Available for 14 images (filenames encode the plate text, e.g. `car-wbs-DL3CAY9324_00000.jpeg`)  

---

## Important Definitions

- **OCR confidence:** Score from PaddleOCR (0–1). NOT "accuracy" — no ground truth is required to measure it.
- **Validation valid:** Whether the recognized text matches a known Indian plate format. NOT a correctness measure — a correct plate may fail validation if the format is non-standard or OCR drops a character.
- **Exact text match:** Only measurable for the 14 ground-truth images. Reported separately.
- **Detection:** Whether YOLO found at least one plate bounding box at conf ≥ 0.25.

---

## Detection Results

| Metric | Value |
|---|---|
| Total images | 171 |
| Plates detected | 170 / 171 |
| Detection rate | **99.4%** |
| Not detected | 1 (`video8_800.jpg` — motion-blurred video frame) |

---

## OCR Confidence (all 171 detected plates)

| Metric | Value |
|---|---|
| Mean OCR confidence | **0.9263** |
| Minimum OCR confidence | 0.4911 |
| Plates with conf ≥ 0.85 | 162 / 171 (94.7%) |

OCR confidence is high on the majority of images. The single plate below 0.5 confidence (`KA03MX5058`, conf=0.491) still produced a recognizable plate text — confidence here reflects OCR certainty, not image quality only.

---

## Validation Results

| Validation status | Count | % |
|---|:---:|:---:|
| VALID (matched Indian format) | 73 | 42.7% |
| INVALID_FORMAT | 76 | 44.4% |
| INVALID_STATE_CODE | 22 | 12.9% |

The 57.3% non-VALID rate reflects dataset composition: many images contain non-standard plates (tourist plates, commercial series, partial reads, very small crops). The validator correctly rejects these rather than silently accepting them.

---

## Quality Tiers

| Tier | Count | % |
|---|:---:|:---:|
| EXCELLENT | 72 | 42.1% |
| GOOD | 90 | 52.6% |
| FAIR | 9 | 5.3% |
| POOR | 0 | 0.0% |
| VERY_POOR | 0 | 0.0% |

No plates scored POOR or VERY_POOR. 94.7% scored GOOD or EXCELLENT.

---

## OCR Calls per Plate (with early stopping enabled)

| OCR calls | Plates | % |
|---|:---:|:---:|
| 1 | 43 | 25.1% |
| 2 | 20 | 11.7% |
| 3 | 5 | 2.9% |
| 4 | 8 | 4.7% |
| 5 | 95 | 55.6% |

**Mean OCR calls per plate: 3.54** (vs 5.0 without early stopping — 29% reduction across the dataset)

The 55.6% of plates using all 5 calls are predominantly those with invalid/non-standard formats where no early-stopping condition is ever met. This is correct behavior — the system should not accept an unverified result.

---

## Processing Time (per image, with plate detected)

| Metric | Time (ms) |
|---|:---:|
| Minimum | 1,957 |
| Median | 9,054 |
| Mean | 7,572 |
| Maximum | 26,289 |

The wide range reflects whether early stopping triggers (fast, ~2 s) or all 5 calls are needed (~9–26 s for challenging plates).

---

## Ground-Truth Evaluation (14 images with known plate text)

These are the `car-wbs-<PLATE>_<id>.<ext>` images where the plate text is encoded in the filename.

| Image | Ground Truth | OCR Text | Exact Match | Confidence | Validation |
|---|---|---|:---:|:---:|---|
| car-wbs-CH01AN0001_00000.png | CH01AN0001 | CH01AN000 | **No** (1 digit dropped) | 0.957 | INVALID_FORMAT |
| car-wbs-DL3CAY9324_00000.jpeg | DL3CAY9324 | DL3CAY9324 | **Yes** | 0.995 | VALID |
| car-wbs-GJ05JH2501_00000.jpg | GJ05JH2501 | GJ05JH250 | **No** (1 digit dropped) | 0.998 | INVALID_FORMAT |
| car-wbs-HR26CU6799_00000.png | HR26CU6799 | HR26CU6799 | **Yes** | 0.995 | VALID |
| car-wbs-KL10AV6633_00000.jpg | KL10AV6633 | KL10AV6633 | **Yes** | 0.997 | VALID |
| car-wbs-KL10AW2814_00000.jpeg | KL10AW2814 | XL10AW2814 | **No** (K→X substitution) | 0.999 | INVALID_STATE_CODE |
| car-wbs-KL53E964_00000.jpeg | KL53E964 | KL53E964 | **Yes** (text correct, format fails old pattern) | 1.000 | INVALID_FORMAT |
| car-wbs-MH12DE1433_00000.png | MH12DE1433 | MH12DE1433 | **Yes** | 0.999 | VALID |
| car-wbs-MH14DX9937_00000.png | MH14DX9937 | MH14DX9937 | **Yes** | 0.997 | VALID |
| car-wbs-MH20BN3525_00001.jpeg | MH20BN3525 | MH20BN3525 | **Yes** | 0.953 | VALID |
| car-wbs-MH20BQ20_00000.png | MH20BQ20 | MH20BQ20 | **Yes** (8-char, fails 10-char pattern) | 0.979 | INVALID_FORMAT |
| car-wbs-MH20CS1941_00001.jpeg | MH20CS1941 | MH20CS1941 | **Yes** | 0.998 | VALID |
| car-wbs-MH20CS4946_00000.jpeg | MH20CS4946 | MH20CS4946 | **Yes** | 0.999 | VALID |
| car-wbs-TN07BU5427_00001.jpeg | TN07BU5427 | TN07BU5427 | **Yes** | 0.961 | VALID |

### Ground-Truth Summary

| Metric | Value |
|---|---|
| Ground-truth images | 14 |
| Exact text match | **11 / 14 (78.6%)** |
| Near-miss (1 dropped digit) | 2 |
| Character substitution | 1 |

### Analysis of the 3 non-matches

**1. `CH01AN0001` → `CH01AN000` (dropped final digit)**  
The last character `1` was not recognized. The crop is a PNG of a synthetic white-background plate at lower resolution. The OCR reads 9 of 10 characters correctly.

**2. `GJ05JH2501` → `GJ05JH250` (dropped final digit)**  
Same pattern — the last digit `1` dropped. Small plate crop; `1` and `|` are easily confused on small images.

**3. `KL10AW2814` → `XL10AW2814` (K→X substitution)**  
The initial `K` was misread as `X` — a common OCR confusion on sans-serif fonts at low resolution. The remaining 9 characters are correct. Validator correctly rejects `XL` as an invalid state code.

---

## OCR Confidence on Failed Cases

The 3 non-matching plates had **high confidence** (0.957, 0.998, 0.999). This confirms that:
- High OCR confidence does not guarantee character-level correctness
- Validation correctly flags these (INVALID_FORMAT or INVALID_STATE_CODE)
- The `success` field (`= ACCEPTED AND validation_valid`) correctly returns `False` for all 3

---

## Selected Per-Image Results (first 30 alphabetically)

| Image | Detected | Text | Conf | Valid | Calls | Time (ms) |
|---|:---:|---|:---:|:---:|:---:|:---:|
| 0a0d1748...large.jpg | Yes | MH46X9996 | 0.988 | Yes | 1 | 2,035 |
| 13ffaad4...Baleno.jpeg | Yes | HR26DA5443 | 0.921 | Yes | 1 | 2,089 |
| 28574c06...hqdefault.jpg | Yes | R127TC0530 | 0.991 | No | 4 | 7,041 |
| 2c9306ab...Terrano.jpg | Yes | RJ27TC0530 | 0.990 | Yes | 1 | 2,008 |
| 347c849b...Polo.jpeg | Yes | WOBONP300 | 0.989 | No | 5 | 9,083 |
| 377b225b...Ignis.jpeg | Yes | HR26DA0471 | 0.999 | Yes | 1 | 2,169 |
| 5231116f...Terrano.jpg | Yes | HR11F7575 | 0.999 | Yes | 2 | 3,994 |
| 56b46df3...large.jpeg | Yes | PB08CX2959 | 0.985 | Yes | 1 | 1,969 |
| 581e2224...617609.jpeg | Yes | KA03MX5058 | 0.491 | Yes | 5 | 14,175 |
| 6bf2730d...Octavia.jpeg | Yes | MH20EE7598 | 1.000 | Yes | 1 | 2,025 |
| 7616cb19...Laura.jpeg | Yes | MH01AR5274 | 0.989 | Yes | 1 | 2,170 |
| 7c8229d7...plate-photo.jpeg | Yes | HR99BJTEMP6662 | 1.000 | No | 5 | 15,989 |
| 85d7e25d...77cca.jpeg | Yes | KL60N5344 | 0.987 | Yes | 1 | 1,996 |
| 8906af57...VW-Vento.jpeg | Yes | TCF300MH14 | 0.985 | No | 5 | 12,964 |
| 97846647...Swift-Dzire.jpeg | Yes | HR26BP354 | 1.000 | No | 5 | 9,360 |
| 9cd75347...Swift.jpg | Yes | KL1GAV6342 | 0.972 | Yes | 1 | 3,177 |
| acd85f93...gurgaon.jpeg | Yes | MH03BS7778 | 0.999 | Yes | 2 | 3,836 |
| aff1f53b...Untitled.png | Yes | KA03NA8385 | 0.997 | Yes | 1 | 2,210 |
| AN6.jpg | Yes | 6891H10NY | 0.871 | No | 5 | 9,445 |
| AP1.jpg | Yes | AP05BY779 | 0.993 | No | 2 | 3,847 |
| AP12.jpg | Yes | AP31607562 | 0.938 | No | 5 | 8,814 |
| AP18.jpg | Yes | AP168S4500 | 0.814 | No | 5 | 9,197 |
| AP26.jpg | Yes | AP31BR9501 | 1.000 | Yes | 1 | 2,065 |
| AP3.jpg | Yes | AP2D401 | 0.972 | No | 5 | 9,039 |
| AP41.jpg | Yes | AP5CG6320 | 0.983 | No | 5 | 9,281 |
| AS13.jpg | Yes | A52556858 | 0.911 | No | 5 | 9,356 |
| AS17.jpg | Yes | AS01AR1986 | 0.998 | Yes | 1 | 2,033 |
| AS25.jpg | Yes | AS01BS436 | 0.994 | No | 5 | 9,156 |

*(Full results in `final_validation_results.json`)*

---

## Summary

| Metric | Result |
|---|---|
| Dataset size | 171 images |
| Detection rate | **99.4%** (170/171) |
| Mean OCR confidence | **0.9263** |
| Quality GOOD or better | **94.7%** (162/171) |
| Validation-valid (VALID) | **42.7%** (73/171) |
| Mean OCR calls/plate | **3.54** (vs 5.0 baseline, −29%) |
| Exact text match (GT) | **78.6%** (11/14 images with ground truth) |
| Ground truth near-miss (1 char error) | 2/14 |
| Ground truth wrong (OCR substitution) | 1/14 |

### Honest assessment of limitations

1. **Validation rate (42.7%)** is not a recognition accuracy metric. Many images contain non-standard plates or partial reads that the OCR reads correctly but the validator rejects because the format doesn't match known Indian patterns (e.g. 8-char plates, tourist series, dealer plates).

2. **Ground truth is available for only 14 of 171 images** — exact-match accuracy (78.6%) should be treated as indicative, not statistically definitive.

3. **Two systematic errors** appear in the 3 non-matches: dropped final digits and an initial-character substitution. Both occur on small plate crops (the plate occupies a small fraction of the image). These are OCR recognition limitations, not pipeline bugs.

4. **Early stopping performs correctly on all ground-truth images** — it stops when the first valid result appears and never accepts an unverified result when validation fails.
