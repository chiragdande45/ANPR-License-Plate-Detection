"""
Targeted regression check for Phase 2.6 Step 2 changes.
Tests that:
1. early_stopping=False (default) returns exactly the same fields as before
2. early_stopping=True returns the same text/valid/conf + extra metadata
3. app.py structural checks (no OCR calls, just code inspection)
4. Multi-plate processor still initialises and works
5. OCR calls_made=5 when early_stopping=False
6. Metadata fields present in both modes
"""
import sys
import cv2
import inspect
from pathlib import Path

results = []

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((status, label, detail))
    tag = "OK  " if condition else "FAIL"
    print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")

# ── Load model + OCR once ────────────────────────────────────────────────────
print("Loading model + OCR...")
from ultralytics import YOLO
from ocr_module import PlateOCR
from core import detect_plate
from multi_plate_processor import MultiPlateProcessor

MODEL = YOLO("runs/detect/runs/detect/plate_detector_v2/weights/best.pt")
OCR   = PlateOCR()

IMG_PATH = Path("dataset_yolo/test/images/car-wbs-DL3CAY9324_00000.jpeg")
img = cv2.imread(str(IMG_PATH))
det = detect_plate(MODEL, img, 0.25)
bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
print("Ready.\n")

# ── Section 1: Signature checks ───────────────────────────────────────────────
section("1. Signature: new parameters exist with correct defaults")

import inspect as _ins
sig_compare = _ins.signature(OCR.compare_all_preprocessing_variants)
sig_process  = _ins.signature(OCR.process_plate_with_cleaning_and_validation)

check("compare_all_preprocessing_variants has early_stopping param",
      "early_stopping" in sig_compare.parameters)
check("compare_all_preprocessing_variants early_stopping default=True (production enabled)",
      sig_compare.parameters["early_stopping"].default == True)
check("compare_all_preprocessing_variants has early_stop_threshold param",
      "early_stop_threshold" in sig_compare.parameters)
check("compare_all_preprocessing_variants early_stop_threshold default=None",
      sig_compare.parameters["early_stop_threshold"].default is None)

check("process_plate_with_cleaning_and_validation has early_stopping param",
      "early_stopping" in sig_process.parameters)
check("process_plate_with_cleaning_and_validation early_stopping default=True (production enabled)",
      sig_process.parameters["early_stopping"].default == True)

from multi_plate_processor import MultiPlateProcessor as _MP
sig_mp  = _ins.signature(_MP.process_image_multi_plate)
sig_spp = _ins.signature(_MP._process_single_plate)

check("process_image_multi_plate has early_stopping param",
      "early_stopping" in sig_mp.parameters)
check("process_image_multi_plate early_stopping default=True (production enabled)",
      sig_mp.parameters["early_stopping"].default == True)
check("_process_single_plate has early_stopping param",
      "early_stopping" in sig_spp.parameters)

# ── Section 2: Baseline mode (early_stopping=False) ──────────────────────────
section("2. Baseline mode (early_stopping=False) — fields and values unchanged")

r_base = OCR.process_plate_with_cleaning_and_validation(
    img, bbox, threshold=0.85, return_all_variants=False, early_stopping=False
)

# Core fields still present
for field in ["success", "raw_text", "cleaned_text", "final_text", "confidence",
              "variant_used", "status", "validation_status", "validation_valid",
              "quality_score", "quality_tier", "corrections_made",
              "correction_confidence", "is_valid_format", "threshold_used",
              "variants_above_threshold"]:
    check(f"baseline result has '{field}'", field in r_base)

# New metadata fields present
check("baseline result has 'ocr_calls_made'", "ocr_calls_made" in r_base)
check("baseline result has 'early_stopped'",  "early_stopped"  in r_base)
check("baseline result has 'early_stop_reason'", "early_stop_reason" in r_base)

# Values correct for baseline
check("baseline ocr_calls_made == 5",        r_base.get("ocr_calls_made") == 5,
      f"got {r_base.get('ocr_calls_made')}")
check("baseline early_stopped == False",      r_base.get("early_stopped") == False)
check("baseline early_stop_reason == None",   r_base.get("early_stop_reason") is None)
check("baseline success == True (DL3CAY9324)", r_base.get("success") == True,
      f"got {r_base.get('success')}")
check("baseline final_text == 'DL3CAY9324'",
      r_base.get("final_text") == "DL3CAY9324",
      f"got '{r_base.get('final_text')}'")
check("baseline validation_valid == True",    r_base.get("validation_valid") == True)

# ── Section 3: Experiment mode (early_stopping=True) ─────────────────────────
section("3. Experiment mode (early_stopping=True) — correct behavior")

r_estop = OCR.process_plate_with_cleaning_and_validation(
    img, bbox, threshold=0.85, return_all_variants=False, early_stopping=True
)

check("estop result has all core fields",
      all(f in r_estop for f in ["final_text", "confidence", "validation_valid",
                                  "quality_score", "ocr_calls_made",
                                  "early_stopped", "early_stop_reason"]))
check("estop final_text == 'DL3CAY9324'",
      r_estop.get("final_text") == "DL3CAY9324",
      f"got '{r_estop.get('final_text')}'")
check("estop validation_valid == True",       r_estop.get("validation_valid") == True)
check("estop early_stopped == True",          r_estop.get("early_stopped") == True,
      f"got {r_estop.get('early_stopped')}")
check("estop ocr_calls_made < 5",
      r_estop.get("ocr_calls_made", 5) < 5,
      f"got {r_estop.get('ocr_calls_made')}")
check("estop early_stop_reason is not None",  r_estop.get("early_stop_reason") is not None)
check("estop text matches baseline",
      r_base.get("final_text") == r_estop.get("final_text"),
      f"base='{r_base.get('final_text')}' estop='{r_estop.get('final_text')}'")
check("estop validation matches baseline",
      r_base.get("validation_valid") == r_estop.get("validation_valid"))

# Confidence: early-stop takes the first variant (baseline), baseline takes
# best-of-5.  Should be within reasonable range (not degraded by >5%).
conf_diff = abs(r_base.get("confidence", 0) - r_estop.get("confidence", 0))
check("estop confidence within 5% of baseline",
      conf_diff <= 0.05,
      f"diff={conf_diff:.4f} base={r_base.get('confidence'):.4f} estop={r_estop.get('confidence'):.4f}")

# ── Section 4: No dangerous degradation ──────────────────────────────────────
section("4. No dangerous degradation")

check("baseline valid, estop also valid (no valid→invalid regression)",
      r_base.get("validation_valid") == r_estop.get("validation_valid"))
check("estop text identical to baseline",
      r_base.get("final_text") == r_estop.get("final_text"))
check("estop ocr_status is ACCEPTED",
      r_estop.get("status") == "ACCEPTED",
      f"got {r_estop.get('status')}")

# ── Section 5: app.py structural check ───────────────────────────────────────
section("5. app.py structural: default path unchanged")

import flask as _flask
_flask.Flask.run = lambda *a, **k: None
import app as flask_app

check("app.PROCESSOR is MultiPlateProcessor",
      isinstance(flask_app.PROCESSOR, MultiPlateProcessor))
src = inspect.getsource(flask_app.test)
check("app.test() calls process_image_multi_plate",
      "process_image_multi_plate" in src)
check("app.test() uses early_stopping via default=True (no explicit override needed)",
      "early_stopping=False" not in src,
      "early stopping is now default=True; app.py inherits it silently")

# ── Section 6: MultiPlateProcessor threading ─────────────────────────────────
section("6. MultiPlateProcessor baseline mode produces ocr_calls_made=5")

PROC = MultiPlateProcessor(model=MODEL, ocr_module=OCR)
result = PROC.process_image_multi_plate(img, conf_threshold=0.25,
                                         early_stopping=False)
check("process_image_multi_plate returns 1 plate",
      result.get("plate_count") == 1,
      f"got {result.get('plate_count')}")

if result.get("plates"):
    p = result["plates"][0]
    check("plate.ocr.ocr_calls_made == 5 in baseline mode",
          p["ocr"].get("ocr_calls_made") == 5,
          f"got {p['ocr'].get('ocr_calls_made')}")
    check("plate.ocr.early_stopped == False in baseline mode",
          p["ocr"].get("early_stopped") == False)
    check("plate.ocr.text == 'DL3CAY9324'",
          p["ocr"].get("text") == "DL3CAY9324",
          f"got '{p['ocr'].get('text')}'")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print("  SUMMARY")
print(f"{'='*68}")
total  = len(results)
passed = sum(1 for s,_,_ in results if s=="PASS")
failed = total - passed
print(f"\n  Total : {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
if failed:
    print("\n  FAILED:")
    for s,l,d in results:
        if s=="FAIL":
            print(f"    [FAIL] {l}" + (f"  ({d})" if d else ""))
print()
sys.exit(0 if failed == 0 else 1)
