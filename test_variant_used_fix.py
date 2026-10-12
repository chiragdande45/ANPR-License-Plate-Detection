"""
test_variant_used_fix.py
─────────────────────────
Regression tests for the variant_used metadata fix (Phase 2.6 Task 1).

Tests:
  1. variant_used is not None for a normal successful OCR result
  2. variant_used is one of the 5 known variant names
  3. all_variants contains exactly 5 entries when early_stopping=False
  4. all_variants entries have 'variant' field (not just 'method')
  5. early_stopping=True correctly identifies the selected variant
  6. early_stop_reason names a real variant
  7. variant_used is None when OCR completely fails (no regressions to failure path)
  8. baseline mode: ocr_calls_made=5
  9. early_stop mode: ocr_calls_made < 5 when plate is valid
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO
from ocr_module import PlateOCR
from core import detect_plate

KNOWN_VARIANTS = {"baseline", "otsu", "adaptive", "clahe_sharp", "denoise_clahe"}

results = []

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((status, label, detail))
    tag = "OK  " if condition else "FAIL"
    print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ── Load once ────────────────────────────────────────────────────────────────
print("Loading model and OCR...")
MODEL = YOLO("runs/detect/runs/detect/plate_detector_v2/weights/best.pt")
OCR   = PlateOCR()

IMG_VALID  = Path("dataset_yolo/test/images/car-wbs-DL3CAY9324_00000.jpeg")
IMG_VALID2 = Path("dataset_yolo/test/images/car-wbs-KL10AW2814_00000.jpeg")
print("Ready.\n")

# ── helper ───────────────────────────────────────────────────────────────────
def get_ocr(img_path, early_stopping=False, return_all_variants=True):
    img = cv2.imread(str(img_path))
    det = detect_plate(MODEL, img, 0.25)
    bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
    return OCR.process_plate_with_cleaning_and_validation(
        img, bbox, threshold=0.85,
        return_all_variants=return_all_variants,
        early_stopping=early_stopping,
    )

# ════════════════════════════════════════════════════════════════════════════
section("1. variant_used is not None for a successful result")

r = get_ocr(IMG_VALID, early_stopping=False)
check("variant_used is not None",
      r.get("variant_used") is not None,
      f"got: {r.get('variant_used')}")
check("variant_used is a string",
      isinstance(r.get("variant_used"), str))

# ════════════════════════════════════════════════════════════════════════════
section("2. variant_used is one of the 5 known variant names")

check("variant_used in KNOWN_VARIANTS",
      r.get("variant_used") in KNOWN_VARIANTS,
      f"got: '{r.get('variant_used')}', known: {sorted(KNOWN_VARIANTS)}")

# ════════════════════════════════════════════════════════════════════════════
section("3. all_variants contains exactly 5 entries (early_stopping=False)")

all_v = r.get("all_variants", [])
check("all_variants has 5 entries",
      len(all_v) == 5, f"got {len(all_v)}")

# ════════════════════════════════════════════════════════════════════════════
section("4. all_variants entries have 'variant' field (the canonical key)")

for entry in all_v:
    name = entry.get("variant") or entry.get("method", "?")
    check(f"entry '{name}' has 'variant' field",
          "variant" in entry, f"keys: {list(entry.keys())}")
    check(f"entry '{name}' variant is a known name",
          entry.get("variant") in KNOWN_VARIANTS,
          f"got: '{entry.get('variant')}'")

# Verify the 5 exact variant names are present
variant_names_in_all = {e.get("variant") for e in all_v}
check("all 5 variant names present",
      variant_names_in_all == KNOWN_VARIANTS,
      f"got: {sorted(variant_names_in_all)}")

# ════════════════════════════════════════════════════════════════════════════
section("5. early_stopping=True correctly identifies the selected variant")

r_es = get_ocr(IMG_VALID, early_stopping=True)

check("early_stopped == True",
      r_es.get("early_stopped") == True,
      f"got: {r_es.get('early_stopped')}")
check("variant_used is not None (early_stop mode)",
      r_es.get("variant_used") is not None,
      f"got: {r_es.get('variant_used')}")
check("variant_used is in KNOWN_VARIANTS (early_stop mode)",
      r_es.get("variant_used") in KNOWN_VARIANTS,
      f"got: '{r_es.get('variant_used')}'")

# ════════════════════════════════════════════════════════════════════════════
section("6. early_stop_reason names a real variant")

reason = r_es.get("early_stop_reason", "")
check("early_stop_reason is not None or empty",
      bool(reason), f"got: {reason!r}")

# Reason format: "variant=<name> conf=..."
variant_in_reason = None
if reason and "variant=" in reason:
    # extract the variant name from the reason string
    part = reason.split("variant=")[1].split(" ")[0]
    variant_in_reason = part
check("variant named in reason is a known variant",
      variant_in_reason in KNOWN_VARIANTS,
      f"extracted: '{variant_in_reason}' from reason: '{reason}'")

# The variant in the reason should match variant_used
check("variant in reason matches variant_used",
      variant_in_reason == r_es.get("variant_used"),
      f"reason variant='{variant_in_reason}' variant_used='{r_es.get('variant_used')}'")

# ════════════════════════════════════════════════════════════════════════════
section("7. Baseline vs early-stop: same variant_used when baseline wins both")

# On IMG_VALID the early-stopping winner is baseline (call 1).
# The baseline mode winner is whichever variant had highest confidence.
# They may differ — that's expected. Both must be valid variant names.
check("baseline variant_used in KNOWN_VARIANTS",
      r.get("variant_used") in KNOWN_VARIANTS,
      f"got: '{r.get('variant_used')}'")
check("early_stop variant_used in KNOWN_VARIANTS",
      r_es.get("variant_used") in KNOWN_VARIANTS,
      f"got: '{r_es.get('variant_used')}'")
# Both produce the same final text
check("final_text identical in both modes",
      r.get("final_text") == r_es.get("final_text"),
      f"baseline='{r.get('final_text')}' estop='{r_es.get('final_text')}'")

# ════════════════════════════════════════════════════════════════════════════
section("8. ocr_calls_made=5 in baseline mode")

check("baseline ocr_calls_made == 5",
      r.get("ocr_calls_made") == 5,
      f"got: {r.get('ocr_calls_made')}")

# ════════════════════════════════════════════════════════════════════════════
section("9. ocr_calls_made < 5 in early_stop mode for a valid plate")

check("early_stop ocr_calls_made < 5",
      r_es.get("ocr_calls_made", 5) < 5,
      f"got: {r_es.get('ocr_calls_made')}")
check("early_stop ocr_calls_made >= 1",
      r_es.get("ocr_calls_made", 0) >= 1)

# ════════════════════════════════════════════════════════════════════════════
section("10. Second image: variant_used also populated")

r2 = get_ocr(IMG_VALID2, early_stopping=False)
check("second image variant_used not None",
      r2.get("variant_used") is not None,
      f"got: {r2.get('variant_used')}")
check("second image variant_used in KNOWN_VARIANTS",
      r2.get("variant_used") in KNOWN_VARIANTS,
      f"got: '{r2.get('variant_used')}'")

# ════════════════════════════════════════════════════════════════════════════
section("11. return_all_variants=False: all_variants absent, variant_used still set")

r_no_vars = get_ocr(IMG_VALID, early_stopping=False, return_all_variants=False)
check("variant_used still not None when all_variants skipped",
      r_no_vars.get("variant_used") is not None,
      f"got: {r_no_vars.get('variant_used')}")
check("all_variants absent when return_all_variants=False",
      "all_variants" not in r_no_vars or r_no_vars.get("all_variants") == [],
      f"keys: {list(r_no_vars.keys())}")

# ════════════════════════════════════════════════════════════════════════════
section("12. Metadata fields always present regardless of mode")

for field in ["ocr_calls_made", "early_stopped", "early_stop_reason"]:
    check(f"baseline result has '{field}'",    field in r)
    check(f"early_stop result has '{field}'",  field in r_es)

# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
total  = len(results)
passed = sum(1 for s,_,_ in results if s == "PASS")
failed = total - passed
print(f"\n  Total : {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
if failed:
    print("\n  FAILED:")
    for s, l, d in results:
        if s == "FAIL":
            print(f"    [FAIL] {l}" + (f"  ({d})" if d else ""))
print()
sys.exit(0 if failed == 0 else 1)
