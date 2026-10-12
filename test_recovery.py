"""
Quick recovery test for partial-read plates (MH20BY3665 case).
Tests 4 recovery approaches with a single OCR call each.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from ocr_module import PlateOCR
from core import detect_plate

MODEL_PATH = "runs/detect/runs/detect/plate_detector_v2/weights/best.pt"

# The known-failure image
IMG_PATH = "dataset_yolo/test/images/car-wbs-MH20CS4946_00000.jpeg"
# We don't have the exact MH20BY3665 image so we use a similarly-sized crop
# from an existing difficult image. The MH20CS4946 plate is 140x34px per eval data.

model = YOLO(MODEL_PATH)
ocr   = PlateOCR()

img = cv2.imread(IMG_PATH)
det = detect_plate(model, img, 0.25)

# Get the raw crop
x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
crop = img[y1:y2, x1:x2].copy()
h, w = crop.shape[:2]
print(f"Crop size: {w}×{h} px")

def pad_and_scale(crop, pad_ratio=0.10, scale=4):
    """Add padding then upscale."""
    ph = int(crop.shape[0] * pad_ratio)
    pw = int(crop.shape[1] * pad_ratio)
    padded = cv2.copyMakeBorder(crop, ph, ph, pw, pw,
                                 cv2.BORDER_REPLICATE)
    nh = int(padded.shape[0] * scale)
    nw = int(padded.shape[1] * scale)
    return cv2.resize(padded, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)

def apply_sharpen(img_bgr):
    k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv2.filter2D(img_bgr, -1, k)

def run_ocr(variant_img):
    r = ocr.recognize_text(variant_img)
    return r.get("text",""), r.get("confidence", 0.0)

from indian_plate_validator import PlateValidator
from ocr_cleaning import TextCleaningPipeline

def full_eval(label, variant_img):
    text, conf = run_ocr(variant_img)
    cleaned = TextCleaningPipeline.clean_ocr_text(text).get("corrected_text","")
    val = PlateValidator.validate_indian_plate(cleaned)
    print(f"  [{label}] raw='{text}'  cleaned='{cleaned}'  conf={conf:.3f}  "
          f"valid={val['valid']}  status={val['validation_status']}")
    return cleaned, conf, val['valid']

print("\n--- Test on existing image (MH20CS4946) ---")
print("A: 10% pad + 4× upscale")
full_eval("A", pad_and_scale(crop, 0.10, 4))

print("B: 10% pad + 6× upscale")
full_eval("B", pad_and_scale(crop, 0.10, 6))

print("C: 10% pad + 4× + CLAHE")
full_eval("C", apply_clahe(pad_and_scale(crop, 0.10, 4)))

print("D: 10% pad + 4× + sharpen")
full_eval("D", apply_sharpen(pad_and_scale(crop, 0.10, 4)))
