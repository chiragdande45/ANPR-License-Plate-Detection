import os
import cv2
from ultralytics import YOLO
from ocr_module import PlateOCR
from multi_plate_processor import MultiPlateProcessor

image_path = os.path.abspath(r"dataset_yolo\train\images\304b14b8-dc3c-41cb-94b6-635864e2e964___number-plate-protector-car-safeplate-250x250.jpg.jpeg")

print(f"Loading image from: {image_path}")
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

print("Loading YOLO model...")
model = YOLO("runs/detect/runs/detect/plate_detector_v2/weights/best.pt")

print("Loading PlateOCR...")
ocr = PlateOCR()

print("Initializing MultiPlateProcessor...")
processor = MultiPlateProcessor(model=model, ocr_module=ocr)

print("\n--- RUNNING PIPELINE ---")
result = processor.process_image_multi_plate(
    img,
    conf_threshold=0.25,
    ocr_threshold=0.85,
    require_validation=True,
    return_all_variants=True,
    return_crops=True,
    debug=True
)

print("\n--- RESULTS SUMMARY ---")
print(f"Success: {result['success']}")
print(f"Plate Count: {result['plate_count']}")
for i, plate in enumerate(result['plates']):
    bbox = plate['bbox']
    det_conf = plate['detection_confidence']
    ocr_res = plate['ocr']
    print(f"\n[Plate #{i+1}]")
    print(f"  BBox: {bbox}")
    print(f"  Detection Conf: {det_conf:.4f}")
    print(f"  Crop Dimensions: w={bbox['width']}, h={bbox['height']}")
    print(f"  Raw Text: '{ocr_res.get('raw_text')}'")
    print(f"  Cleaned Text: '{ocr_res.get('cleaned_text')}'")
    print(f"  Final Text: '{ocr_res.get('final_text')}'")
    print(f"  Status: {ocr_res.get('status')}")
    print(f"  OCR Confidence: {ocr_res.get('confidence')}")
    print(f"  Variant Used: {ocr_res.get('variant_used')}")
    print(f"  Validation Valid: {ocr_res.get('validation_valid')}")
    print(f"  Validation Status: {ocr_res.get('validation_status')}")
