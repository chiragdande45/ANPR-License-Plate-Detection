"""
Flask Web App for ANPR Testing with PaddleOCR
Phase 2.5: Uses MultiPlateProcessor for all plate detection and OCR.
"""
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from ocr_module import PlateOCR
from multi_plate_processor import MultiPlateProcessor
import cv2
import os
import base64
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'test_uploads'
app.config['RESULTS_FOLDER'] = 'test_results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load model
print("Loading YOLO model...")
try:
    MODEL = YOLO("runs/detect/runs/detect/plate_detector_v2/weights/best.pt")
    print("[OK] YOLO model loaded!")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    MODEL = None

# Load OCR
print("Loading OCR module...")
try:
    OCR = PlateOCR()
    print("[OK] OCR module loaded!")
except Exception as e:
    print(f"[ERROR] OCR load failed: {e}")
    OCR = None

# Phase 2.5: Build MultiPlateProcessor from the already-loaded MODEL and OCR.
# This shares the same model/OCR instances — no extra initialisation cost.
print("Building MultiPlateProcessor...")
try:
    PROCESSOR = MultiPlateProcessor(model=MODEL, ocr_module=OCR)
    print("[OK] MultiPlateProcessor ready!")
except Exception as e:
    print(f"[ERROR] MultiPlateProcessor init failed: {e}")
    PROCESSOR = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/test', methods=['POST'])
def test():
    try:
        if PROCESSOR is None:
            return jsonify({"error": "MultiPlateProcessor not initialized"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files['file']
        conf = float(request.form.get('confidence', 0.25))

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save file with unique prefix to avoid collisions in batch/multi-upload
        import uuid
        safe_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        # Read image
        img = cv2.imread(filepath)
        if img is None:
            # Fallback reading in case of non-ascii paths or strange formats
            try:
                file_bytes = np.fromfile(filepath, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                img = None
        if img is None:
            return jsonify({"error": f"Could not read image: {file.filename}"}), 400

        h, w = img.shape[:2]

        # ── Phase 2.5: single entry-point for all plate detection + OCR ────────
        # MultiPlateProcessor handles:
        #   - YOLO detection of ALL plates
        #   - IoU duplicate removal
        #   - deterministic top→bottom / left→right ordering
        #   - independent Phase 2.3/2.4 OCR per plate
        #   - graceful per-plate error isolation
        mp_result = PROCESSOR.process_image_multi_plate(
            img,
            conf_threshold=conf,
            ocr_threshold=0.85,
            require_validation=True,
            return_all_variants=True,   # needed for frontend "Show All Variants"
            return_crops=True,          # needed for cropped plate thumbnails
            debug=False
        )

        # ── Build the detections list in the shape the frontend expects ─────────
        detections = []
        for plate in mp_result.get("plates", []):
            pid   = plate["plate_id"]
            bbox  = plate["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            det_conf = plate["detection_confidence"]
            ocr      = plate["ocr"]

            # Encode cropped plate image to base64 for the frontend
            crop_img = plate.get("crop")
            cropped_base64 = ""
            if crop_img is not None and isinstance(crop_img, np.ndarray) and crop_img.size > 0:
                _, buf = cv2.imencode('.jpg', crop_img)
                cropped_base64 = base64.b64encode(buf).decode('utf-8')

            detection = {
                "id": pid,
                "confidence": det_conf,           # detection confidence (from YOLO)
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": bbox["width"], "h": bbox["height"],
                "cropped_image": f"data:image/jpeg;base64,{cropped_base64}",
                "ocr": {
                    "success":                ocr.get("success", False),
                    "raw_text":               ocr.get("raw_text", ""),
                    "cleaned_text":           ocr.get("cleaned_text", ""),
                    "final_text":             ocr.get("text", ""),
                    "text":                   ocr.get("text", ""),   # backward compat
                    "confidence":             ocr.get("confidence", 0.0),
                    "variant_used":           ocr.get("variant_used"),
                    "status":                 ocr.get("status", "UNKNOWN"),
                    "validation_status":      ocr.get("validation_status", "UNKNOWN"),
                    "validation_valid":       ocr.get("validation_valid", False),
                    "state_code":             ocr.get("state_code"),
                    "quality_score":          ocr.get("quality_score", 0.0),
                    "quality_tier":           ocr.get("quality_tier", "UNKNOWN"),
                    "corrections_made":       ocr.get("corrections_made", 0),
                    "threshold_used":         ocr.get("threshold_used", 0.85),
                    "variants_above_threshold": ocr.get("variants_above_threshold", 0),
                    "is_valid_format":        ocr.get("is_valid_format", False),
                    "all_variants":           ocr.get("all_variants", []),
                    # Phase 2.6 early-stopping metadata
                    "ocr_calls_made":         ocr.get("ocr_calls_made"),
                    "early_stopped":          ocr.get("early_stopped"),
                    "early_stop_reason":      ocr.get("early_stop_reason"),
                    # Difficult-plate recovery metadata
                    "recovery_applied":       ocr.get("recovery_applied", False),
                    "recovery_variant":       ocr.get("recovery_variant", None),
                }
            }
            detections.append(detection)

        # ── Draw annotated result image ─────────────────────────────────────────
        # Use YOLO's own plot() on a fresh prediction for the overlay boxes,
        # then overdraw OCR text ourselves — same as before.
        raw_results = MODEL.predict(filepath, conf=conf, verbose=False)
        img_drawn = raw_results[0].plot()

        for det in detections:
            if det["ocr"]["success"]:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                text    = det["ocr"]["final_text"]
                o_conf  = det["ocr"]["confidence"]
                variant = (
                    det["ocr"].get("recovery_variant")
                    if det["ocr"].get("recovery_applied")
                    else det["ocr"].get("variant_used")
                )

                cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_text = f"{text} ({o_conf*100:.1f}%{' via ' + variant if variant else ''})"
                cv2.putText(img_drawn, display_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{safe_filename}")
        cv2.imwrite(out_path, img_drawn)

        _, buffer = cv2.imencode('.jpg', img_drawn)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        plate_count       = mp_result.get("plate_count", 0)
        avg_conf          = (
            np.mean([d["confidence"] for d in detections])
            if detections else 0.0
        )
        recognized_count  = sum(1 for d in detections if d["ocr"]["success"])

        return jsonify({
            "success":          True,
            "detections":       plate_count,
            "plates":           detections,
            "avg_confidence":   round(float(avg_conf), 4),
            "recognized_count": recognized_count,
            "image_size":       f"{w}x{h}",
            "result_image":     f"data:image/jpeg;base64,{img_base64}",
            # Phase 2.5 timing metadata (informational)
            "processing_time_ms":  mp_result.get("processing_time_ms", 0),
            "detection_time_ms":   mp_result.get("detection_time_ms", 0),
            "ocr_time_ms":         mp_result.get("ocr_time_ms", 0),
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n[OK] Starting Flask app at http://localhost:5000")
    print("[OK] Features: YOLO Plate Detection + PaddleOCR + MultiPlateProcessor")
    app.run(debug=False, host='127.0.0.1', port=5000)
