"""
Flask Web App for ANPR Testing with PaddleOCR
Redesigned Detect UI + REST API / Export Layer
Phase 2.5 MultiPlateProcessor Pipeline
"""
import os
import cv2
import time
import uuid
import base64
import shutil
import threading
import io
import csv
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from ultralytics import YOLO
from ocr_module import PlateOCR
from multi_plate_processor import MultiPlateProcessor
from ocr_preprocessing import PlatePreprocessor
from indian_plate_validator import IndianStateCodeValidator, validate_plate

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Total request limit
app.config['UPLOAD_FOLDER'] = 'test_uploads'
app.config['RESULTS_FOLDER'] = 'test_results'
app.config['STATIC_RESULTS'] = os.path.join(app.root_path, 'static', 'results')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_RESULTS'], exist_ok=True)

# Threading lock around pipeline calls
PIPELINE_LOCK = threading.Lock()

# In-memory batch cache for export
BATCH_CACHE = {}
BATCH_ORDER = []
MAX_CACHED_BATCHES = 50

# Load YOLO model once at startup
print("Loading YOLO model...")
try:
    MODEL = YOLO("runs/detect/runs/detect/plate_detector_v2/weights/best.pt")
    print("[OK] YOLO model loaded!")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    MODEL = None

# Load OCR module once at startup
print("Loading OCR module...")
try:
    OCR = PlateOCR()
    print("[OK] OCR module loaded!")
except Exception as e:
    print(f"[ERROR] OCR load failed: {e}")
    OCR = None

# MultiPlateProcessor singleton
print("Building MultiPlateProcessor...")
try:
    PROCESSOR = MultiPlateProcessor(model=MODEL, ocr_module=OCR)
    print("[OK] MultiPlateProcessor ready!")
except Exception as e:
    print(f"[ERROR] MultiPlateProcessor init failed: {e}")
    PROCESSOR = None


def cleanup_old_results():
    """Delete result folders older than 1 hour on each request."""
    try:
        results_dir = app.config['STATIC_RESULTS']
        if not os.path.exists(results_dir):
            return
        now = time.time()
        for entry in os.scandir(results_dir):
            if entry.is_dir():
                try:
                    if now - entry.stat().st_mtime > 3600:
                        shutil.rmtree(entry.path, ignore_errors=True)
                except Exception:
                    pass
    except Exception as e:
        print(f"[CLEANUP WARNING] {e}")


def format_plain_english_reason(status, raw_reason, ocr_data, orig_crop_w):
    """
    Format a concise, plain-English reason sentence from existing pipeline data.
    Does NOT change pipeline logic or status.
    """
    status = str(status or "").upper()
    raw_reason_str = str(raw_reason or "")

    if status == "ACCEPTED":
        return "Valid format, 2 variants agreed"
    elif status == "LOW_RESOLUTION":
        w = orig_crop_w or ocr_data.get("bbox", {}).get("width", 0)
        return f"Plate is only {w}px wide, too small to read reliably"
    elif status == "OCR_FAILED":
        return "No readable text found on the plate"
    elif status == "NEEDS_REVIEW":
        if ocr_data.get("has_temp") or "TEMP" in raw_reason_str.upper():
            return "Temporary plate (TEMP), check manually"
        if "single variant" in raw_reason_str.lower():
            return "Only one variant produced this text"
        if not ocr_data.get("validation_valid", False):
            return "Text does not match the Indian plate format"
        if "no agreement" in raw_reason_str.lower() or "disagreed" in raw_reason_str.lower():
            return "Preprocessing variants disagreed on plate text"
        return "Needs manual review"
    return raw_reason_str or "Status pending review"


def sanitize_csv_cell(val):
    """Prevent CSV injection by prefixing =, +, -, @ with an apostrophe."""
    s = str(val if val is not None else "")
    if s and s[0] in ('=', '+', '-', '@'):
        return "'" + s
    return s


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/api/detect', methods=['POST'])
def api_detect():
    cleanup_old_results()

    if PROCESSOR is None:
        return jsonify({"error": "ANPR pipeline not initialized on server."}), 500

    # Get uploaded files from "images" field
    uploaded_files = request.files.getlist("images")
    if not uploaded_files or len(uploaded_files) == 0 or (len(uploaded_files) == 1 and uploaded_files[0].filename == ''):
        return jsonify({"error": "No image files provided in request."}), 400

    # Max 10 files check
    if len(uploaded_files) > 10:
        return jsonify({"error": "Maximum 10 files allowed per request."}), 400

    allowed_exts = {'jpg', 'jpeg', 'png', 'webp'}
    max_file_size = 10 * 1024 * 1024  # 10 MB

    # Server-side validation
    for file in uploaded_files:
        fn = file.filename or ""
        ext = fn.rsplit('.', 1)[-1].lower() if '.' in fn else ''
        if ext not in allowed_exts:
            return jsonify({
                "error": f"File '{fn}' has invalid extension. Allowed extensions: jpg, jpeg, png, webp."
            }), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > max_file_size:
            return jsonify({
                "error": f"File '{fn}' exceeds the 10 MB limit ({size / (1024 * 1024):.2f} MB)."
            }), 400

    # Create random batch folder
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = os.path.join(app.config['STATIC_RESULTS'], batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    batch_images = []
    total_plates = 0
    accepted_count = 0
    needs_review_count = 0
    low_res_count = 0
    failed_count = 0
    batch_start_time = time.time()

    for file in uploaded_files:
        orig_name = os.path.basename(file.filename)
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            batch_images.append({
                "image_name": orig_name,
                "width": 0,
                "height": 0,
                "processing_seconds": 0.0,
                "annotated_url": "",
                "error": f"Could not decode image '{orig_name}'",
                "plates": []
            })
            continue

        h, w = img.shape[:2]
        t0 = time.time()

        # Thread-safe pipeline execution
        with PIPELINE_LOCK:
            mp_result = PROCESSOR.process_image_multi_plate(
                img,
                conf_threshold=0.25,
                ocr_threshold=0.85,
                require_validation=True,
                return_all_variants=True,
                return_crops=True,
                debug=False
            )

        proc_seconds = round(time.time() - t0, 3)

        # Draw annotated image with status-based colors
        annotated_img = img.copy()
        processed_plates = []

        status_colors = {
            "ACCEPTED": (46, 204, 113),      # Green (BGR)
            "NEEDS_REVIEW": (0, 165, 255),    # Amber
            "LOW_RESOLUTION": (0, 140, 255),  # Orange
            "OCR_FAILED": (50, 50, 220),      # Red
            "UNKNOWN": (150, 150, 150),
        }

        for idx, plate in enumerate(mp_result.get("plates", []), 1):
            total_plates += 1
            bbox = plate.get("bbox", {})
            det_conf = round(float(plate.get("detection_confidence", 0.0)), 4)
            ocr_data = plate.get("ocr", {})
            crop_img = plate.get("crop")

            status = ocr_data.get("status", "NEEDS_REVIEW")
            raw_reason = ocr_data.get("reason", "")
            orig_crop_w = bbox.get("width", 0)

            # Update summary counts
            if status == "ACCEPTED":
                accepted_count += 1
            elif status == "LOW_RESOLUTION":
                low_res_count += 1
            elif status == "OCR_FAILED":
                failed_count += 1
            else:
                needs_review_count += 1

            reason_str = format_plain_english_reason(status, raw_reason, ocr_data, orig_crop_w)
            plate_text = ocr_data.get("final_text") or ocr_data.get("text") or ""
            state_code = ocr_data.get("state_code")
            state_name = IndianStateCodeValidator.STATE_CODES.get(state_code) if state_code else None

            # Draw on annotated image
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            b_color = status_colors.get(status, (0, 165, 255))

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), b_color, 2)
            label_text = f"#{idx}: {plate_text or 'NO_TEXT'} [{status}]"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), b_color, -1)
            cv2.putText(annotated_img, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Save crop image
            crop_url = ""
            deskewed_url = ""
            if crop_img is not None and isinstance(crop_img, np.ndarray) and crop_img.size > 0:
                crop_name = f"{uuid.uuid4().hex[:10]}.jpg"
                crop_path = os.path.join(batch_dir, crop_name)
                cv2.imwrite(crop_path, crop_img)
                crop_url = f"/static/results/{batch_id}/{crop_name}"

                # Deskewed crop
                try:
                    deskewed_img = PlateOCR.deskew_plate(crop_img)
                except Exception:
                    deskewed_img = crop_img
                deskew_name = f"{uuid.uuid4().hex[:10]}.jpg"
                deskew_path = os.path.join(batch_dir, deskew_name)
                cv2.imwrite(deskew_path, deskewed_img)
                deskewed_url = f"/static/results/{batch_id}/{deskew_name}"

            # Prepare variant details
            variants_list = []
            used_variant_name = ocr_data.get("variant_used")
            for v in ocr_data.get("all_variants", []):
                v_name = v.get("variant")
                v_text = v.get("text", "")
                _, is_v = validate_plate(v_text)
                variants_list.append({
                    "name": v_name,
                    "text": v_text,
                    "valid": is_v,
                    "used": (v_name == used_variant_name),
                    "image_url": None
                })

            processed_plates.append({
                "index": idx,
                "text": plate_text,
                "status": status,
                "reason": reason_str,
                "state_code": state_code,
                "state_name": state_name,
                "yolo_conf": det_conf,
                "ocr_conf": round(float(ocr_data.get("confidence", 0.0)), 4),
                "bbox": bbox,
                "crop_url": crop_url,
                "deskewed_url": deskewed_url,
                "variants": variants_list
            })

        # Save annotated image with random name
        annotated_name = f"{uuid.uuid4().hex[:10]}.jpg"
        annotated_path = os.path.join(batch_dir, annotated_name)
        cv2.imwrite(annotated_path, annotated_img)
        annotated_url = f"/static/results/{batch_id}/{annotated_name}"

        batch_images.append({
            "image_name": orig_name,
            "width": w,
            "height": h,
            "processing_seconds": proc_seconds,
            "annotated_url": annotated_url,
            "plates": processed_plates
        })

    total_batch_time = round(time.time() - batch_start_time, 3)

    summary = {
        "total_images": len(uploaded_files),
        "total_plates": total_plates,
        "accepted": accepted_count,
        "needs_review": needs_review_count,
        "low_resolution": low_res_count,
        "failed": failed_count,
        "total_seconds": total_batch_time
    }

    response_data = {
        "batch_id": batch_id,
        "images": batch_images,
        "summary": summary
    }

    # Cache batch for CSV/JSON export
    BATCH_CACHE[batch_id] = response_data
    BATCH_ORDER.append(batch_id)
    if len(BATCH_ORDER) > MAX_CACHED_BATCHES:
        oldest = BATCH_ORDER.pop(0)
        BATCH_CACHE.pop(oldest, None)

    return jsonify(response_data)


@app.route('/api/export/<path:export_req>', methods=['GET'])
def api_export(export_req):
    """
    Export batch as CSV or JSON.
    Format 1: /api/export/<batch_id>.<csv|json>
    Format 2: /api/export/<csv|json>?batch_id=<batch_id>
    """
    batch_id = None
    fmt = None

    if '.' in export_req:
        parts = export_req.rsplit('.', 1)
        batch_id = parts[0]
        fmt = parts[1].lower()
    else:
        fmt = export_req.lower()
        batch_id = request.args.get('batch_id')

    # If no batch_id specified, use the most recent
    if not batch_id and BATCH_ORDER:
        batch_id = BATCH_ORDER[-1]

    if not batch_id or batch_id not in BATCH_CACHE:
        return jsonify({"error": f"Batch '{batch_id}' not found in server cache."}), 404

    batch_data = BATCH_CACHE[batch_id]

    if fmt == 'json':
        return Response(
            jsonify(batch_data).get_data(as_text=True),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename=anpr_batch_{batch_id}.json"}
        )

    elif fmt == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        # Header row
        writer.writerow([
            "image_name", "plate_index", "text", "status", "reason",
            "state_code", "state_name", "yolo_conf", "ocr_conf", "processing_seconds"
        ])

        for img in batch_data.get("images", []):
            img_name = img.get("image_name", "")
            proc_sec = img.get("processing_seconds", 0.0)
            plates = img.get("plates", [])
            if not plates:
                writer.writerow([
                    sanitize_csv_cell(img_name), 0, "", "NO_PLATES", "No plates detected",
                    "", "", 0.0, 0.0, proc_sec
                ])
            else:
                for p in plates:
                    writer.writerow([
                        sanitize_csv_cell(img_name),
                        p.get("index", 1),
                        sanitize_csv_cell(p.get("text", "")),
                        sanitize_csv_cell(p.get("status", "")),
                        sanitize_csv_cell(p.get("reason", "")),
                        sanitize_csv_cell(p.get("state_code", "")),
                        sanitize_csv_cell(p.get("state_name", "")),
                        p.get("yolo_conf", 0.0),
                        p.get("ocr_conf", 0.0),
                        proc_sec
                    ])

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=anpr_batch_{batch_id}.csv"}
        )

    return jsonify({"error": f"Unsupported export format '{fmt}'. Use 'csv' or 'json'."}), 400


# ── Keep Old /api/test Route Functional ────────────────────────────────────────

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

        safe_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            try:
                file_bytes = np.fromfile(filepath, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                img = None
        if img is None:
            return jsonify({"error": f"Could not read image: {file.filename}"}), 400

        h, w = img.shape[:2]

        with PIPELINE_LOCK:
            mp_result = PROCESSOR.process_image_multi_plate(
                img,
                conf_threshold=conf,
                ocr_threshold=0.85,
                require_validation=True,
                return_all_variants=True,
                return_crops=True,
                debug=False
            )

        detections = []
        for plate in mp_result.get("plates", []):
            pid   = plate["plate_id"]
            bbox  = plate["bbox"]
            det_conf = plate["detection_confidence"]
            ocr      = plate["ocr"]

            crop_img = plate.get("crop")
            cropped_base64 = ""
            if crop_img is not None and isinstance(crop_img, np.ndarray) and crop_img.size > 0:
                _, buf = cv2.imencode('.jpg', crop_img)
                cropped_base64 = base64.b64encode(buf).decode('utf-8')

            detection = {
                "id": pid,
                "confidence": det_conf,
                "x1": bbox["x1"], "y1": bbox["y1"], "x2": bbox["x2"], "y2": bbox["y2"],
                "w": bbox["width"], "h": bbox["height"],
                "cropped_image": f"data:image/jpeg;base64,{cropped_base64}",
                "ocr": {
                    "success":                ocr.get("success", False),
                    "raw_text":               ocr.get("raw_text", ""),
                    "cleaned_text":           ocr.get("cleaned_text", ""),
                    "final_text":             ocr.get("text", ""),
                    "text":                   ocr.get("text", ""),
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
                    "ocr_calls_made":         ocr.get("ocr_calls_made"),
                    "early_stopped":          ocr.get("early_stopped"),
                    "early_stop_reason":      ocr.get("early_stop_reason"),
                    "recovery_applied":       ocr.get("recovery_applied", False),
                    "recovery_variant":       ocr.get("recovery_variant", None),
                }
            }
            detections.append(detection)

        img_drawn = img.copy()
        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            det_conf = det["confidence"]
            ocr_info = det["ocr"]
            is_acc = ocr_info.get("status") == "ACCEPTED"
            box_color = (0, 255, 0) if is_acc else (0, 165, 255)

            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), box_color, 2)
            text = ocr_info.get("final_text") or ocr_info.get("raw_text") or "PLATE"
            status_str = ocr_info.get("status", "")
            o_conf = ocr_info.get("confidence", 0.0)

            display_text = f"{text} [{status_str}] ({det_conf:.2f} det, {o_conf*100:.1f}% ocr)"
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_drawn, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(img_drawn, display_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{safe_filename}")
        cv2.imwrite(out_path, img_drawn)

        _, buffer = cv2.imencode('.jpg', img_drawn)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        plate_count = mp_result.get("plate_count", 0)
        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0.0
        accepted_count = sum(1 for d in detections if d["ocr"].get("status") == "ACCEPTED")
        review_count = sum(1 for d in detections if d["ocr"].get("status") in ("NEEDS_REVIEW", "LOW_RESOLUTION"))
        recognized_count = sum(1 for d in detections if bool(d["ocr"].get("final_text") or d["ocr"].get("raw_text")))

        return jsonify({
            "success":          True,
            "detections":       plate_count,
            "plates":           detections,
            "avg_confidence":   round(float(avg_conf), 4),
            "accepted_count":   accepted_count,
            "review_count":     review_count,
            "recognized_count": recognized_count,
            "image_size":       f"{w}x{h}",
            "result_image":     f"data:image/jpeg;base64,{img_base64}",
            "processing_time_ms":  mp_result.get("processing_time_ms", 0),
            "detection_time_ms":   mp_result.get("detection_time_ms", 0),
            "ocr_time_ms":         mp_result.get("ocr_time_ms", 0),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n[OK] Starting Flask app at http://localhost:5000")
    print("[OK] Features: YOLO Plate Detection + PaddleOCR + MultiPlateProcessor")
    app.run(debug=False, host='127.0.0.1', port=5000)
