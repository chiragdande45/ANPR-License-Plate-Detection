"""
Flask Web App for ANPR Testing with PaddleOCR
"""
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from ocr_module import PlateOCR
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test', methods=['POST'])
def test():
    try:
        if MODEL is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        if OCR is None:
            return jsonify({"error": "OCR not initialized"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        conf = float(request.form.get('confidence', 0.25))
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400
        
        h, w = img.shape[:2]
        
        # Predict plates
        results = MODEL.predict(filepath, conf=conf, verbose=False)
        result = results[0]
        
        detections = []
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                # Handle tensor properly
                conf_val = box.conf.cpu().numpy()
                if isinstance(conf_val, np.ndarray):
                    conf_val = float(conf_val.flatten()[0])
                else:
                    conf_val = float(conf_val)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Run OCR on plate region, also get the cropped image
                ocr_result = OCR.process_plate(img, [x1, y1, x2, y2], return_crop=True)
                
                # Extract cropped image and convert to base64
                cropped_image = ocr_result.pop("cropped_image", None)
                cropped_base64 = ""
                if cropped_image is not None:
                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                
                detection = {
                    "id": i + 1,
                    "confidence": round(conf_val, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": x2 - x1, "h": y2 - y1,
                    "cropped_image": f"data:image/jpeg;base64,{cropped_base64}",
                    "ocr": {
                        "success": ocr_result.get("success", False),
                        "text": ocr_result.get("text", ""),
                        "confidence": ocr_result.get("confidence", 0.0),
                        "is_valid_format": ocr_result.get("is_valid_format", False)
                    }
                }
                
                detections.append(detection)
        
        # Draw and save
        img_drawn = result.plot()
        
        # Draw OCR text on detected plates
        for det in detections:
            if det["ocr"]["success"]:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                text = det["ocr"]["text"]
                conf = det["ocr"]["confidence"]
                
                # Draw text above box
                cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_text = f"{text} ({conf*100:.1f}%)"
                cv2.putText(img_drawn, display_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{file.filename}")
        cv2.imwrite(out_path, img_drawn)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img_drawn)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
        recognized_count = sum(1 for d in detections if d["ocr"]["success"])
        
        return jsonify({
            "success": True,
            "detections": len(result.boxes),
            "plates": detections,
            "avg_confidence": round(avg_conf, 4),
            "recognized_count": recognized_count,
            "image_size": f"{w}x{h}",
            "result_image": f"data:image/jpeg;base64,{img_base64}"
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n[OK] Starting Flask app at http://localhost:5000")
    print("[OK] Features: YOLO Plate Detection + PaddleOCR Recognition")
    app.run(debug=False, host='127.0.0.1', port=5000)
