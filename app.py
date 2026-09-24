"""
Simple Flask Web App for ANPR Testing
"""
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
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
print("Loading model...")
try:
    MODEL = YOLO("runs/detect/runs/detect/plate_detector_v2/weights/best.pt")
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    MODEL = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test', methods=['POST'])
def test():
    try:
        if MODEL is None:
            return jsonify({"error": "Model not loaded"}), 500
        
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
        
        # Predict
        results = MODEL.predict(filepath, conf=conf, verbose=False)
        result = results[0]
        
        detections = []
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                # Fix: Handle tensor properly
                conf_val = box.conf.cpu().numpy()
                if isinstance(conf_val, np.ndarray):
                    conf_val = float(conf_val.flatten()[0])
                else:
                    conf_val = float(conf_val)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                detections.append({
                    "id": i + 1,
                    "confidence": round(conf_val, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": x2 - x1, "h": y2 - y1
                })
        
        # Draw and save
        img_drawn = result.plot()
        out_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{file.filename}")
        cv2.imwrite(out_path, img_drawn)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img_drawn)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
        
        return jsonify({
            "success": True,
            "detections": len(result.boxes),
            "plates": detections,
            "avg_confidence": round(avg_conf, 4),
            "image_size": f"{w}x{h}",
            "result_image": f"data:image/jpeg;base64,{img_base64}"
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting app at http://localhost:5000")
    app.run(debug=False, host='127.0.0.1', port=5000)
