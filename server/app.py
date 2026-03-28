"""
PostureCoach — Flask API Server
Provides REST endpoints for posture analysis using the trained YOLOv8 pose model.
"""

import io
import os
import sys
import base64
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

# ── Resolve project paths ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best.pt"
FRONTEND_DIR = ROOT / "frontend"

# ── Flask app ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

# ── Lazy-load YOLOv8 model ──────────────────────────────────────────
_model = None

def get_model():
    """Load the YOLO model once and cache it."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            _model = YOLO(str(MODEL_PATH), task="pose")
            print(f"✓ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}", file=sys.stderr)
            raise
    return _model


# ── Serve Frontend ──────────────────────────────────────────────────
@app.route("/")
def serve_frontend():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(FRONTEND_DIR), path)


# ── API: Health Check ───────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    model_loaded = _model is not None
    model_exists = MODEL_PATH.exists()
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "model_exists": model_exists,
        "model_path": str(MODEL_PATH),
    })


# ── API: Model Info ─────────────────────────────────────────────────
@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "name": "PostureCoach YOLOv8n-pose",
        "architecture": "YOLOv8n-pose",
        "classes": ["Bad", "Good"],
        "input_size": 640,
        "epochs_trained": 25,
        "metrics": {
            "mAP50": 97.95,
            "mAP50_95_box": 68.88,
            "mAP50_95_pose": 97.85,
            "precision": 89.44,
            "recall": 96.19,
            "f1": 92.69,
        },
        "dataset": {
            "total_images": 655,
            "train": 573,
            "valid": 55,
            "test": 27,
            "source": "Roboflow (ikornproject/sitting-posture-rofqf)",
            "license": "CC BY 4.0",
        },
        "peak_performance": {
            "best_mAP50": 98.34,
            "best_precision": 97.36,
            "best_epoch": 17,
        }
    })


# ── API: Analyze Posture ────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts an image via:
      - multipart/form-data with field 'image'
      - JSON body with field 'image' containing a base64-encoded image

    Returns posture classification, confidence, keypoints, and annotated image.
    """
    t0 = time.time()

    try:
        # ── Parse input image ───────────────────────────────────────
        img = None
        if request.content_type and "multipart" in request.content_type:
            file = request.files.get("image")
            if file is None:
                return jsonify({"error": "No 'image' file in request"}), 400
            img = Image.open(file.stream).convert("RGB")
        else:
            data = request.get_json(silent=True)
            if data and "image" in data:
                b64 = data["image"]
                # Strip data URL prefix if present
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if img is None:
            return jsonify({"error": "No image provided"}), 400

        # ── Run inference ───────────────────────────────────────────
        model = get_model()
        img_np = np.array(img)
        results = model.predict(
            source=img_np,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

        # ── Extract results ─────────────────────────────────────────
        detections = []
        overall_posture = "unknown"
        overall_confidence = 0.0

        for r in results:
            boxes = r.boxes
            keypoints_data = r.keypoints

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].tolist()
                    cls_name = model.names[cls_id]

                    det = {
                        "class": cls_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 1) for v in xyxy],
                    }

                    # Extract keypoints if available
                    if keypoints_data is not None and i < len(keypoints_data):
                        kpts = keypoints_data[i]
                        if hasattr(kpts, 'data') and kpts.data is not None:
                            kpt_list = kpts.data[0].tolist()
                            det["keypoints"] = [
                                {"x": round(k[0], 1), "y": round(k[1], 1), "conf": round(k[2], 3)}
                                for k in kpt_list
                            ]

                    detections.append(det)

                    if conf > overall_confidence:
                        overall_confidence = conf
                        overall_posture = cls_name

            # ── Generate annotated image ────────────────────────────
            annotated = r.plot()
            annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR→RGB
            buf = io.BytesIO()
            annotated_pil.save(buf, format="JPEG", quality=85)
            annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed = round(time.time() - t0, 3)

        return jsonify({
            "posture": overall_posture,
            "confidence": round(overall_confidence, 4),
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "inference_time_ms": round(elapsed * 1000, 1),
            "image_size": {"width": img.width, "height": img.height},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Analyze Frame (optimized for live camera) ──────────────────
@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    """
    Lightweight endpoint for live camera frames.
    Accepts base64 image, returns only classification + confidence (no annotated image).
    """
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        model = get_model()
        results = model.predict(source=img_np, imgsz=640, conf=0.25, verbose=False)

        posture = "unknown"
        confidence = 0.0
        keypoints_list = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i].item())
                    conf = float(r.boxes.conf[i].item())
                    if conf > confidence:
                        confidence = conf
                        posture = model.names[cls_id]

                    if r.keypoints is not None and i < len(r.keypoints):
                        kpts = r.keypoints[i]
                        if hasattr(kpts, 'data') and kpts.data is not None:
                            kpt_list = kpts.data[0].tolist()
                            keypoints_list.append([
                                {"x": round(k[0], 1), "y": round(k[1], 1), "conf": round(k[2], 3)}
                                for k in kpt_list
                            ])

        return jsonify({
            "posture": posture,
            "confidence": round(confidence, 4),
            "keypoints": keypoints_list,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PostureCoach API Server")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Frontend: {FRONTEND_DIR}")
    print("=" * 60)

    # Pre-load model on startup
    try:
        get_model()
    except Exception as e:
        print(f"⚠ Model pre-load failed: {e}")
        print("  Server will still start — model loads on first request.")

    app.run(host="0.0.0.0", port=5000, debug=True)
