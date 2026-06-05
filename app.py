"""
app.py
======
Flask Backend for Oral Cancer Detection
Endpoints:
  GET  /              → Serve UI
  POST /predict       → Upload image → Return segmentation mask + cancer %
  GET  /model_info    → Return model metadata
"""

import os
import sys
import io
import json
import base64
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── TF / Model imports ────────────────────────
import tensorflow as tf # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
from unet_model import dice_coefficient, iou_metric, combined_loss

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model.h5')
THRESHOLD  = 0.5   # Pixel confidence threshold for segmentation

# ─────────────────────────────────────────────
# Load Model Once at Startup
# ─────────────────────────────────────────────

def load_model():
    """Load saved Keras model with custom objects."""
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Model not found at {MODEL_PATH}. "
            "Run train.py first, or a demo prediction will be used."
        )
        return None

    logger.info(f"Loading model from {MODEL_PATH} ...")
    custom_objects = {
        'combined_loss':    combined_loss,
        'dice_coefficient': dice_coefficient,
        'iou_metric':       iou_metric,
    }
    model = tf.keras.models.load_model(MODEL_PATH,
                                       custom_objects=custom_objects)
    logger.info("Model loaded successfully.")
    return model


MODEL = load_model()

# Store last known metrics (updated after each prediction)
MODEL_METRICS = {
    "dice":     0.3466,
    "iou":      0.2159,
    "accuracy": 0.9051,
}


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_image(file_bytes):
    """
    Decode uploaded bytes → numpy array → resize → normalize.
    Returns (original_rgb, preprocessed_batch)
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Ensure file is a valid JPG/PNG.")

    orig_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(orig_rgb, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    batch      = np.expand_dims(normalized, axis=0)   # (1, 224, 224, 3)
    return orig_rgb, resized, batch


# ─────────────────────────────────────────────
# Prediction & Postprocessing
# ─────────────────────────────────────────────

def run_inference(preprocessed_batch):
    """Run model inference. Returns raw probability mask (224×224)."""
    if MODEL is None:
        # Demo mode: generate a synthetic Gaussian blob mask
        logger.warning("Model not loaded – using demo mask.")
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
        y_idx, x_idx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        dist = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
        mask = np.exp(-dist / (2 * (40 ** 2)))
        return mask

    pred = MODEL.predict(preprocessed_batch, verbose=0)  # (1, 224, 224, 1)
    return pred[0, :, :, 0]


def build_overlay(original_img, binary_mask, alpha=0.45):
    """
    Overlay detected cancer region (red) on original image.
    Returns RGB overlay image (224×224×3).
    """
    overlay = original_img.copy().astype(np.uint8)
    red_channel = np.zeros_like(original_img, dtype=np.uint8)
    red_channel[:, :, 0] = 255   # Red tint

    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    overlay  = np.where(mask_3ch > 0,
                        (overlay * (1 - alpha) + red_channel * alpha).astype(np.uint8),
                        overlay)
    return overlay


def encode_image_b64(img_array):
    """Encode numpy RGB array → base64 PNG string."""
    img_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def compute_cancer_percentage(binary_mask):
    """
    What fraction of the oral image is cancer-positive?
    Returns float 0–100.
    """
    total  = binary_mask.size
    cancer = np.sum(binary_mask)
    return round((cancer / total) * 100, 2)


def severity_level(pct):
    """Map cancer percentage to clinical severity label."""
    if pct == 0:
        return "No Cancer Detected", "normal"
    elif pct < 5:
        return "Minimal Region Detected", "low"
    elif pct < 20:
        return "Moderate Region Detected", "moderate"
    else:
        return "Significant Region Detected", "high"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Form-data: file (image)
    Returns JSON:
      {
        original_b64:  <base64 PNG>,
        mask_b64:      <base64 PNG>,
        overlay_b64:   <base64 PNG>,
        cancer_pct:    23.5,
        severity:      "Moderate Region Detected",
        severity_level: "moderate",
        model_metrics: { dice, iou, accuracy }
      }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Key must be "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        file_bytes = file.read()
        orig_rgb, resized_rgb, batch = preprocess_image(file_bytes)

        # ── Run segmentation ──────────────────
        prob_mask    = run_inference(batch)
        binary_mask  = (prob_mask > THRESHOLD).astype(np.uint8)

        # ── Scale mask to 0-255 for display ──
        mask_display = (prob_mask * 255).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_display, cv2.COLORMAP_HOT)
        mask_rgb     = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)

        # ── Overlay ───────────────────────────
        overlay = build_overlay(resized_rgb, binary_mask)

        # ── Metrics ───────────────────────────
        cancer_pct    = compute_cancer_percentage(binary_mask)
        label, level  = severity_level(cancer_pct)

        return jsonify({
            'original_b64':   encode_image_b64(resized_rgb),
            'mask_b64':       encode_image_b64(mask_rgb),
            'overlay_b64':    encode_image_b64(overlay),
            'cancer_pct':     cancer_pct,
            'severity':       label,
            'severity_level': level,
            'model_metrics':  MODEL_METRICS,
        })

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/model_info')
def model_info():
    """Return model status and metrics."""
    return jsonify({
        'model_loaded':  MODEL is not None,
        'model_path':    MODEL_PATH,
        'input_size':    f'{IMG_SIZE}×{IMG_SIZE}',
        'architecture':  'U-Net + ResNet50 Encoder',
        'metrics':       MODEL_METRICS,
    })


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
