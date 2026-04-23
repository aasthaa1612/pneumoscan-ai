"""
app.py — Flask API server (auto-detects model architecture).
Run from project root: python backend/app.py
"""

import base64, io, json, sys, os
import urllib.request, urllib.error
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.cnn_model import get_model_for_checkpoint
from backend.gradcam.gradcam  import GradCAM
from backend.utils.dataset    import get_transforms

MODEL_PATH  = ROOT / "outputs" / "models" / "best_model.pt"
CALIB_PATH  = ROOT / "backend" / "calibration.json"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ── Calibration ───────────────────────────────────────────────────────────
TEMPERATURE, THRESHOLD = 1.0, 0.5
if CALIB_PATH.exists():
    c = json.loads(CALIB_PATH.read_text())
    TEMPERATURE = c.get("temperature", 1.0)
    THRESHOLD   = c.get("threshold",   0.5)
    print(f"✅ Calibration: T={TEMPERATURE:.4f}  threshold={THRESHOLD:.4f}")
else:
    print("⚠️  No calibration.json — run python backend/fix_threshold.py")

# ── Model (auto-detects variant A or B) ───────────────────────────────────
if not MODEL_PATH.exists():
    print(f"❌ No model at {MODEL_PATH}"); sys.exit(1)

model, ckpt, variant = get_model_for_checkpoint(str(MODEL_PATH), device=str(DEVICE))
grad_cam  = GradCAM(model, target_layer=model.layer4)
transform = get_transforms("test")
print(f"✅ Model variant={variant}  epoch={ckpt.get('epoch','?')}  val_auc={ckpt.get('val_auc',0):.4f}")
print(f"🚀 http://localhost:5000\n")

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO(); img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/health")
def health():
    return jsonify({"status":"ok","device":str(DEVICE)})

@app.route("/model-info")
def model_info():
    return jsonify({
        "architecture"  : f"ResNet-50 variant-{variant}",
        "classes"       : CLASS_NAMES,
        "device"        : str(DEVICE),
        "val_auc"       : round(ckpt.get("val_auc",0), 4),
        "trained_epochs": ckpt.get("epoch","?"),
        "calibration"   : {"temperature":TEMPERATURE,"threshold":THRESHOLD,
                           "calibrated":CALIB_PATH.exists()},
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"No image. Use form key 'image'."}), 400
    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    orig_w, orig_h = pil_img.size
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor) / TEMPERATURE, dim=1)[0]

    pneu_prob  = probs[1].item()
    pred_idx   = 1 if pneu_prob >= THRESHOLD else 0
    confidence = probs[pred_idx].item()

    cam_t     = tensor.clone().requires_grad_(True)
    cam,_,_   = grad_cam(cam_t, class_idx=pred_idx)
    orig_224  = pil_img.resize((224,224))
    overlay   = GradCAM.pil_to_overlay(cam, orig_224, alpha=0.45)

    return jsonify({
        "prediction"    : CLASS_NAMES[pred_idx],
        "confidence"    : round(confidence*100, 2),
        "probabilities" : {"NORMAL":round(probs[0].item()*100,2),
                           "PNEUMONIA":round(probs[1].item()*100,2)},
        "calibration"   : {"temperature":TEMPERATURE,"threshold":THRESHOLD,
                           "raw_pneu_prob":round(pneu_prob,4)},
        "original_image" : pil_to_b64(orig_224),
        "gradcam_overlay": pil_to_b64(overlay),
        "image_size"     : {"width":orig_w,"height":orig_h},
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    api_key = os.environ.get("ANTHROPIC_API_KEY", "sk-hQP9bcnrn4FgW1DOMCkJTg")
    if not api_key:
        return jsonify({"content": [{"text": "I am currently in demo mode as no Anthropic API key is configured on the backend. Please set the ANTHROPIC_API_KEY environment variable to chat with me!"}]}), 200
        
    try:
        req_data = json.dumps({
            "model": data.get("model", "claude-3-5-sonnet-20241022"),
            "max_tokens": data.get("max_tokens", 1000),
            "system": data.get("system", ""),
            "messages": data.get("messages", [])
        }).encode("utf-8")
        
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=req_data,
            headers={
                "x-api-key": "sk-hQP9bcnrn4FgW1DOMCkJTg",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return jsonify(res_data), response.status
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        return jsonify({"error": f"API Error: {error_msg}"}), e.code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)