# 🫁 PneumoScan AI

> **AI-powered pneumonia detection from chest X-rays** using ResNet-50, Grad-CAM visualization, and an integrated medical AI chatbot.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- 🔬 **ResNet-50 deep learning** model trained on 5,216 chest X-ray images
- 🎯 **Val AUC 0.9941** with temperature scaling + threshold calibration
- 🌡️ **Grad-CAM heatmaps** — visually highlights regions the model focused on
- 🖼️ **Compare Slider** — drag to compare original vs. Grad-CAM overlay
- 🤖 **AI Chatbot** — powered by Groq (Llama 3.1) for medical Q&A
- ⚡ **Real-time inference** on CPU or GPU
- 🎨 **Premium animated UI** — particle canvas, 360° animations, glassmorphism

---

## 🗂️ Project Structure

```
project/
├── backend/
│   ├── app.py                  # Flask API server (predict, chat, model-info)
│   ├── train.py                # Model training script
│   ├── retrain_balanced.py     # Balanced retraining with class weights
│   ├── evaluate.py             # Evaluation — accuracy, F1, classification report
│   ├── diagnose.py             # Diagnosis & debugging utilities
│   ├── fix_threshold.py        # Temperature scaling + threshold calibration
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── calibration.json        # Auto-generated calibration (T, threshold, AUC)
│   ├── class_weights.json      # Class imbalance weights
│   ├── gradcam/
│   │   └── gradcam.py          # Grad-CAM implementation
│   ├── models/
│   │   └── cnn_model.py        # ResNet-50 model definition (variant A & B)
│   └── utils/
│       └── dataset.py          # Dataset loader & transforms
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React application
│   │   └── App.css             # Styles + animations
│   └── package.json
├── outputs/
│   └── models/
│       └── best_model.pt       # Trained model checkpoint
└── data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    └── test/
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- A GPU is optional — CPU inference works fine

### 1. Clone & set up the virtual environment

```bash
git clone https://github.com/your-username/pneumoscan-ai.git
cd pneumoscan-ai

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the backend

```bash
python backend/app.py
# Server starts at http://localhost:5000
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm start
# Opens at http://localhost:3000
```

---

## 🤖 AI Chatbot Setup

The chatbot uses **Groq** (free, no credit card required).

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Set it before starting the backend:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your_key_here"
python backend/app.py
```

The chatbot uses **Llama 3.1 8B Instant** via Groq's OpenAI-compatible API.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet-50 (fine-tuned, Variant B) |
| Dataset | Chest X-Ray Images (Pneumonia) — Kaggle |
| Training images | 5,216 |
| Val AUC | **0.9941** |
| Temperature (T) | 1.2553 |
| Decision threshold | 0.4575 |
| Classes | `NORMAL`, `PNEUMONIA` |
| Device | Auto (CUDA or CPU) |

### Calibration

After training, run temperature scaling to generate `calibration.json`:

```bash
python backend/fix_threshold.py
```

This sets the optimal decision threshold and temperature to reduce overconfident predictions.

---

## 🔌 API Reference

### `GET /health`
Returns server status and device info.

### `GET /model-info`
Returns model architecture, Val AUC, calibration settings.

### `POST /predict`
Upload a chest X-ray for inference.

- **Form field:** `image` (JPEG/PNG, max 10 MB)
- **Returns:** `prediction`, `confidence`, `probabilities`, Grad-CAM overlay (base64)

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@chest_xray.jpg"
```

### `POST /chat`
Proxy to Groq LLM for medical Q&A.

```json
{
  "messages": [{"role": "user", "content": "What is pneumonia?"}],
  "system": "You are a radiology assistant...",
  "max_tokens": 1000
}
```

---

## 🏋️ Training Your Own Model

```bash
# Initial training
python backend/train.py

# Balanced retraining (fixes class imbalance bias)
python backend/retrain_balanced.py

# Calibrate threshold after training
python backend/fix_threshold.py

# Evaluate on test set
python backend/evaluate.py
```

---

## 🛠️ Tech Stack

### Backend
| Library | Purpose |
|---|---|
| Flask | REST API server |
| PyTorch | Model inference |
| torchvision | ResNet-50 backbone |
| Pillow | Image processing |
| flask-cors | CORS for frontend |

### Frontend
| Library | Purpose |
|---|---|
| React 18 | UI framework |
| Framer Motion | Animations |
| Recharts | Radial confidence gauges |
| React Dropzone | Drag & drop upload |
| Lucide React | Icons |
| Axios | API calls |

---

## ⚠️ Disclaimer

> This tool is intended for **research and educational purposes only**.
> It is **not a substitute for professional clinical diagnosis**.
> Always consult a licensed medical professional for health decisions.

---

## 📄 License

MIT © 2026 PneumoScan AI
