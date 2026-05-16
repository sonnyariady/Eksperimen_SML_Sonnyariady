import json
import time
from pathlib import Path
import io

import numpy as np
import psutil
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]

# Cek beberapa kemungkinan lokasi model karena hasil training bisa tersimpan
# di root project atau di folder Membangun_model tergantung dari mana modelling.py dijalankan.
MODEL_CANDIDATES = [
    BASE_DIR / "Membangun_model" / "model_artifacts" / "keras_model",
    BASE_DIR / "model_artifacts" / "keras_model",
    Path.cwd().parent / "Membangun_model" / "model_artifacts" / "keras_model",
    Path.cwd().parent / "model_artifacts" / "keras_model",
]

LABEL_CANDIDATES = [
    BASE_DIR / "Membangun_model" / "labels.json",
    BASE_DIR / "labels.json",
    Path.cwd().parent / "Membangun_model" / "labels.json",
    Path.cwd().parent / "labels.json",
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if (p / "saved_model.pb").exists()), MODEL_CANDIDATES[0])
LABEL_PATH = next((p for p in LABEL_CANDIDATES if p.exists()), LABEL_CANDIDATES[0])

IMG_SIZE = (128, 128)

app = FastAPI(title="Indonesian Food Classification API")

model = None
infer = None
labels = []

REQUEST_COUNT = Counter("food_api_request_total", "Total API requests")
PREDICTION_COUNT = Counter("food_prediction_total", "Total predictions")
ERROR_COUNT = Counter("food_error_total", "Total prediction errors")
LATENCY = Histogram("food_prediction_latency_seconds", "Prediction latency in seconds")
CPU_USAGE = Gauge("food_cpu_usage_percent", "CPU usage percent")
MEMORY_USAGE = Gauge("food_memory_usage_percent", "Memory usage percent")


@app.on_event("startup")
def load_model():
    global model, infer, labels

    if not MODEL_PATH.exists() or not (MODEL_PATH / "saved_model.pb").exists():
        checked = "\n".join(str(p) for p in MODEL_CANDIDATES)
        raise FileNotFoundError(
            "Model SavedModel tidak ditemukan. Lokasi yang dicek:\n" + checked
        )

    model = tf.saved_model.load(str(MODEL_PATH))

    signatures = list(model.signatures.keys())
    if "serving_default" not in model.signatures:
        raise RuntimeError(f"Signature 'serving_default' tidak ditemukan. Signature tersedia: {signatures}")

    infer = model.signatures["serving_default"]

    if LABEL_PATH.exists():
        labels = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    else:
        labels = []

    print("Model loaded from:", MODEL_PATH)
    print("Labels loaded from:", LABEL_PATH)
    print("Available signatures:", signatures)


@app.get("/")
def home():
    REQUEST_COUNT.inc()
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "label_path": str(LABEL_PATH),
        "labels": labels,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    start = time.time()

    try:
        content = await file.read()

        img = Image.open(io.BytesIO(content)).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img), axis=0).astype(np.float32)

        output = infer(tf.constant(arr))
        pred = list(output.values())[0].numpy()[0]

        idx = int(np.argmax(pred))
        confidence = float(pred[idx])

        PREDICTION_COUNT.inc()
        LATENCY.observe(time.time() - start)
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        return {
            "label": labels[idx] if idx < len(labels) else str(idx),
            "confidence": confidence,
            "class_index": idx,
        }

    except Exception as exc:
        ERROR_COUNT.inc()
        raise exc


@app.get("/metrics")
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
