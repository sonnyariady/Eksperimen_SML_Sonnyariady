# SMSML Sonnyariady - Indonesian Food Classification

Dataset:
https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification

GitHub:
https://github.com/sonnyariady

---

# Deskripsi Project

Project ini merupakan implementasi end-to-end MLOps untuk klasifikasi makanan Indonesia menggunakan TensorFlow, MLflow, FastAPI, Prometheus, dan Grafana.

Pipeline project mencakup:

- Dataset preprocessing
- Model training TensorFlow CNN
- Experiment tracking menggunakan MLflow
- Model registry
- FastAPI serving
- Monitoring metrics menggunakan Prometheus
- Visualization dashboard menggunakan Grafana
- Docker Compose orchestration

---

# Struktur Project

```text
SMSML_Sonnyariady/
│
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   └── requirements.txt
│
├── model_artifacts/
│   └── keras_model/
│
├── Monitoring dan Logging/
│   ├── inference.py
│   ├── prometheus.yml
│   ├── docker-compose.yml
│   ├── prometheus_exporter.py
│   ├── 4.bukti monitoring Prometheus/
│   ├── 5.bukti monitoring Grafana/
│   └── 6.bukti alerting Grafana/
│
├── mlruns/
├── mlartifacts/
├── labels.json
├── README.md
└── .gitignore

# Requirements

- Python 3.10+
- Docker Desktop
- TensorFlow
- FastAPI
- MLflow
- Prometheus
- Grafana

# Install Dependency

```bash
pip install -r requirements.txt

```md
# Contoh Request Inference

Endpoint:

http://127.0.0.1:8000/predict

Method:

POST

Contoh menggunakan curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
-F "file=@contoh.jpg"