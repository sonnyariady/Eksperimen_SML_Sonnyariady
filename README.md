# SMSML Sonnyariady - Indonesian Food Classification

Dataset: https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification

Project ini adalah submission MLOps untuk klasifikasi makanan Indonesia menggunakan TensorFlow, MLflow autolog, GitHub Actions, FastAPI, Prometheus, dan Grafana.

## Struktur Penting

```text
SMSML_Sonnyariady/
├── Eksperimen_SML_Sonnyariady.txt
├── Workflow-CI.txt
├── preprocessing/
│   ├── Eksperimen_Sonnyariady.ipynb
│   ├── automate_Sonnyariady.py
│   └── namadataset_preprocessing/
├── Membangun_model/
│   ├── Eksperimen_Sonnyariady.ipynb
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   └── screenshot_artifak.png
├── Workflow-CI/
│   ├── .github/workflows/mlflow-ci.yml
│   ├── MLProject/
│   │   ├── MLproject
│   │   ├── conda.yaml
│   │   ├── modelling.py
│   │   └── requirements.txt
│   └── screenshot_workflow_success.png
└── Monitoring dan Logging/
    ├── inference.py
    ├── docker-compose.yml
    ├── prometheus.yml
    ├── prometheus_exporter.py
    ├── 1.bukti_serving/
    ├── 4.bukti monitoring Prometheus/
    ├── 5.bukti monitoring Grafana/
    └── 6.bukti alerting Grafana/
```

## Kriteria 1 - Eksperimen Dataset

Notebook eksperimen berada di:

```text
preprocessing/Eksperimen_Sonnyariady.ipynb
```

Notebook tersebut sudah berisi data loading, EDA, distribusi kelas, visualisasi sampel gambar, dan tahapan preprocessing.

## Kriteria 2 - Model Machine Learning

Training model berada di:

```text
Membangun_model/modelling.py
```

File ini menggunakan:

```python
mlflow.tensorflow.autolog(log_models=True)
```

Autolog digunakan untuk mencatat parameter, metrik, dan artefak model otomatis. Tidak menggunakan manual MLflow logging APIs untuk parameter, metrik, atau model artifact.

## Kriteria 3 - Workflow CI

Repository Workflow CI:

```text
https://github.com/sonnyariady/Workflow-CI-SMSML-Sonnyariady/actions
```

Workflow menjalankan MLflow Project dari folder `MLProject` dan bukti sukses tersedia di:

```text
Workflow-CI/screenshot_workflow_success.png
```

## Kriteria 4 - Monitoring dan Logging

Folder `Monitoring dan Logging` berisi bukti serving, konfigurasi Prometheus, dashboard Grafana, serta alerting Grafana.
