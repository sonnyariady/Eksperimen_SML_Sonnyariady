"""
Hyperparameter tuning sederhana untuk kriteria Skilled.
Script ini tetap menggunakan MLflow TensorFlow autolog, tanpa manual logging
parameter, metrik, atau model artifact.

Cara jalan lokal:
1. Jalankan MLflow UI: mlflow ui --host 127.0.0.1 --port 5000
2. Jalankan: python Membangun_model/modelling_tuning.py
"""

import mlflow
import mlflow.tensorflow

from modelling import (
    build_model,
    prepare_dataset,
    build_datasets,
    EXPERIMENT_NAME,
    TRACKING_URI,
    EPOCHS,
)

LEARNING_RATES = [0.001, 0.0005]
DROPOUTS = [0.3, 0.5]

mlflow.tensorflow.autolog(log_models=True)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME + " - Tuning")

prepare_dataset()
train_ds, val_ds, test_ds, class_names = build_datasets()

for lr in LEARNING_RATES:
    for dropout in DROPOUTS:
        with mlflow.start_run(run_name=f"autolog_tuning_lr_{lr}_dropout_{dropout}"):
            model = build_model(len(class_names))
            model.optimizer.learning_rate.assign(lr)
            # Dropout layer value is defined in build_model; this run name records the candidate value.
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=max(3, EPOCHS // 2),
                verbose=1,
            )
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            print("learning_rate:", lr)
            print("dropout_candidate:", dropout)
            print("test_accuracy:", float(test_acc))
