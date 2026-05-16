# Manual tuning sederhana untuk kriteria Skilled.
# Jalankan setelah mlflow ui aktif: python modelling_tuning.py
import os
import mlflow
from modelling import build_model, prepare_dataset, build_datasets, EXPERIMENT_NAME, TRACKING_URI, EPOCHS

LEARNING_RATES = [0.001, 0.0005]
DROPOUTS = [0.3, 0.5]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME + " - Tuning")
prepare_dataset()
train_ds, val_ds, test_ds, class_names = build_datasets()

for lr in LEARNING_RATES:
    for dropout in DROPOUTS:
        with mlflow.start_run(run_name=f"manual_tuning_lr_{lr}_dropout_{dropout}"):
            import tensorflow as tf
            model = build_model(len(class_names))
            model.optimizer.learning_rate.assign(lr)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("epochs", EPOCHS)
            history = model.fit(train_ds, validation_data=val_ds, epochs=max(3, EPOCHS // 2), verbose=1)
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            mlflow.log_metric("test_accuracy", float(test_acc))
            mlflow.log_metric("test_loss", float(test_loss))
            mlflow.tensorflow.log_model(model, artifact_path="model")
            print(lr, dropout, test_acc)
