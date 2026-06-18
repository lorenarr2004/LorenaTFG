# ============================================================
# Métricas comunes para todos los experimentos
# ============================================================
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    """
    Calcula las métricas principales.
    Devuelve un dict con accuracy, precision, recall, F1 (clase explicit)
    y la matriz de confusión.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_explicit": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_explicit": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_explicit": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(metrics, name=""):
    """Muestra las métricas por consola de forma legible."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Precision (explicit): {metrics['precision_explicit']:.4f}")
    print(f"  Recall (explicit):    {metrics['recall_explicit']:.4f}")
    print(f"  F1 (explicit):        {metrics['f1_explicit']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print()


def full_report(y_true, y_pred):
    """Devuelve el classification_report completo como string."""
    return classification_report(
        y_true, y_pred, target_names=["clean", "explicit"], zero_division=0
    )


def save_results(metrics, name, config_name, cfg, n_examples, elapsed=None):
    """
    Guarda las métricas de un experimento en JSON.
    - name: identificador del modelo/método
    - config_name: 'full_dataset' o 'split_test'
    """
    result = {
        "method": name,
        "config": config_name,
        "n_examples": n_examples,
        "elapsed_seconds": elapsed,
        **metrics,
    }
    path = os.path.join(cfg["metrics_dir"], f"{name}_{config_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Métricas guardadas en {path}")


def save_predictions(df, preds, scores, name, config_name, cfg):
    """
    Guarda las predicciones por instancia en CSV.
    - preds: lista de predicciones (0/1)
    - scores: lista de scores/probabilidades (puede ser None)
    """
    out = pd.DataFrame({
        "index": range(len(preds)),
        "label": df["label"].values[:len(preds)],
        "prediction": preds,
    })
    if scores is not None:
        out["score"] = scores
    # Añadir referencia al texto (primeros 80 chars)
    text_col = cfg["text_column"]
    out["text_preview"] = [str(t)[:80] for t in df[text_col].values[:len(preds)]]

    path = os.path.join(cfg["predictions_dir"], f"{name}_{config_name}.csv")
    out.to_csv(path, index=False)
    print(f"Predicciones guardadas en {path}")
