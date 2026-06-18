# ============================================================
# Modelos especializados (toxicidad / ofensividad)
# Uso: python -m src.specialized_models --model <nombre_corto>
#   nombres cortos: toxic-roberta, unbiased-toxic, snlp-toxicity, twitter-offensive
#   o --model all  para ejecutar todos
# ============================================================
import argparse
import time
import torch
import numpy as np
from transformers import pipeline
from tqdm import tqdm

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.data_loading import load_dataset, get_splits
from src.metrics import compute_metrics, print_metrics, full_report, save_results, save_predictions


# ---- Mapping de etiquetas de cada modelo a explicit (1) / clean (0) ----
# Cada función recibe la salida del pipeline y devuelve (pred, score)
LABEL_MAPPERS = {
    "toxic-roberta": lambda out: _map_binary_label(out, pos_label="LABEL_1"),
    "unbiased-toxic": lambda out: _map_max_toxic(out),
    "snlp-toxicity": lambda out: _map_binary_label(out, pos_label="toxic"),
    "twitter-offensive": lambda out: _map_binary_label(out, pos_label="offensive"),
}


def _map_binary_label(output, pos_label):
    """
    Para modelos que devuelven un ranking de etiquetas.
    Busca la etiqueta positiva y usa su score.
    """
    for item in output:
        if item["label"].lower() == pos_label.lower():
            score = item["score"]
            return (1 if score > 0.5 else 0), score
    # Si no se encuentra, usar la primera etiqueta
    score = output[0]["score"]
    return (1 if output[0]["label"].lower() == pos_label.lower() else 0), score


def _map_max_toxic(output):
    """
    Para unitary/unbiased-toxic-roberta que devuelve varias categorías de toxicidad.
    Si alguna categoría tóxica supera 0.5 -> explicit.
    """
    toxic_labels = ["toxicity", "severe_toxicity", "obscene", "sexual_explicit",
                    "identity_attack", "insult", "threat"]
    max_score = 0.0
    for item in output:
        if item["label"].lower() in toxic_labels:
            max_score = max(max_score, item["score"])
    return (1 if max_score > 0.5 else 0), max_score


def run_specialized_model(model_key, df, config_name, cfg):
    """Ejecuta un modelo especializado sobre un DataFrame."""
    model_info = cfg["specialized_models"][model_key]
    model_name = model_info["name"]
    device = get_device()
    mapper = LABEL_MAPPERS[model_key]

    print(f"\nCargando modelo: {model_name}")
    # top_k=None para obtener todos los scores
    pipe = pipeline(
        "text-classification",
        model=model_name,
        device=device,
        top_k=None,
        truncation=True,
        max_length=512,
    )

    texts = df[cfg["text_column"]].tolist()
    labels = df["label"].tolist()
    preds, scores = [], []

    t0 = time.time()
    # Inferencia por lotes
    for out in tqdm(pipe(texts, batch_size=32), total=len(texts), desc=model_key):
        pred, score = mapper(out)
        preds.append(pred)
        scores.append(score)
    elapsed = time.time() - t0

    metrics = compute_metrics(labels, preds)
    print_metrics(metrics, name=f"{model_key} ({config_name})")
    print(full_report(labels, preds))

    save_results(metrics, model_key, config_name, cfg, len(labels), elapsed)
    save_predictions(df, preds, scores, name=model_key, config_name=config_name, cfg=cfg)

    # Liberar memoria GPU
    del pipe
    torch.cuda.empty_cache()
    return metrics


# ---- Ejecución directa ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelos especializados de toxicidad")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Nombre corto: toxic-roberta | unbiased-toxic | snlp-toxicity | twitter-offensive | all"
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    df = load_dataset(cfg)

    # Determinar qué modelos ejecutar
    if args.model == "all":
        keys = list(cfg["specialized_models"].keys())
    else:
        keys = [args.model]

    for key in keys:
        assert key in cfg["specialized_models"], f"Modelo '{key}' no encontrado en la config"

        # Configuración A: dataset completo
        print(f"\n>>> {key} - Configuración A: dataset completo")
        run_specialized_model(key, df, "full_dataset", cfg)

        # Configuración B: solo test
        print(f"\n>>> {key} - Configuración B: split test")
        _, _, test_df = get_splits(df, cfg)
        run_specialized_model(key, test_df, "split_test", cfg)
