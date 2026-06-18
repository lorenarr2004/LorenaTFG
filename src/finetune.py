# ============================================================
# Fine-tuning supervisado: BERT / RoBERTa / DeBERTa
# Uso: python -m src.finetune --model <nombre_corto>
#   nombres cortos: bert | roberta | deberta
# ============================================================
import argparse
import os
import time
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.data_loading import load_dataset, get_splits
from src.metrics import compute_metrics, print_metrics, full_report, save_results, save_predictions


def tokenize_dataset(df, tokenizer, text_col, max_length):
    """Convierte un DataFrame a Dataset de HuggingFace tokenizado."""
    dataset = Dataset.from_dict({
        "text": df[text_col].tolist(),
        "label": df["label"].tolist(),
    })

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return dataset


def compute_trainer_metrics(eval_pred):
    """Función de métricas compatible con el Trainer de HuggingFace."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, pos_label=1, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_finetune(model_key, cfg):
    """Entrena y evalúa un modelo de clasificación."""
    model_info = cfg["finetune_models"][model_key]
    model_name = model_info["name"]
    params = cfg["finetune_params"]
    text_col = cfg["text_column"]

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {model_key} ({model_name})")
    print(f"{'='*60}")

    # Cargar datos y hacer split
    df = load_dataset(cfg)
    train_df, val_df, test_df = get_splits(df, cfg)

    # Tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "clean", 1: "explicit"},
        label2id={"clean": 0, "explicit": 1},
    )

    # Tokenizar los splits
    train_ds = tokenize_dataset(train_df, tokenizer, text_col, params["max_length"])
    val_ds = tokenize_dataset(val_df, tokenizer, text_col, params["max_length"])
    test_ds = tokenize_dataset(test_df, tokenizer, text_col, params["max_length"])

    # Directorio de salida para checkpoints
    output_model_dir = os.path.join(cfg["output_dir"], f"finetune_{model_key}")

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=params["num_epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"] * 2,
        learning_rate=params["learning_rate"],
        warmup_ratio=params.get("warmup_ratio", 0.1),
        weight_decay=params["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        report_to="none",  # sin wandb ni tensorboard
    )

    # Callbacks
    callbacks = []
    if params.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=params["early_stopping_patience"])
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_trainer_metrics,
        callbacks=callbacks,
    )

    # Entrenar
    t0 = time.time()
    trainer.train()
    elapsed_train = time.time() - t0
    print(f"\nEntrenamiento completado en {elapsed_train:.1f}s")

    # Evaluar en test
    t0 = time.time()
    test_output = trainer.predict(test_ds)
    elapsed_test = time.time() - t0

    preds = np.argmax(test_output.predictions, axis=-1).tolist()
    scores = torch.softmax(torch.tensor(test_output.predictions), dim=-1)[:, 1].tolist()
    labels = test_df["label"].tolist()

    # Métricas
    metrics = compute_metrics(labels, preds)
    print_metrics(metrics, name=f"{model_key} fine-tuned (split_test)")
    print(full_report(labels, preds))

    save_results(metrics, f"{model_key}_finetuned", "split_test", cfg, len(labels), elapsed_test)
    save_predictions(test_df, preds, scores, name=f"{model_key}_finetuned", config_name="split_test", cfg=cfg)

    # Guardar el mejor modelo
    best_dir = os.path.join(output_model_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Mejor modelo guardado en {best_dir}")

    # Liberar memoria
    del model, trainer
    torch.cuda.empty_cache()
    return metrics


# ---- Ejecución directa ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning BERT / RoBERTa / DeBERTa")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Nombre corto: bert | roberta | deberta"
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    if args.model == "all":
        keys = list(cfg["finetune_models"].keys())
    else:
        keys = [args.model]

    for key in keys:
        assert key in cfg["finetune_models"], f"Modelo '{key}' no encontrado en la config"
        run_finetune(key, cfg)
