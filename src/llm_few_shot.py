# ============================================================
# LLMs generales — experimentación k-shot
# Evalúa el impacto del número de ejemplos (k) en el prompt.
# Uso: python -m src.llm_few_shot --model <nombre_corto> [--k <valor>]
#   nombres cortos: qwen | llama | phi | mistral | all
#   --k: valor específico de k (por defecto ejecuta todos los de la config)
# ============================================================
import argparse
import os
import time
import torch
import pandas as pd
from tqdm import tqdm

from src.utils import load_config, set_seed, ensure_dirs
from src.data_loading import load_dataset, get_splits
from src.prompts import build_few_shot_prompt
from src.llm_zero_shot import parse_response, load_llm
from src.metrics import compute_metrics, print_metrics, full_report, save_results, save_predictions


def select_few_shot_examples(train_df, n_shots, cfg):
    """
    Selecciona n_shots ejemplos del train set de forma reproducible.
    Siempre balanceado: n_shots/2 explicit + n_shots/2 clean.
    La selección para k=4 coincide con los primeros 2+2 ejemplos,
    manteniendo compatibilidad con resultados previos.
    """
    n_per_class = n_shots // 2
    text_col = cfg["text_column"]
    seed = cfg["seed"]

    # Ordenar por índice para reproducibilidad con semilla fija
    explicit_df = train_df[train_df["label"] == 1].sample(
        n=n_per_class, random_state=seed
    )
    clean_df = train_df[train_df["label"] == 0].sample(
        n=n_per_class, random_state=seed
    )

    # Intercalar: primero clean, luego explicit (como en la versión original)
    examples = []
    for _, row in clean_df.iterrows():
        examples.append({"text": row[text_col], "label_str": "clean"})
    for _, row in explicit_df.iterrows():
        examples.append({"text": row[text_col], "label_str": "explicit"})

    return examples


def classify_few_shot(texts, examples, tokenizer, model, cfg):
    """
    Clasifica textos usando prompt few-shot con k ejemplos.
    Devuelve listas de predicciones y respuestas crudas.
    """
    max_len = cfg.get("llm_max_length", 2048)
    max_new = cfg.get("llm_max_new_tokens", 10)
    preds, raw_responses = [], []

    for text in tqdm(texts, desc=f"K-shot (k={len(examples)})"):
        prompt = build_few_shot_prompt(text[:3000], examples)

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=max_len
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw_responses.append(response)
        preds.append(parse_response(response))

    return preds, raw_responses


def run_llm_kshot(model_key, test_df, train_df, n_shots, cfg):
    """Ejecuta un LLM en k-shot sobre el test set (Configuración B)."""
    model_info = cfg["llm_models"][model_key]
    model_name = model_info["name"]

    tokenizer, model = load_llm(model_name, None)
    examples = select_few_shot_examples(train_df, n_shots, cfg)

    print(f"\n  k={n_shots} — {len(examples)} ejemplos seleccionados")
    for ex in examples:
        print(f"    - {ex['label_str']}: {ex['text'][:60]}...")

    texts = test_df[cfg["text_column"]].tolist()
    labels = test_df["label"].tolist()

    t0 = time.time()
    preds, raw = classify_few_shot(texts, examples, tokenizer, model, cfg)
    elapsed = time.time() - t0

    metrics = compute_metrics(labels, preds)
    print_metrics(metrics, name=f"{model_key} k-shot k={n_shots} (split_test)")
    print(full_report(labels, preds))

    # Guardar resultados con nombre que incluye k
    exp_name = f"{model_key}_kshot_k{n_shots}"
    save_results(metrics, exp_name, "split_test", cfg, len(labels), elapsed)
    save_predictions(test_df, preds, scores=None, name=exp_name, config_name="split_test", cfg=cfg)

    del model, tokenizer
    torch.cuda.empty_cache()

    return metrics


def run_kshot_experiment(cfg):
    """
    Ejecuta el experimento k-shot completo: para cada modelo y cada valor de k,
    evalúa en el test set y genera las tablas resumen.
    """
    df = load_dataset(cfg)
    train_df, val_df, test_df = get_splits(df, cfg)

    k_values = cfg["k_shot_values"]
    model_keys = list(cfg["llm_models"].keys())

    results = []

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"  K-SHOT EXPERIMENT: {model_key}")
        print(f"{'='*60}")

        for k in k_values:
            print(f"\n--- {model_key} con k={k} ---")
            metrics = run_llm_kshot(model_key, test_df, train_df, k, cfg)
            results.append({
                "model": model_key,
                "k": k,
                "accuracy": metrics["accuracy"],
                "precision_explicit": metrics["precision_explicit"],
                "recall_explicit": metrics["recall_explicit"],
                "f1_explicit": metrics["f1_explicit"],
            })

    # Generar CSV resumen
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(cfg["output_dir"], "kshot_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCSV resumen k-shot guardado en {summary_path}")

    # Generar tabla Markdown comparativa
    generate_kshot_markdown_table(summary_df, cfg)

    return summary_df


def generate_kshot_markdown_table(summary_df, cfg):
    """Genera una tabla Markdown con la evolución del F1 explícito por modelo y k."""
    pivot = summary_df.pivot(index="k", columns="model", values="f1_explicit")
    pivot = pivot.reset_index()

    lines = []
    lines.append("# Resultados K-shot: F1 (clase explícita) por modelo y k\n")
    lines.append("| k | " + " | ".join(pivot.columns[1:]) + " |")
    lines.append("|---" + "|---" * len(pivot.columns[1:]) + "|")

    for _, row in pivot.iterrows():
        vals = [f"{row[col]:.4f}" for col in pivot.columns[1:]]
        lines.append(f"| {int(row['k'])} | " + " | ".join(vals) + " |")

    md_path = os.path.join(cfg["output_dir"], "kshot_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tabla Markdown guardada en {md_path}")


# ---- Ejecución directa ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMs generales — k-shot experiment")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Nombre corto: qwen | llama | phi | mistral | all"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Valor específico de k (si no se indica, ejecuta todos los de la config)"
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    df = load_dataset(cfg)
    train_df, val_df, test_df = get_splits(df, cfg)

    if args.model == "all":
        keys = list(cfg["llm_models"].keys())
    else:
        keys = [args.model]

    k_values = [args.k] if args.k else cfg["k_shot_values"]

    results = []
    for key in keys:
        assert key in cfg["llm_models"], f"Modelo '{key}' no encontrado en la config"

        print(f"\n{'='*60}")
        print(f"  K-SHOT: {key}")
        print(f"{'='*60}")

        for k in k_values:
            metrics = run_llm_kshot(key, test_df, train_df, k, cfg)
            results.append({
                "model": key,
                "k": k,
                "accuracy": metrics["accuracy"],
                "precision_explicit": metrics["precision_explicit"],
                "recall_explicit": metrics["recall_explicit"],
                "f1_explicit": metrics["f1_explicit"],
            })

    # Generar resumen
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(cfg["output_dir"], "kshot_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCSV resumen guardado en {summary_path}")

    generate_kshot_markdown_table(summary_df, cfg)
