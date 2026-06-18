# ============================================================
# LLMs generales — clasificación zero-shot
# Uso: python -m src.llm_zero_shot --model <nombre_corto>
#   nombres cortos: qwen | llama | phi | mistral | all
# ============================================================
import argparse
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.data_loading import load_dataset, get_splits
from src.prompts import build_zero_shot_prompt
from src.metrics import compute_metrics, print_metrics, full_report, save_results, save_predictions


def parse_response(text: str) -> int:
    """
    Extrae la predicción del texto generado por el LLM.
    Busca 'explicit' o 'clean'. Si es ambiguo, devuelve 0 (clean).
    """
    text_lower = text.strip().lower()
    # Buscar la primera aparición de alguna de las dos etiquetas
    match = re.search(r"\b(explicit|clean)\b", text_lower)
    if match:
        return 1 if match.group(1) == "explicit" else 0
    # Fallback: clean
    return 0


def load_llm(model_name, device):
    """Carga tokenizer y modelo causal con cuantización automática si es necesario."""
    print(f"Cargando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Intentar cargar en float16; si no cabe, usar quantización 4-bit
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        print("  -> float16 falló, usando cuantización 4-bit")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def classify_batch(texts, tokenizer, model, cfg):
    """
    Clasifica una lista de textos construyendo prompts zero-shot.
    Devuelve listas de predicciones y respuestas crudas.
    """
    max_len = cfg.get("llm_max_length", 2048)
    max_new = cfg.get("llm_max_new_tokens", 10)
    preds, raw_responses = [], []

    for text in tqdm(texts, desc="Zero-shot"):
        prompt = build_zero_shot_prompt(text[:3000])  # truncar letras muy largas

        # Construir mensajes en formato chat si el modelo lo soporta
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
                do_sample=False,  # greedy para reproducibilidad
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decodificar solo los tokens nuevos
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw_responses.append(response)
        preds.append(parse_response(response))

    return preds, raw_responses


def run_llm_zero_shot(model_key, df, config_name, cfg):
    """Ejecuta un LLM general en zero-shot sobre un DataFrame."""
    model_info = cfg["llm_models"][model_key]
    model_name = model_info["name"]
    device = get_device()

    tokenizer, model = load_llm(model_name, device)

    texts = df[cfg["text_column"]].tolist()
    labels = df["label"].tolist()

    t0 = time.time()
    preds, raw = classify_batch(texts, tokenizer, model, cfg)
    elapsed = time.time() - t0

    metrics = compute_metrics(labels, preds)
    print_metrics(metrics, name=f"{model_key} zero-shot ({config_name})")
    print(full_report(labels, preds))

    save_results(metrics, f"{model_key}_zeroshot", config_name, cfg, len(labels), elapsed)
    save_predictions(df, preds, scores=None, name=f"{model_key}_zeroshot", config_name=config_name, cfg=cfg)

    # Liberar memoria
    del model, tokenizer
    torch.cuda.empty_cache()
    return metrics


# ---- Ejecución directa ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMs generales — zero-shot")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Nombre corto: qwen | llama | phi | mistral | all"
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    df = load_dataset(cfg)

    if args.model == "all":
        keys = list(cfg["llm_models"].keys())
    else:
        keys = [args.model]

    for key in keys:
        assert key in cfg["llm_models"], f"Modelo '{key}' no encontrado en la config"

        # Configuración A: dataset completo
        print(f"\n>>> {key} zero-shot - Configuración A: dataset completo")
        run_llm_zero_shot(key, df, "full_dataset", cfg)

        # Configuración B: solo test
        print(f"\n>>> {key} zero-shot - Configuración B: split test")
        _, _, test_df = get_splits(df, cfg)
        run_llm_zero_shot(key, test_df, "split_test", cfg)
