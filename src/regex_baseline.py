# ============================================================
# Baseline regex: clasifica como explicit si contiene
# alguna palabra malsonante de una lista predefinida.
# ============================================================
import re
import time
from src.utils import load_config, set_seed, ensure_dirs
from src.data_loading import load_dataset, get_splits
from src.metrics import compute_metrics, print_metrics, full_report, save_results, save_predictions

# Lista de términos explícitos (inglés principalmente).
# Ampliable fácilmente añadiendo palabras a esta lista.
EXPLICIT_WORDS = [
    # Insultos / palabrotas comunes
    r"\bfuck\w*\b", r"\bshit\w*\b", r"\bbitch\w*\b", r"\bass\b", r"\basshole\w*\b",
    r"\bdamn\w*\b", r"\bhell\b", r"\bcrap\b", r"\bdick\w*\b", r"\bcock\w*\b",
    r"\bpussy\w*\b", r"\bcunt\w*\b", r"\bbastard\w*\b", r"\bwhore\w*\b",
    r"\bslut\w*\b", r"\bprick\w*\b", r"\bmotherfuck\w*\b",
    # Slurs
    r"\bnigga\w*\b", r"\bnigger\w*\b", r"\bfaggot\w*\b", r"\bretard\w*\b",
    # Drogas
    r"\bcocaine\b", r"\bheroin\b", r"\bcrack\b", r"\bblunt\b", r"\bweed\b",
    # Violencia
    r"\bkill\w*\b", r"\bmurder\w*\b",
    # Sexo explícito
    r"\bsex\b", r"\bporn\w*\b", r"\btitties\b", r"\btits\b", r"\bbooty\b",
]


def build_pattern(ignore_case=True):
    """Compila un único patrón regex con todas las palabras."""
    combined = "|".join(EXPLICIT_WORDS)
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(combined, flags)


def predict_regex(texts, ignore_case=True):
    """Devuelve lista de predicciones (1=explicit, 0=clean)."""
    pattern = build_pattern(ignore_case)
    preds = []
    for text in texts:
        match = pattern.search(str(text))
        preds.append(1 if match else 0)
    return preds


def run_regex(df, config_name, cfg):
    """Ejecuta el baseline regex sobre un DataFrame y guarda resultados."""
    texts = df[cfg["text_column"]].tolist()
    labels = df["label"].tolist()

    t0 = time.time()
    preds = predict_regex(texts, ignore_case=cfg.get("regex_ignore_case", True))
    elapsed = time.time() - t0

    metrics = compute_metrics(labels, preds)
    print_metrics(metrics, name=f"Regex Baseline ({config_name})")
    print(full_report(labels, preds))

    save_results(metrics, "regex", config_name, cfg, len(labels), elapsed)
    save_predictions(df, preds, scores=None, name="regex", config_name=config_name, cfg=cfg)
    return metrics


# ---- Ejecución directa ----
if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    df = load_dataset(cfg)

    # Configuración A: dataset completo
    print("\n>>> Configuración A: dataset completo")
    run_regex(df, "full_dataset", cfg)

    # Configuración B: solo test
    print("\n>>> Configuración B: split test")
    _, _, test_df = get_splits(df, cfg)
    run_regex(test_df, "split_test", cfg)
