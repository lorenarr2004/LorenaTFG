# ============================================================
# Orquestador: ejecuta todos los experimentos del TFG
# Uso: python run_all.py
# ============================================================
import json
import os
import glob
import pandas as pd

from src.utils import load_config, set_seed, ensure_dirs
from src.data_loading import load_dataset, get_splits
from src.regex_baseline import run_regex
from src.specialized_models import run_specialized_model
from src.llm_zero_shot import run_llm_zero_shot
from src.llm_few_shot import run_kshot_experiment
from src.finetune import run_finetune


def run_all():
    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    df = load_dataset(cfg)
    train_df, val_df, test_df = get_splits(df, cfg)

    # ========================================
    # 1. Regex baseline
    # ========================================
#    print("\n" + "="*60)
#    print("  REGEX BASELINE")
#    print("="*60)
#    run_regex(df, "full_dataset", cfg)          # Config A
#    run_regex(test_df, "split_test", cfg)       # Config B
#
#    # ========================================
#    # 2. Modelos especializados
#    # ========================================
#    print("\n" + "="*60)
#    print("  MODELOS ESPECIALIZADOS")
#    print("="*60)
#    for key in cfg["specialized_models"]:
#        run_specialized_model(key, df, "full_dataset", cfg)       # Config A
#        run_specialized_model(key, test_df, "split_test", cfg)    # Config B

    # ========================================
    # 3. LLMs generales — zero-shot
    # ========================================
#    print("\n" + "="*60)
#    print("  LLMs ZERO-SHOT")
#    print("="*60)
#    for key in cfg["llm_models"]:
#        run_llm_zero_shot(key, df, "full_dataset", cfg)           # Config A
#        run_llm_zero_shot(key, test_df, "split_test", cfg)        # Config B

    # ========================================
    # 4. LLMs generales — k-shot (Config B)
    # ========================================
    print("\n" + "="*60)
    print("  LLMs K-SHOT EXPERIMENT")
    print("="*60)
    run_kshot_experiment(cfg)

#    # ========================================
#    # 5. Fine-tuning (solo Config B)
#    # ========================================
#    print("\n" + "="*60)
#    print("  FINE-TUNING")
#    print("="*60)
#    for key in cfg["finetune_models"]:
#        run_finetune(key, cfg)

    # ========================================
    # 6. Tabla resumen
    # ========================================
#    print("\n" + "="*60)
#    print("  GENERANDO TABLA RESUMEN")
#    print("="*60)
#    generate_summary(cfg)


def generate_summary(cfg):
    """Lee todos los JSON de métricas y genera una tabla comparativa."""
    json_files = glob.glob(os.path.join(cfg["metrics_dir"], "*.json"))
    rows = []
    for path in sorted(json_files):
        with open(path) as f:
            data = json.load(f)
        rows.append({
            "method": data.get("method", ""),
            "config": data.get("config", ""),
            "accuracy": data.get("accuracy", None),
            "precision_explicit": data.get("precision_explicit", None),
            "recall_explicit": data.get("recall_explicit", None),
            "f1_explicit": data.get("f1_explicit", None),
            "n_examples": data.get("n_examples", None),
            "elapsed_seconds": data.get("elapsed_seconds", None),
        })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(cfg["output_dir"], "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nTabla resumen guardada en {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_all()
