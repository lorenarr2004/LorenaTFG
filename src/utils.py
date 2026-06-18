# ============================================================
# Utilidades comunes: carga de config, semillas, dispositivo
# ============================================================
import os
import random
import yaml
import torch
import numpy as np

# ============================================================
# ⚠️ TOKEN DE HUGGING FACE - SOLO PARA MODELOS GATED (llama)
# ============================================================
# OPCIÓN 1: Hardcodea tu token aquí
HF_TOKEN = "TU_TOKEN_AQUI"

# OPCIÓN 2: O usa variable de entorno HF_TOKEN
# Si no está en HF_TOKEN, intenta leer de la variable de entorno
if HF_TOKEN is None:
    HF_TOKEN = os.getenv("HF_TOKEN")

# Autenticar si hay token disponible
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("✓ Autenticado en Hugging Face")
    except Exception as e:
        print(f"⚠️ Error al autenticar en HF: {e}")


def load_config(path="configs/config.yaml"):
    """Carga el fichero YAML de configuración."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Devuelve el dispositivo disponible (cuda > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(cfg):
    """Crea los directorios de salida si no existen."""
    for d in [cfg["output_dir"], cfg["predictions_dir"], cfg["metrics_dir"]]:
        os.makedirs(d, exist_ok=True)
