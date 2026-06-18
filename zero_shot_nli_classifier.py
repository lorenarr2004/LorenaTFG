 #!/usr/bin/env python3
"""
Zero-Shot NLI Classifier for Explicit Content Detection in Song Lyrics
=======================================================================
Este script clasifica letras de canciones como "explicit" o "clean" usando
modelos de Natural Language Inference (NLI) en modo zero-shot.

Implementa chunking inteligente con estrategia ANY-POSITIVE: si algún chunk
de la letra es explícito, toda la canción es clasificada como explícita.

Uso:
    python zero_shot_nli_classifier.py --model bart
    python zero_shot_nli_classifier.py --model deberta-v3-large --overlap 50
    python zero_shot_nli_classifier.py --list-models

Autor: TFG - Clasificación de contenido explícito
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import pipeline, AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =============================================================================
# CONFIGURACIÓN DE MODELOS
# =============================================================================

MODELS_CONFIG: Dict[str, Dict] = {
    "bart": {
        "model_id": "facebook/bart-large-mnli",
        "description": "BART Large fine-tuned on MNLI (418M params)",
        "max_length": 1024,
        "max_tokens": 900,  # Dejar margen para hypothesis
    },
    "deberta-v3-large": {
        "model_id": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "description": "DeBERTa V3 Large fine-tuned on MNLI+FEVER+ANLI+LingNLI+WANLI (435M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
    "deberta-v3-base": {
        "model_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "description": "DeBERTa V3 Base fine-tuned on MNLI+FEVER+ANLI (184M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
    "deberta-v3-xsmall": {
        "model_id": "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",
        "description": "DeBERTa V3 XSmall fine-tuned on MNLI+FEVER+ANLI+LingNLI binary (22M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
    "mdeberta": {
        "model_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "description": "mDeBERTa V3 Base multilingual NLI (280M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
    "cross-encoder": {
        "model_id": "cross-encoder/nli-deberta-v3-large",
        "description": "Cross-Encoder DeBERTa V3 Large for NLI (435M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
    "roberta-nli": {
        "model_id": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "description": "RoBERTa Large trained on multiple NLI datasets (355M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
    "nli-roberta-base": {
        "model_id": "cross-encoder/nli-roberta-base",
        "description": "Cross-Encoder RoBERTa Base for NLI (125M params)",
        "max_length": 512,
        "max_tokens": 450,
    },
}

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

CSV_INPUT = "results/lyrics/lyrics_10k.csv"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "results/lyrics"

# =============================================================================
# HIPÓTESIS NLI Y LABELS
# =============================================================================

# Template de hipótesis para zero-shot NLI
# Se insertará cada label en el placeholder {}
HYPOTHESIS_TEMPLATE = "This song contains {}."

# Labels candidatos para zero-shot - Definición completa de explicit content
# OPCIÓN 1: Descriptiva y completa (RECOMENDADA) - ~12 tokens
CANDIDATE_LABELS = [
    "profanity, sexual content, violence, drugs, or hate speech",
    "clean and family-friendly content"
]

# ===== ALTERNATIVAS PARA EXPERIMENTAR =====

# Opción 2: Más concisa sin drogas (~10 tokens)
# CANDIDATE_LABELS = [
#     "profanity, sexual content, violence, or hate speech",
#     "clean and appropriate content"
# ]

# Opción 3: Con categorización explícita (~14 tokens)
# CANDIDATE_LABELS = [
#     "explicit content: profanity, sex, violence, drugs, or hate",
#     "clean content appropriate for all audiences"
# ]

# Opción 4: Más corta pero menos específica (~3-5 tokens)
# CANDIDATE_LABELS = ["explicit content", "clean content"]

# Opción 5: Labels simples (~1 token) - NO RECOMENDADO
# CANDIDATE_LABELS = ["explicit", "clean"]

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_device() -> torch.device:
    """Detecta y devuelve el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detectada: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Usando MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("🐌 Usando CPU")
        return torch.device("cpu")


def list_available_models() -> None:
    """Muestra todos los modelos disponibles."""
    print("\n" + "=" * 70)
    print("MODELOS DISPONIBLES PARA ZERO-SHOT NLI")
    print("=" * 70)
    for key, config in MODELS_CONFIG.items():
        print(f"\n  --model {key}")
        print(f"     ID: {config['model_id']}")
        print(f"     {config['description']}")
    print("\n" + "=" * 70)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carga y prepara el dataset.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con columnas 'lyrics' y 'explicit'
    """
    print(f"\n📂 Cargando dataset desde: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Filas totales: {len(df)}")
    
    # Verificar columnas requeridas
    if 'lyrics' not in df.columns:
        # Intentar encontrar columna de texto alternativa
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'lyric' in c.lower()]
        if text_cols:
            df = df.rename(columns={text_cols[0]: 'lyrics'})
        else:
            raise ValueError("No se encontró columna 'lyrics' o similar")
    
    if 'explicit' not in df.columns:
        # Intentar encontrar columna de etiqueta alternativa
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'explicit' in c.lower()]
        if label_cols:
            df = df.rename(columns={label_cols[0]: 'explicit'})
        else:
            raise ValueError("No se encontró columna 'explicit' o 'label'")
    
    # Limpiar datos
    original_len = len(df)
    df = df.dropna(subset=['lyrics'])
    df = df[df['lyrics'].str.strip() != '']
    dropped = original_len - len(df)
    
    if dropped > 0:
        print(f"   ⚠️  Eliminadas {dropped} filas con texto vacío/nulo")
    
    # Convertir explicit a int si no lo es
    df['explicit'] = df['explicit'].astype(int)
    
    # Estadísticas
    explicit_count = df['explicit'].sum()
    clean_count = len(df) - explicit_count
    print(f"   Distribución: {explicit_count} explicit ({100*explicit_count/len(df):.1f}%) | "
          f"{clean_count} clean ({100*clean_count/len(df):.1f}%)")
    
    return df


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """
    Trunca texto largo para evitar problemas de memoria.
    
    Args:
        text: Texto a truncar
        max_chars: Máximo de caracteres
        
    Returns:
        Texto truncado
    """
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())  # Normalizar espacios
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 450,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Divide un texto en chunks basados en tokens, con overlap para no perder contexto.
    
    Estrategia: Dividir en chunks de max_tokens con overlap, asegurando
    que no se cortan palabras a mitad.
    
    Args:
        text: Texto a dividir
        tokenizer: Tokenizer del modelo
        max_tokens: Máximo de tokens por chunk
        overlap_tokens: Tokens de solapamiento entre chunks
        
    Returns:
        Lista de chunks de texto
    """
    if not isinstance(text, str) or not text.strip():
        return [""]
    
    # Normalizar espacios
    text = " ".join(text.split())
    
    # Tokenizar el texto completo
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Si cabe en un solo chunk, devolver tal cual
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    step = max_tokens - overlap_tokens
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Decodificar chunk
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunk_text = chunk_text.strip()
        
        if chunk_text:
            chunks.append(chunk_text)
        
        # Avanzar con overlap
        start += step
        
        # Evitar chunks muy pequeños al final
        if len(tokens) - start < overlap_tokens:
            break
    
    return chunks if chunks else [text[:1000]]  # Fallback


def classify_single_text(
    pipe,
    text: str,
    tokenizer,
    max_tokens: int = 450,
    overlap_tokens: int = 50,
) -> Tuple[bool, float, int]:
    """
    Clasifica un texto, dividiendo en chunks si es necesario.
    
    Estrategia ANY-POSITIVE: Si CUALQUIER chunk es explícito, 
    toda la canción es explícita.
    
    Args:
        pipe: Pipeline de zero-shot
        text: Texto a clasificar
        tokenizer: Tokenizer para chunking
        max_tokens: Máximo tokens por chunk
        overlap_tokens: Solapamiento entre chunks
        
    Returns:
        Tuple de (is_explicit, max_probability, num_chunks)
    """
    # Dividir en chunks
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens, overlap_tokens)
    
    max_prob_explicit = 0.0
    any_explicit = False
    
    # Label explícito es SIEMPRE el primero de CANDIDATE_LABELS
    explicit_label = CANDIDATE_LABELS[0]
    
    for chunk in chunks:
        try:
            result = pipe(
                chunk,
                candidate_labels=CANDIDATE_LABELS,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )
            
            # El pipeline devuelve labels y scores ordenados por score descendente.
            # Buscamos el score correspondiente a nuestro label explícito.
            label_to_score = dict(zip(result['labels'], result['scores']))
            prob_explicit = label_to_score.get(explicit_label, 0.0)
            
            # Si la probabilidad de explícito >= 0.5, este chunk es explícito
            if prob_explicit >= 0.5:
                any_explicit = True
            
            # Guardar la máxima probabilidad
            max_prob_explicit = max(max_prob_explicit, prob_explicit)
            
            # Early exit: si ya encontramos explicit con alta confianza
            if any_explicit and prob_explicit > 0.7:
                break
                
        except Exception as e:
            # En caso de error, continuar con el siguiente chunk
            continue
    
    return any_explicit, max_prob_explicit, len(chunks)


def create_pipeline(model_key: str, device: torch.device) -> Tuple[pipeline, AutoTokenizer, dict]:
    """
    Crea el pipeline de zero-shot classification y el tokenizer.
    
    Args:
        model_key: Clave del modelo en MODELS_CONFIG
        device: Dispositivo (cuda/cpu/mps)
        
    Returns:
        Tuple de (pipeline, tokenizer, config)
    """
    if model_key not in MODELS_CONFIG:
        raise ValueError(f"Modelo '{model_key}' no reconocido. Usa --list-models")
    
    config = MODELS_CONFIG[model_key]
    model_id = config["model_id"]
    
    print(f"\n🔧 Cargando modelo: {model_id}")
    print(f"   {config['description']}")
    
    # Determinar device index para pipeline
    device_idx = 0 if device.type == "cuda" else -1
    
    # Cargar tokenizer para chunking (use_fast=False para compatibilidad
    # en entornos sin tiktoken/protobuf extras)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    
    # Crear pipeline
    pipe = pipeline(
        "zero-shot-classification",
        model=model_id,
        device=device_idx,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
    )
    
    print("   ✅ Modelo cargado correctamente")
    
    return pipe, tokenizer, config


def run_inference(
    pipe,
    tokenizer,
    texts: List[str],
    max_tokens: int = 450,
    overlap_tokens: int = 50,
) -> Tuple[List[bool], List[float], List[int], float]:
    """
    Ejecuta inferencia zero-shot sobre los textos con chunking inteligente.
    
    Estrategia ANY-POSITIVE: Una canción es explícita si CUALQUIER 
    chunk de la letra es clasificado como explícito.
    
    Args:
        pipe: Pipeline de transformers
        tokenizer: Tokenizer para chunking
        texts: Lista de textos
        max_tokens: Máximo de tokens por chunk
        overlap_tokens: Solapamiento entre chunks
        
    Returns:
        Tuple de (predicciones, probabilidades, num_chunks_por_texto, tiempo_total)
    """
    predictions = []
    probabilities = []
    chunks_counts = []
    
    print(f"\n🔄 Ejecutando inferencia con chunking inteligente...")
    print(f"   Template: '{HYPOTHESIS_TEMPLATE}'")
    print(f"   Labels: {CANDIDATE_LABELS}")
    print(f"   Max tokens/chunk: {max_tokens}")
    print(f"   Overlap tokens: {overlap_tokens}")
    print(f"   Estrategia: ANY-POSITIVE (si algún chunk es explícito → canción explícita)")
    
    start_time = time.time()
    total_chunks = 0
    
    # Procesar cada texto con barra de progreso
    for text in tqdm(texts, desc="Clasificando canciones"):
        try:
            is_explicit, prob_explicit, num_chunks = classify_single_text(
                pipe=pipe,
                text=text,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            
            predictions.append(is_explicit)
            probabilities.append(prob_explicit)
            chunks_counts.append(num_chunks)
            total_chunks += num_chunks
            
        except Exception as e:
            print(f"\n   ⚠️  Error procesando texto: {e}")
            predictions.append(False)
            probabilities.append(0.5)
            chunks_counts.append(1)
    
    total_time = time.time() - start_time
    
    # Estadísticas de chunking
    avg_chunks = np.mean(chunks_counts)
    max_chunks = max(chunks_counts)
    songs_with_multiple = sum(1 for c in chunks_counts if c > 1)
    
    print(f"\n📊 Estadísticas de chunking:")
    print(f"   Total chunks procesados: {total_chunks}")
    print(f"   Promedio chunks/canción: {avg_chunks:.2f}")
    print(f"   Máximo chunks en una canción: {max_chunks}")
    print(f"   Canciones divididas en múltiples chunks: {songs_with_multiple} ({100*songs_with_multiple/len(texts):.1f}%)")
    
    return predictions, probabilities, chunks_counts, total_time


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_prob: Probabilidades de la clase positiva
        
    Returns:
        Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    
    # AUC solo si hay ambas clases
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = None
    
    return metrics


def generate_report(
    model_key: str,
    model_config: Dict,
    metrics: Dict,
    total_time: float,
    n_samples: int,
    chunks_stats: Dict,
    output_path: str,
) -> str:
    """
    Genera el informe de resultados.
    
    Args:
        model_key: Clave del modelo
        model_config: Configuración del modelo
        metrics: Diccionario de métricas
        total_time: Tiempo total de inferencia
        n_samples: Número de muestras
        chunks_stats: Estadísticas de chunking
        output_path: Ruta de salida
        
    Returns:
        Contenido del informe
    """
    cm = metrics['confusion_matrix']
    avg_time = total_time / n_samples if n_samples > 0 else 0
    sep = '=' * 70
    
    # Formatear AUC fuera del f-string para evitar problemas con el condicional
    auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else 'N/A'
    
    report = f"""{sep}
INFORME DE CLASIFICACIÓN ZERO-SHOT NLI
{sep}

Modelo: {model_key}
ID: {model_config['model_id']}
Descripción: {model_config['description']}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Hypothesis Template: "{HYPOTHESIS_TEMPLATE}"
Candidate Labels: {CANDIDATE_LABELS}

{sep}
ESTRATEGIA DE CHUNKING
{sep}

Estrategia: ANY-POSITIVE (si algún chunk es explícito → canción explícita)
Max tokens por chunk: {model_config.get('max_tokens', 450)}
Overlap tokens: 50

Estadísticas:
  - Total chunks procesados:     {chunks_stats['total_chunks']}
  - Promedio chunks/canción:     {chunks_stats['avg_chunks']:.2f}
  - Máximo chunks en una canción: {chunks_stats['max_chunks']}
  - Canciones con múltiples chunks: {chunks_stats['songs_multiple']} ({chunks_stats['pct_multiple']:.1f}%)

{sep}
MÉTRICAS DE CLASIFICACIÓN
{sep}

Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1']:.4f}
AUC-ROC:   {auc_str}

{sep}
MATRIZ DE CONFUSIÓN
{sep}

                    Predicted
                 No (Clean)  Yes (Explicit)
Actual No (Clean)    {cm[0][0]:6d}        {cm[0][1]:6d}
Actual Yes (Explicit){cm[1][0]:6d}        {cm[1][1]:6d}

Interpretación:
  - True Negatives (TN):  {cm[0][0]:6d} (Clean clasificado como Clean)
  - False Positives (FP): {cm[0][1]:6d} (Clean clasificado como Explicit)
  - False Negatives (FN): {cm[1][0]:6d} (Explicit clasificado como Clean)
  - True Positives (TP):  {cm[1][1]:6d} (Explicit clasificado como Explicit)

{sep}
TIEMPOS DE EJECUCIÓN
{sep}

Muestras procesadas: {n_samples}
Tiempo total:        {total_time:.2f} segundos
Tiempo promedio:     {avg_time*1000:.2f} ms/muestra
Velocidad:           {n_samples/total_time:.1f} muestras/segundo

{sep}
"""
    
    # Guardar informe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def save_predictions(
    df: pd.DataFrame,
    predictions: List[bool],
    probabilities: List[float],
    model_key: str,
) -> str:
    """
    Guarda las predicciones en un CSV.
    
    Args:
        df: DataFrame original
        predictions: Lista de predicciones
        probabilities: Lista de probabilidades
        model_key: Clave del modelo
        
    Returns:
        Ruta del archivo guardado
    """
    df_out = df.copy()
    df_out['predicted_explicit'] = predictions
    df_out['prob_explicit'] = probabilities
    
    output_path = os.path.join(PREDICTIONS_DIR, f"lyrics_10k_predictions_zeroshot_{model_key}.csv")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    
    return output_path


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal del script."""
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description="Zero-Shot NLI Classifier for Explicit Content Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python zero_shot_nli_classifier.py --model bart
  python zero_shot_nli_classifier.py --model deberta-v3-large
  python zero_shot_nli_classifier.py --list-models
  
Estrategia de chunking:
  - Las canciones largas se dividen en chunks de max_tokens
  - Se usa overlap para no perder contexto entre chunks
  - ANY-POSITIVE: Si algún chunk es explícito, la canción es explícita
        """
    )
    
    parser.add_argument('--model', '-m',
                        type=str,choices=list(MODELS_CONFIG.keys()),
                        help='Modelo a usar (ver --list-models para opciones)')
    
    parser.add_argument( '--list-models', '-l',
        action='store_true',
        help='Lista todos los modelos disponibles'
    )
    
    parser.add_argument( '--overlap', '-o',
        type=int,
        default=50,
        help='Tokens de solapamiento entre chunks (default: 50)'
    )
    
    parser.add_argument( '--input', '-i',
        type=str,
        default=CSV_INPUT,
        help=f'Ruta al CSV de entrada (default: {CSV_INPUT})'
    )
    
    args = parser.parse_args()
    
    # Listar modelos si se solicita
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Verificar que se especificó un modelo
    if not args.model:
        parser.print_help()
        print("\n❌ Error: Debes especificar un modelo con --model")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ZERO-SHOT NLI CLASSIFIER - EXPLICIT CONTENT DETECTION")
    print("=" * 70)
    
    # Detectar dispositivo
    device = get_device()
    
    # Cargar dataset
    df = load_dataset(args.input)
    
    # Crear pipeline con tokenizer
    pipe, tokenizer, config = create_pipeline(args.model, device)
    
    # Obtener max_tokens del modelo
    max_tokens = config.get('max_tokens', 450)
    
    # Ejecutar inferencia con chunking
    predictions, probabilities, chunks_counts, total_time = run_inference(
        pipe=pipe,
        tokenizer=tokenizer,
        texts=df['lyrics'].tolist(),
        max_tokens=max_tokens,
        overlap_tokens=args.overlap,
    )
    
    # Calcular estadísticas de chunking
    chunks_stats = {
        'total_chunks': sum(chunks_counts),
        'avg_chunks': np.mean(chunks_counts),
        'max_chunks': max(chunks_counts),
        'songs_multiple': sum(1 for c in chunks_counts if c > 1),
        'pct_multiple': 100 * sum(1 for c in chunks_counts if c > 1) / len(chunks_counts),
    }
    
    # Calcular métricas
    y_true = df['explicit'].values
    y_pred = np.array(predictions).astype(int)
    y_prob = np.array(probabilities)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Generar informe
    report_path = os.path.join(RESULTS_DIR, f"zeroshot_{args.model}.txt")
    report = generate_report(
        model_key=args.model,
        model_config=config,
        metrics=metrics,
        total_time=total_time,
        n_samples=len(df),
        chunks_stats=chunks_stats,
        output_path=report_path,
    )
    
    # Guardar predicciones con información de chunks
    df_out = df.copy()
    df_out['predicted_explicit'] = predictions
    df_out['prob_explicit'] = probabilities
    df_out['num_chunks'] = chunks_counts
    
    predictions_path = os.path.join(PREDICTIONS_DIR, f"lyrics_10k_predictions_zeroshot_{args.model}.csv")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    df_out.to_csv(predictions_path, index=False)
    
    # Mostrar resultados
    print(report)
    print(f"📄 Informe guardado en: {report_path}")
    print(f"📊 Predicciones guardadas en: {predictions_path}")
    print("\n✅ Ejecución completada exitosamente")


if __name__ == "__main__":
    main()
