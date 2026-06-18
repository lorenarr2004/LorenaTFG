# Resumen del Proyecto: Detección de Contenido Explícito en Letras de Canciones

## Objetivo

Este proyecto es un **Trabajo de Fin de Grado (TFG)** cuyo objetivo es evaluar distintos modelos de lenguaje pre-entrenados para **clasificar letras de canciones como "explícitas" o "limpias"**, replicando la etiqueta de contenido explícito que usa Spotify. Todos los modelos se usan en modo **zero-shot** (sin fine-tuning), es decir, directamente sin entrenamiento específico para esta tarea.

---

## Datos de entrada

El dataset es `lyrics_10k.csv`, con ~10.000 canciones obtenidas de Spotify. Cada fila contiene:

- Título de la canción y artista
- Letra completa de la canción
- Etiqueta real de Spotify (`explicit: True/False`)
- Géneros musicales

**Distribución**: ~78,6% canciones limpias / ~21,4% explícitas (dataset desbalanceado).

---

## Enfoques y modelos evaluados

### 1. Clasificación Zero-Shot vía NLI (`zero_shot_nli_classifier.py`)
Se usa el pipeline de clasificación zero-shot de Hugging Face, que reformula el problema como **inferencia de lenguaje natural (NLI)**. Las letras largas se dividen en fragmentos solapados; si algún fragmento supera el umbral de "explícito", la canción se marca como tal.

Modelos probados: BART-large-MNLI, DeBERTa-v3 (varias tallas), mDeBERTa, RoBERTa-NLI, cross-encoder.

### 2. Modelos generativos como clasificadores (`small_ultrafast_DistilBART.py`, `small_ultrafast_TinyLlama.py`)
Se le pide al modelo que genere exactamente un token: "true" o "false". Modelos: DistilBART, TinyLlama (1.1B parámetros).

### 3. Clasificadores discriminativos reutilizados zero-shot (`small_ultrafast_distilbert.py`, `small_ultrafast_roberTA.py`, `small_ultrafast_electra_nativa.py`)
Modelos de clasificación reutilizados con etiquetas nuevas sin reentrenamiento. Incluye DistilBERT, ELECTRA-small, y un modelo de sentimiento de Twitter (RoBERTa) al que se le da un uso inusual: se mapea su etiqueta "negativo" como proxy de contenido explícito.

---

## Resultados destacados

| Modelo | Accuracy | F1 |
|---|---|---|
| RoBERTa-Sentiment (Twitter) | 0.752 | **0.459** |
| DistilBERT-base | 0.788 | 0.199 |
| ELECTRA-small | 0.647 | 0.181 |
| DistilBART / BART-MNLI / mDeBERTa | ~0.786 | 0.000 |

- La mayoría de modelos NLI **colapsan prediciendo siempre "limpia"**, logrando solo la precisión de la clase mayoritaria (~78,6%).
- El mejor resultado lo obtiene **RoBERTa-Sentiment**, aprovechando que el contenido explícito se correlaciona con sentimiento negativo.
- Los modelos generativos pequeños no siguen bien las instrucciones de generar un único token de clasificación.

---

## Infraestructura

- Ejecución en **clúster GPU con SLURM** (`exe.sh`), con hasta 8 trabajos en paralelo (job array).
- Entorno Python aislado en `./env/` (conda).
- Modelos cargados desde **Hugging Face Hub**.
- Los resultados se guardan en `results/` como métricas (`.txt`) y predicciones (`.csv`).
