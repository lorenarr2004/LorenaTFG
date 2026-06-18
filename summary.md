# Análisis de errores por familia de modelos

- Métrica de selección del mejor modelo por familia: `f1_explicit`
- Split analizado: `split_test`  (1300 canciones)

## Modelos seleccionados

| Familia | Modelo elegido | f1_explicit |
|---|---|---|
| regex | `regex` | 0.6406 |
| roberta | `unbiased-toxic` | 0.6815 |
| llama | `llama_zeroshot` | 0.7170 |
| qwen | `qwen_kshot_k4` | 0.6994 |
| bert | `bert_finetuned` | 0.7492 |

## Falsos positivos y falsos negativos por modelo

- **FP** (falso positivo): canción *clean* (label=0) clasificada como *explicit*.
- **FN** (falso negativo): canción *explicit* (label=1) clasificada como *clean*.

- Reales en test: 278 explicit (1), 1022 clean (0).

| Modelo | FP | FN | Errores totales |
|---|---|---|---|
| regex | 261 | 24 | 285 |
| roberta | 167 | 48 | 215 |
| llama | 130 | 50 | 180 |
| qwen | 146 | 50 | 196 |
| bert | 121 | 39 | 160 |

## Distribución por nº de modelos que fallan

| nº fallos | nº canciones | % | de los cuales FP | FN |
|---|---|---|---|---|
| 0 | 886 | 68.2% | 0 | 0 |
| 1 | 184 | 14.2% | 150 | 34 |
| 2 | 64 | 4.9% | 44 | 20 |
| 3 | 41 | 3.2% | 30 | 11 |
| 4 | 24 | 1.8% | 13 | 11 |
| 5 | 101 | 7.8% | 89 | 12 |
