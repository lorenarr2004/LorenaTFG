# Tarea a implementar

Debes programar el pipeline experimental completo para el TFG de Lorena sobre detección de contenido explícito en letras de canciones.

El objetivo es comparar distintos enfoques de clasificación binaria (`explicit` vs `clean`) usando el dataset `lyrics_10k.csv`.

---

# Requisitos generales

## Dataset
El dataset principal es `lyrics_10k.csv`.

Debe contener al menos:
- texto de la letra
- etiqueta `explicit`

Si los nombres exactos de las columnas cambian, el código debe ser lo bastante robusto como para adaptarse fácilmente o permitir configurarlos en un fichero de configuración.

## Salida esperada
El sistema debe:
- cargar y preparar el dataset
- ejecutar todos los experimentos definidos
- guardar predicciones
- calcular métricas
- generar tablas comparativas de resultados
- guardar resultados en CSV/JSON
- ser modular y reproducible

---

# Plan experimental a implementar

## 1. Preparación de datos

### 1.1. Carga del dataset
Implementar una rutina para:
- cargar `lyrics_10k.csv`
- validar columnas necesarias
- eliminar filas inválidas si falta texto o etiqueta
- normalizar etiquetas a formato binario consistente:
  - `explicit = 1`
  - `clean = 0`

### 1.2. Texto de entrada
Usar la letra completa como texto principal.

Debe existir una función clara para extraer el texto de cada ejemplo.

### 1.3. Split estratificado
Crear una partición estratificada:
- 70% train
- 15% validation
- 15% test

La estratificación debe hacerse sobre la etiqueta `explicit`.

Guardar los índices o particiones para que el experimento sea reproducible.

### 1.4. Reproducibilidad
Fijar semillas aleatorias globales.

---

## 2. Configuraciones de evaluación

Se deben implementar dos configuraciones experimentales.

### Configuración A: evaluación sobre todo el dataset
Aplicar directamente los siguientes métodos al dataset completo:
- regex simple
- modelos especializados
- LLMs generales zero-shot
- LLMs generales few-shot

Objetivo:
- comparación global descriptiva

### Configuración B: evaluación con split
Aplicar:
- regex simple sobre test
- modelos especializados sobre test
- LLMs generales zero-shot sobre test
- LLMs generales few-shot sobre test
- BERT fine-tuned
- RoBERTa fine-tuned

Objetivo:
- evaluación formal de generalización

---

## 3. Métodos a implementar

## 3.1. Baseline regex simple
Implementar un clasificador basado en lista de palabras malsonantes.

### Requisitos
- lista configurable de términos explícitos
- clasificación binaria:
  - si aparece al menos un término -> `explicit`
  - si no aparece ninguno -> `clean`
- opción para ignorar mayúsculas/minúsculas
- guardar predicciones por ejemplo

### Salidas
- predicciones
- métricas
- archivo de resultados

---

## 3.2. Modelos especializados
Implementar inferencia con los siguientes modelos de Hugging Face:

1. `Intel/toxic-prompt-roberta`
2. `unitary/unbiased-toxic-roberta`
3. `s-nlp/roberta_toxicity_classifier`
4. `cardiffnlp/twitter-roberta-base-offensive`

### Requisitos
- usar pipeline o inferencia directa de clasificación
- mapear sus etiquetas a binario `explicit/clean`
- documentar claramente el mapping usado en cada modelo
- permitir truncation configurable
- guardar logits o scores si es posible
- guardar predicciones y métricas

### Nota
Puede ser necesario adaptar cada modelo porque sus etiquetas no siempre coincidirán exactamente con `explicit/clean`.

---

## 3.3. LLMs generales en zero-shot
Implementar clasificación mediante prompting con estos modelos:

1. `Qwen/Qwen2.5-7B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`
3. `microsoft/Phi-4-mini-instruct`
4. `mistralai/Mistral-7B-Instruct-v0.3`

### Requisitos
- usar formato instruct/chat si corresponde
- construir un prompt zero-shot común para todos los modelos
- salida obligatoria restringida a:
  - `explicit`
  - `clean`
- implementar parser robusto de salida
- si la respuesta es ambigua, registrar el caso y aplicar una estrategia consistente
- soportar inferencia por lotes si es viable
- permitir cuantización si hace falta por memoria

### Prompt base
Debe existir una plantilla parecida a esta:

Classify the following song lyrics as either `explicit` or `clean`.
Label as `explicit` if the lyrics contain profanity, offensive slurs, explicit sexual references, or clearly inappropriate violent/offensive language.
Otherwise label as `clean`.
Reply with exactly one word: `explicit` or `clean`.

Lyrics:
{lyrics}

Response:

La implementación puede refinar el prompt, pero debe mantenerse consistente entre modelos.

---

## 3.4. LLMs generales en few-shot
Usar los mismos 4 modelos generales, pero con ejemplos en el prompt.

### Requisitos
- usar una única configuración few-shot fija
- incluir 4 ejemplos en total:
  - 2 explícitos
  - 2 limpios
- los ejemplos few-shot deben provenir solo del conjunto de entrenamiento cuando se trabaje con la configuración con split
- no debe haber fuga de test hacia el prompt
- el prompt few-shot debe mantenerse constante entre modelos

### Importante
La selección de ejemplos debe ser reproducible.

---

## 3.5. Fine-tuning supervisado: BERT
Implementar fine-tuning de:
- `google-bert/bert-base-uncased`

### Requisitos
- usar `BertForSequenceClassification`
- clasificación binaria
- entrenar en train
- validar en validation
- evaluar en test
- usar `Trainer` o una implementación equivalente clara y mantenible
- guardar:
  - mejor checkpoint
  - métricas por época
  - predicciones finales en test
  - matriz de confusión

### Hiperparámetros
Deben ser configurables, por ejemplo:
- batch size
- learning rate
- número de épocas
- max length
- early stopping opcional

---

## 3.6. Fine-tuning supervisado: RoBERTa
Implementar fine-tuning de:
- `FacebookAI/roberta-base`

### Requisitos
- usar `RobertaForSequenceClassification`
- mismo protocolo experimental que BERT
- guardar los mismos artefactos y métricas

RoBERTa será el modelo supervisado principal.

---

# 4. Preprocesamiento y longitud de entrada

## Requisito mínimo
Implementar truncation configurable para todos los modelos que lo necesiten.

## Deseable
Diseñar el código de forma que en el futuro pueda ampliarse a chunking, aunque no es obligatorio implementarlo ahora para el fine-tuning.

---

# 5. Métricas

Calcular al menos:
- accuracy
- precision
- recall
- F1
- F1 de la clase positiva (`explicit`)
- matriz de confusión

Si es sencillo, añadir también:
- support por clase
- classification report completo

La métrica principal a destacar será:
- **F1 de la clase explícita**

---

# 6. Resultados y persistencia

## Guardar resultados por experimento
Cada experimento debe guardar:
- nombre del método/modelo
- configuración usada (`full_dataset` o `split_test`)
- métricas
- número de ejemplos evaluados
- tiempo de ejecución si es posible

## Guardar predicciones por instancia
Guardar para cada ejemplo:
- id o índice
- texto o referencia al ejemplo
- etiqueta real
- predicción
- score/probabilidad si existe

## Formatos
Usar formatos fáciles de analizar después:
- CSV
- JSON

---

# 7. Organización del código

Se desea una estructura modular, por ejemplo:

- `data/`
- `src/`
  - `data_loading.py`
  - `splits.py`
  - `metrics.py`
  - `regex_baseline.py`
  - `specialized_models.py`
  - `llm_zero_shot.py`
  - `llm_few_shot.py`
  - `finetune_bert.py`
  - `finetune_roberta.py`
  - `prompts.py`
  - `utils.py`
- `results/`
- `configs/`
- `run_all.py`

No es obligatorio que los nombres sean exactamente estos, pero sí que la estructura sea limpia.

---

# 8. Requisitos metodológicos importantes

## Comparabilidad
- usar métricas homogéneas para todos los modelos
- documentar claramente el mapping de etiquetas en modelos especializados
- usar prompts consistentes entre LLMs generales

## No fuga de información
- en la configuración con split, el few-shot no puede usar ejemplos del test
- el test solo debe usarse para evaluación final

## Reproducibilidad
- semillas fijas
- particiones guardadas
- configuración serializable

---

# 9. Entregables esperados del código

Claude debe generar un sistema que permita ejecutar:

1. baseline regex
2. inferencia con 4 modelos especializados
3. zero-shot con 4 LLMs generales
4. few-shot con 4 LLMs generales
5. fine-tuning de BERT
6. fine-tuning de RoBERTa
7. resumen final comparativo de resultados

También se desea un script principal o conjunto de scripts que permitan lanzar:
- experimentos individuales
- todos los experimentos de forma controlada

---

# 10. Prioridades

## Prioridad alta
- regex simple
- modelos especializados
- zero-shot con LLMs generales
- split estratificado
- fine-tuning de BERT
- fine-tuning de RoBERTa
- métricas y guardado de resultados

## Prioridad media
- few-shot con LLMs generales
- logging más detallado
- tiempos de inferencia

## Prioridad baja
- optimizaciones avanzadas
- chunking sofisticado
- visualizaciones

---

# 11. Restricciones prácticas
- hay una GPU con 40 GB de VRAM
- el código debe ser realista para ejecutarse en ese entorno
- evitar soluciones innecesariamente complejas
- priorizar robustez, claridad y reproducibilidad

---

# 12. Objetivo final
El resultado debe permitir comparar de manera rigurosa y reproducible cuatro enfoques:

1. reglas simples
2. modelos especializados ya ajustados
3. LLMs generales por prompting
4. modelos ajustados específicamente al dominio mediante fine-tuning

El código debe quedar preparado para producir resultados utilizables directamente en el TFG.