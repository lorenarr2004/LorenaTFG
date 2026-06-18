# Contexto del proyecto: TFG de Lorena

## Título / tema general
Este proyecto corresponde al Trabajo de Fin de Grado (TFG) de Lorena y trata sobre la detección de contenido explícito en letras de canciones mediante modelos de lenguaje.

## Objetivo del TFG
El objetivo principal es evaluar distintos enfoques para clasificar letras de canciones como:

- `explicit`
- `clean`

La etiqueta de referencia es la etiqueta `explicit` proporcionada por Spotify.

## Dataset
El dataset principal es `lyrics_10k.csv`, con aproximadamente 10.000 canciones.

Cada fila contiene información como:
- título de la canción
- artista
- letra completa
- géneros musicales
- etiqueta real `explicit` (`True/False`)

### Distribución de clases
El dataset está desbalanceado:
- aproximadamente 78,6% canciones limpias
- aproximadamente 21,4% canciones explícitas

Esto implica que la accuracy por sí sola no es suficiente para evaluar correctamente los modelos.

## Naturaleza de la tarea
Se trata de una tarea de clasificación binaria sobre texto largo.

La tarea consiste en predecir si una letra debe considerarse explícita o no explícita.

El contenido explícito puede incluir, entre otros:
- palabrotas
- insultos
- referencias sexuales explícitas
- lenguaje ofensivo
- violencia gráfica
- slurs o expresiones claramente inapropiadas

## Enfoques que se quieren comparar
El TFG no se centra en un único modelo, sino en la comparación de varias familias de métodos.

### 1. Baseline léxico
Se utilizará un baseline simple basado en expresiones regulares.
La regla general será clasificar una letra como explícita si contiene determinadas palabras malsonantes predefinidas.

No se usará un baseline léxico enriquecido. Solo regex simple.

### 2. Modelos especializados
Se usarán modelos ya fine-tuned por terceros para tareas cercanas como:
- toxicidad
- lenguaje ofensivo
- moderación textual

Estos modelos se consideran "especializados" porque ya han sido ajustados previamente para tareas similares.

Los modelos especializados seleccionados son:
1. `Intel/toxic-prompt-roberta`
2. `unitary/unbiased-toxic-roberta`
3. `s-nlp/roberta_toxicity_classifier`
4. `cardiffnlp/twitter-roberta-base-offensive`

Estos modelos se usarán directamente, sin reentrenamiento adicional sobre el dataset del TFG.

### 3. LLMs generales
También se usarán LLMs generales de propósito amplio como clasificadores mediante prompting.

Estos modelos no están especializados específicamente en detección de contenido explícito, pero pueden utilizarse en:
- zero-shot
- few-shot

Los LLMs generales seleccionados son:
1. `Qwen/Qwen2.5-7B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`
3. `microsoft/Phi-4-mini-instruct`
4. `mistralai/Mistral-7B-Instruct-v0.3`

Importante:
- deben usarse en formato instruct, no base
- se les pedirá que respondan con una etiqueta cerrada, idealmente `explicit` o `clean`
- se evaluarán en zero-shot y few-shot

### 4. Modelos fine-tuned específicamente para la tarea
Además, se entrenarán modelos supervisados sobre el propio dataset del TFG.

Los modelos elegidos son:
1. `google-bert/bert-base-uncased`
2. `FacebookAI/roberta-base`

#### Rol de cada uno
- **BERT**: baseline supervisado clásico
- **RoBERTa**: modelo supervisado principal

No se pretende que BERT sea el mejor modelo del trabajo, sino un baseline sólido, estándar y defendible académicamente.

## Qué no se va a hacer
- No se usará regex enriquecido
- No se plantea como eje principal el fine-tuning de LLMs grandes como Llama, Qwen o Gemma
- Los LLMs generales se usarán principalmente para zero-shot y few-shot
- El fine-tuning principal será sobre modelos encoder clásicos (BERT y RoBERTa)

## Configuraciones de evaluación
Los modelos se evaluarán en dos configuraciones distintas.

### Configuración A: todo el dataset
Se evaluarán algunos métodos sobre el dataset completo para análisis descriptivo y comparación global.

Esto se aplicará principalmente a:
- regex simple
- modelos especializados
- LLMs generales zero-shot
- LLMs generales few-shot

Esta configuración no debe interpretarse como evaluación final de generalización, sino como comparación global exploratoria.

### Configuración B: split estratificado
El dataset se dividirá en:
- 70% train
- 15% validation
- 15% test

La división debe ser estratificada respecto a la etiqueta `explicit`.

Esta será la configuración principal para evaluación formal.

## Métricas
La métrica principal será:
- **F1 de la clase explícita**

También se reportarán:
- accuracy
- precision
- recall
- matriz de confusión

Debido al desbalance del dataset, la accuracy no debe utilizarse como única métrica principal.

## Consideraciones técnicas
- Hay una GPU con 40 GB de VRAM
- No se desea probar una cantidad excesiva de modelos adicionales
- El proyecto debe mantenerse razonablemente manejable a nivel computacional
- El código debe ser claro, modular y reproducible

## Intención científica del TFG
La idea del TFG es comparar cuatro grandes enfoques:

1. reglas simples
2. modelos especializados ya ajustados
3. LLMs generales por prompting
4. modelos adaptados específicamente al dominio mediante fine-tuning

La narrativa experimental debe mostrar:
- qué se consigue con un baseline muy simple
- si la especialización previa transfiere bien a letras de canciones
- si los LLMs generales pueden resolver la tarea sin entrenamiento específico
- si el fine-tuning supervisado sobre el dataset del dominio mejora los resultados