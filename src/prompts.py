# ============================================================
# Plantillas de prompts para LLMs generales
# ============================================================

# --- Prompt zero-shot ---
ZERO_SHOT_PROMPT = """Classify the following song lyrics as either `explicit` or `clean`.
Label as `explicit` if the lyrics contain profanity, offensive slurs, explicit sexual references, or clearly inappropriate violent/offensive language.
Otherwise label as `clean`.
Reply with exactly one word: `explicit` or `clean`.

Lyrics:
{lyrics}

Response:"""

# --- Prompt few-shot ---
# Los ejemplos se insertan dinámicamente desde el train set
FEW_SHOT_TEMPLATE = """Classify the following song lyrics as either `explicit` or `clean`.
Label as `explicit` if the lyrics contain profanity, offensive slurs, explicit sexual references, or clearly inappropriate violent/offensive language.
Otherwise label as `clean`.
Reply with exactly one word: `explicit` or `clean`.

Here are some examples:

{examples}
Now classify:

Lyrics:
{lyrics}

Response:"""

# Formato para cada ejemplo few-shot
FEW_SHOT_EXAMPLE = """Lyrics:
{lyrics}

Response: {label}"""


def build_zero_shot_prompt(lyrics_text: str) -> str:
    """Construye el prompt zero-shot para una letra."""
    return ZERO_SHOT_PROMPT.format(lyrics=lyrics_text)


def build_few_shot_prompt(lyrics_text: str, examples: list[dict]) -> str:
    """
    Construye el prompt few-shot.
    examples: lista de dicts con claves 'text' y 'label_str' ("explicit"/"clean")
    """
    examples_str = "\n\n".join(
        FEW_SHOT_EXAMPLE.format(lyrics=ex["text"][:500], label=ex["label_str"])
        for ex in examples
    )
    return FEW_SHOT_TEMPLATE.format(examples=examples_str, lyrics=lyrics_text)
