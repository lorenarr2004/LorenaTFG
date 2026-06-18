# ============================================================
# Análisis de interpretabilidad / errores por familia de modelos
# ------------------------------------------------------------
# Para el MEJOR modelo de cada familia (regex, roberta-toxic,
# llama, qwen, bert-finetuned) calcula, para cada canción del
# split de test:
#   - qué modelos han acertado y cuáles han fallado
#   - cuántos modelos han fallado (0 = nadie falla ... N = todos fallan)
# Y agrupa las canciones por nº de modelos que fallan, indicando
# siempre QUIÉN ha fallado, para poder analizar las letras y
# formular hipótesis (ambigüedad, sentido figurado, etc.).
#
# Salidas (en output/interpretability/):
#   - per_song.csv        -> matriz completa canción x modelo + letra
#   - summary.md          -> resumen legible por nº de fallos
#   - all_failed.csv      -> canciones que fallan TODOS los modelos
#   - all_but_one.csv     -> canciones donde acierta solo 1 modelo
#
# Uso:
#   python -m src.error_analysis            # selección automática por F1
#   python -m src.error_analysis --metric accuracy
# ============================================================
import argparse
import glob
import json
import os

import pandas as pd

from src.data_loading import get_splits, load_dataset
from src.utils import load_config

# Familias -> "alias" de cada candidato (nombre base del fichero,
# sin _split_test). Dentro de cada familia se elige el mejor por métrica.
FAMILIES = {
    "regex": ["regex"],
    "roberta": ["toxic-roberta", "unbiased-toxic", "snlp-toxicity", "twitter-offensive"],
    "llama": ["llama_zeroshot"] + [f"llama_kshot_k{k}" for k in range(2, 22, 2)],
    "qwen": ["qwen_zeroshot"] + [f"qwen_kshot_k{k}" for k in range(2, 22, 2)],
    "bert": ["bert_finetuned"],
}

CONFIG = "split_test"


def pick_best_per_family(metrics_dir, metric):
    """Para cada familia devuelve (alias_elegido, valor_metrica)."""
    chosen = {}
    for family, candidates in FAMILIES.items():
        best_alias, best_val = None, -1.0
        for alias in candidates:
            path = os.path.join(metrics_dir, f"{alias}_{CONFIG}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                m = json.load(f)
            val = m.get(metric)
            if val is not None and val > best_val:
                best_alias, best_val = alias, val
        if best_alias is None:
            print(f"  [!] Familia '{family}': ningún candidato encontrado, se omite.")
            continue
        chosen[family] = (best_alias, best_val)
    return chosen


def load_predictions(predictions_dir, alias):
    """Devuelve un DataFrame indexado por 'index' con columnas label/prediction."""
    path = os.path.join(predictions_dir, f"{alias}_{CONFIG}.csv")
    df = pd.read_csv(path, usecols=lambda c: c in ("index", "label", "prediction"))
    return df.set_index("index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Ruta al config.yaml"
    )
    parser.add_argument(
        "--metric",
        default="f1_explicit",
        help="Métrica para elegir el mejor modelo de cada familia "
        "(f1_explicit, accuracy, precision_explicit, recall_explicit)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictions_dir = cfg["predictions_dir"]
    metrics_dir = cfg["metrics_dir"]
    out_dir = os.path.join(cfg["output_dir"], "interpretability")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Elegir el mejor modelo de cada familia ----------------------------
    print(f"Seleccionando mejor modelo por familia (métrica: {args.metric})")
    chosen = pick_best_per_family(metrics_dir, args.metric)
    for family, (alias, val) in chosen.items():
        print(f"  {family:10s} -> {alias:22s} ({args.metric}={val:.4f})")

    family_to_alias = {f: a for f, (a, _) in chosen.items()}
    models = list(family_to_alias.keys())  # nombres de familia = columnas

    # 2) Cargar predicciones y alinear por index ---------------------------
    label_series = None
    pred_cols = {}
    for family, alias in family_to_alias.items():
        df = load_predictions(predictions_dir, alias)
        pred_cols[family] = df["prediction"]
        if label_series is None:
            label_series = df["label"]

    data = pd.DataFrame({"label": label_series})
    for family in models:
        data[f"pred_{family}"] = pred_cols[family]

    # Verificar alineación (todos los ficheros deberían cubrir los mismos index)
    if data.isna().any().any():
        n_bad = data.isna().any(axis=1).sum()
        print(f"  [!] {n_bad} índices con predicciones ausentes en alguna familia.")

    # 3) Recuperar la letra completa desde el split de test ----------------
    df_full = load_dataset(cfg)
    _, _, test_df = get_splits(df_full, cfg)
    test_df = test_df.reset_index(drop=True)  # index 0..N-1 = index de predicciones
    text_col = cfg["text_column"]

    data["song"] = test_df["song"].reindex(data.index).values
    data["artists"] = test_df["artists"].reindex(data.index).values
    data["lyrics"] = test_df[text_col].reindex(data.index).values

    # 4) Acierto / fallo por modelo ----------------------------------------
    for family in models:
        # correct = 1 si la predicción coincide con la etiqueta real
        data[f"ok_{family}"] = (data[f"pred_{family}"] == data["label"]).astype(int)

    ok_cols = [f"ok_{m}" for m in models]
    data["n_failed"] = len(models) - data[ok_cols].sum(axis=1)

    # Lista de quién falla y quién acierta (legible)
    def who(row, want_fail):
        return ",".join(
            m for m in models if (row[f"ok_{m}"] == 0) == want_fail
        )

    data["models_failed"] = data.apply(lambda r: who(r, want_fail=True), axis=1)
    data["models_ok"] = data.apply(lambda r: who(r, want_fail=False), axis=1)

    # Dirección del error de los modelos que fallan en esa canción.
    # Como todos comparten la misma etiqueta real, el tipo de error lo fija label:
    #   label=1 (explicit) y se falla -> FN (falso negativo: explícita no detectada)
    #   label=0 (clean)    y se falla -> FP (falso positivo: clean marcada explícita)
    def err_type(row):
        if row["n_failed"] == 0:
            return ""  # nadie falla
        return "FN" if row["label"] == 1 else "FP"

    data["error_type"] = data.apply(err_type, axis=1)

    # 5) Guardar matriz completa -------------------------------------------
    col_order = (
        ["label", "n_failed", "error_type", "models_failed", "models_ok"]
        + [f"pred_{m}" for m in models]
        + ["song", "artists", "lyrics"]
    )
    per_song = data[col_order].sort_values("n_failed", ascending=False)
    per_song.to_csv(os.path.join(out_dir, "per_song.csv"), index_label="index")

    # 6) Resumen por nº de fallos ------------------------------------------
    n_models = len(models)
    lines = []
    lines.append("# Análisis de errores por familia de modelos\n")
    lines.append(f"- Métrica de selección del mejor modelo por familia: `{args.metric}`")
    lines.append(f"- Split analizado: `{CONFIG}`  ({len(data)} canciones)\n")
    lines.append("## Modelos seleccionados\n")
    lines.append("| Familia | Modelo elegido | " + args.metric + " |")
    lines.append("|---|---|---|")
    for family, (alias, val) in chosen.items():
        lines.append(f"| {family} | `{alias}` | {val:.4f} |")
    lines.append("")

    # Falsos positivos / falsos negativos por modelo (clase positiva = explicit = 1)
    lines.append("## Falsos positivos y falsos negativos por modelo\n")
    lines.append(
        "- **FP** (falso positivo): canción *clean* (label=0) clasificada como *explicit*.\n"
        "- **FN** (falso negativo): canción *explicit* (label=1) clasificada como *clean*.\n"
    )
    n_pos = int((data["label"] == 1).sum())
    n_neg = int((data["label"] == 0).sum())
    lines.append(f"- Reales en test: {n_pos} explicit (1), {n_neg} clean (0).\n")
    lines.append("| Modelo | FP | FN | Errores totales |")
    lines.append("|---|---|---|---|")
    for m in models:
        fp = int(((data[f"ok_{m}"] == 0) & (data["label"] == 0)).sum())
        fn = int(((data[f"ok_{m}"] == 0) & (data["label"] == 1)).sum())
        lines.append(f"| {m} | {fp} | {fn} | {fp + fn} |")
    lines.append("")

    lines.append("## Distribución por nº de modelos que fallan\n")
    lines.append("| nº fallos | nº canciones | % | de los cuales FP | FN |")
    lines.append("|---|---|---|---|---|")
    dist = data["n_failed"].value_counts().sort_index()
    for n in range(n_models + 1):
        sub = data[data["n_failed"] == n]
        cnt = len(sub)
        pct = 100.0 * cnt / len(data)
        fp = int((sub["error_type"] == "FP").sum())
        fn = int((sub["error_type"] == "FN").sum())
        lines.append(f"| {n} | {cnt} | {pct:.1f}% | {fp} | {fn} |")
    lines.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    # 7) Subconjuntos clave para análisis cualitativo ----------------------
    # Columnas útiles para leer las letras
    detail_cols = ["label", "n_failed", "error_type", "models_failed", "song", "artists", "lyrics"]

    all_failed = data[data["n_failed"] == n_models][detail_cols]
    all_failed.to_csv(os.path.join(out_dir, "all_failed.csv"), index_label="index")

    all_but_one = data[data["n_failed"] == n_models - 1][detail_cols]
    all_but_one.to_csv(os.path.join(out_dir, "all_but_one.csv"), index_label="index")

    print("\nResumen rápido:")
    print(f"  Canciones totales (test): {len(data)}")
    print(f"  Fallan TODOS ({n_models}): {len(all_failed)}")
    print(f"  Fallan todos menos 1:     {len(all_but_one)}")
    print(f"\nFicheros generados en: {out_dir}/")
    for fn in ["per_song.csv", "summary.md", "all_failed.csv", "all_but_one.csv"]:
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
