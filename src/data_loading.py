# ============================================================
# Carga del dataset y split estratificado
# ============================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config, set_seed


def load_dataset(cfg):
    """
    Carga el CSV, limpia filas inválidas y normaliza la etiqueta
    a formato binario: 1 = explicit, 0 = clean.
    """
    df = pd.read_csv(cfg["dataset_path"])

    text_col = cfg["text_column"]
    label_col = cfg["label_column"]

    # Eliminar filas sin texto o sin etiqueta
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    # Normalizar etiqueta a int: True/1 -> 1, False/0 -> 0
    df["label"] = df[label_col].map({True: 1, False: 0, "True": 1, "False": 0})
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    return df


def get_splits(df, cfg):
    """
    Divide el dataset en train/val/test con estratificación.
    Devuelve tres DataFrames.
    """
    seed = cfg["seed"]
    set_seed(seed)

    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    # test_ratio es el resto

    # Primer split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df["label"], random_state=seed
    )

    # Segundo split: val vs test (proporcional dentro del 30% restante)
    val_relative = val_ratio / (val_ratio + cfg["test_ratio"])
    val_df, test_df = train_test_split(
        temp_df, train_size=val_relative, stratify=temp_df["label"], random_state=seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df
