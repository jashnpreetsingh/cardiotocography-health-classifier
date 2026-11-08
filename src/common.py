from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
REPORT_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "ctg_raw.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

ORIGINAL_TARGET = "NSP"
TARGET = "fetal_risk"
CLASS_NAMES = ["Normal", "At risk"]

FEATURES = [
    "LB", "AC", "FM", "UC", "DL", "DS", "DP",
    "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    return df

def add_binary_target(df):
    out = df.copy()
    out[TARGET] = (out[ORIGINAL_TARGET] != 1).astype(int)
    return out.drop(columns=[ORIGINAL_TARGET])

def compute_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
    }

def df_to_markdown(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for i in range(len(df)):
        row = df.iloc[i]
        row_vals = []
        for val in row:
            if isinstance(val, float):
                row_vals.append(f"{val:.4f}")
            else:
                row_vals.append(str(val))
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)