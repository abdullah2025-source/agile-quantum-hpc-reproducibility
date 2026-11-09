"""Shared evaluation metrics — used by every training script.

Definitions (stated explicitly, because the paper does not define
all of them):
- accuracy, sensitivity (recall), specificity, F1, AUC: standard,
  via scikit-learn.
- false_alarm_rate = FP / (FP + TN) = 1 - specificity.
  NOTE: the paper's Table 6 does NOT satisfy this identity
  (e.g. Agile-Quantum specificity 97.8% but false-alarm 6.8%,
  not 2.2%). The manuscript must define false-alarm rate explicitly;
  This function uses
  the standard definition and reports it as such.
- PSNR (dB) for images, with MAX=1.0 for normalised images.
- mean +/- SD across the three fixed seeds 42, 123, 888.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "n": int(len(y_true)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if y_score is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["auc"] = None
    return out


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": [float(v) for v in values]}


def psnr_db(img_true, img_pred, max_val=1.0):
    mse = float(np.mean((np.asarray(img_true, dtype=float) - np.asarray(img_pred, dtype=float)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))


def summarise_runs(run_metrics, keys=("accuracy", "sensitivity", "specificity", "f1", "auc")):
    """Mean +/- SD across runs (seeds 42, 123, 888)."""
    out = {}
    for k in keys:
        vals = [r[k] for r in run_metrics if r.get(k) is not None]
        if vals:
            out[k] = mean_sd(vals)
    return out
