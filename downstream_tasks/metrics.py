import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# -------------------------
# Helpers
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_confusion_matrix(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception:
        tn = fp = fn = tp = 0
    return tn, fp, fn, tp


# -------------------------
# Binary classification
# -------------------------
def compute_binary_metrics(targets, logits):
    targets = np.asarray(targets)
    logits = np.asarray(logits)

    probs = sigmoid(logits)
    preds = (probs > 0.5).astype(int)

    tn, fp, fn, tp = safe_confusion_matrix(targets, preds)

    return {
        "accuracy": accuracy_score(targets, preds),
        "sensitivity": recall_score(targets, preds, pos_label=1),
        "specificity": recall_score(targets, preds, pos_label=0),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "roc_auc": roc_auc_score(targets, probs),
        "auprc": average_precision_score(targets, probs),
        "f1_score": f1_score(targets, preds),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# -------------------------
# Multiclass classification
# -------------------------
def compute_multiclass_metrics(targets, logits):
    targets = np.asarray(targets)
    logits = np.asarray(logits)

    preds = np.argmax(logits, axis=1)

    # convert one-hot targets if needed
    if targets.ndim > 1:
        targets = np.argmax(targets, axis=1)

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    return {
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "roc_auc": roc_auc_score(targets, probs, multi_class="ovr"),
        "f1_score": f1_score(targets, preds, average="weighted"),
        "confusion_matrix": confusion_matrix(targets, preds),
    }


# -------------------------
# Regression (echo)
# -------------------------
def compute_regression_metrics(targets, preds):
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    return {
        "mae": mean_absolute_error(targets, preds),
        "rmse": np.sqrt(mean_squared_error(targets, preds)),
        "r2": r2_score(targets, preds),
    }


# -------------------------
# Unified entry point
# -------------------------
def compute_metrics(targets, logits, task="binary"):
    """
    task: "binary" | "multiclass" | "regression" | "auto"
    """

    targets = np.asarray(targets)
    logits = np.asarray(logits)

    if task == "regression":
        return compute_regression_metrics(targets, logits)

    if task == "binary":
        return compute_binary_metrics(targets, logits)

    if task == "multiclass":
        return compute_multiclass_metrics(targets, logits)

    # ---- auto detect ----
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return compute_binary_metrics(targets, logits)
    else:
        return compute_multiclass_metrics(targets, logits)


# -------------------------
# Track best metric
# -------------------------
class BestMetricVal:
    def __init__(self, objective="max"):
        if objective not in ["min", "max"]:
            raise ValueError("objective must be 'min' or 'max'")
        self.objective = objective
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
            return True

        if (self.objective == "max" and new_value > self.value) or \
           (self.objective == "min" and new_value < self.value):
            self.value = new_value
            return True

        return False