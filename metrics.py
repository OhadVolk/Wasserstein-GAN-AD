from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
from scipy.interpolate import interp1d


def get_auc_roc(y_true: np.ndarray, y_probas: np.ndarray) -> float:
    return roc_auc_score(y_true, y_probas)


def fpr_at_tpr(tpr: np.ndarray, fpr: np.ndarray, threshold: float) -> float:
    return interp1d(tpr, fpr)(threshold)


def get_auc_prc(y_true: np.ndarray, y_probas: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_probas)
    return auc(recall, precision)


def get_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred)


def get_mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return matthews_corrcoef(y_true, y_pred)


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        'auc_roc': get_auc_roc(y_true, y_proba),
        'auc_prc': get_auc_prc(y_true, y_proba),
        'fpr_at_tpr_0.95': fpr_at_tpr(y_true, y_proba, 0.95),
        'f1_score': get_f1_score(y_true, y_pred),
        'mcc_score': get_mcc_score(y_true, y_pred)
    }
    return metrics
