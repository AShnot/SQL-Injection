from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn import metrics

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdResult:
    detection_threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def _get_best_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    unique_scores = np.unique(similarities)
    best_tau = float(unique_scores.mean()) if unique_scores.size else 0.5
    best_f1 = -1.0
    for tau in unique_scores:
        preds = (similarities >= tau).astype(int)
        f1 = metrics.f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    logger.info("Selected detection threshold %.4f with F1=%.4f", best_tau, best_f1)
    return best_tau


def calibrate_detection_threshold(similarities: np.ndarray, labels: np.ndarray) -> ThresholdResult:
    if similarities.size == 0:
        raise ValueError("No similarities provided for calibration")
    tau = _get_best_threshold(similarities, labels)
    preds = (similarities >= tau).astype(int)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)
    try:
        roc_auc = metrics.roc_auc_score(labels, similarities)
    except ValueError:
        roc_auc = float("nan")
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, similarities)
    pr_auc = metrics.auc(recall_curve, precision_curve)
    return ThresholdResult(
        detection_threshold=float(tau),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def save_thresholds(result: ThresholdResult) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with config.THRESHOLDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2)
    logger.info("Saved thresholds to %s", config.THRESHOLDS_FILE)


