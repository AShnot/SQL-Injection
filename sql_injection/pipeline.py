from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from . import config
from .centroid_utils import compute_centroids, save_centroids
from .data_loader import load_dataset
from .embedding_manager import EmbeddingManager
from .logging_utils import get_logger
from .preprocessing import iter_windows, window_metadata, lex_query
from .thresholds import ThresholdResult, calibrate_detection_threshold, save_thresholds

logger = get_logger(__name__)


class TrainingArtifacts:
    def __init__(self, embeddings: np.ndarray, ids: np.ndarray, centroids: Dict[str, Dict[str, object]], thresholds: ThresholdResult, report: Dict[str, object]):
        self.embeddings = embeddings
        self.ids = ids
        self.centroids = centroids
        self.thresholds = thresholds
        self.report = report


def ensure_directories() -> None:
    for path in [
        config.ARTIFACTS_DIR,
        config.EMBEDDINGS_DIR,
        config.CENTROIDS_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.LOGS_DIR,
        config.ANALYSIS_DIR,
        config.WINDOW_EMBEDDINGS_DIR,
        config.TOKEN_METADATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_key = df["label"].astype(str) + "_" + df["attack_technique"].astype(str)
    train_val, test = train_test_split(
        df,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=stratify_key,
    )
    stratify_key_train = train_val["label"].astype(str) + "_" + train_val["attack_technique"].astype(str)
    val_size = config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=config.RANDOM_STATE,
        stratify=stratify_key_train,
    )
    logger.info("Dataset split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train.sort_index(), val.sort_index(), test.sort_index()


def build_embeddings(df: pd.DataFrame, manager: EmbeddingManager) -> np.ndarray:
    texts = df["normalized_query"].tolist()
    embeddings = manager.encode(texts)
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, embeddings.astype(np.float32))
    np.save(config.IDS_FILE, df["row_id"].to_numpy(dtype=np.int32))
    logger.info("Saved full-query embeddings with shape %s", embeddings.shape)
    return embeddings


def build_window_embeddings(df: pd.DataFrame, manager: EmbeddingManager) -> None:
    for _, row in df.iterrows():
        row_id = int(row["row_id"])
        tokens = lex_query(row["normalized_query"])
        windows = []
        for start, end, token_window in iter_windows(tokens, config.WINDOW_TOKENS, config.WINDOW_STRIDE):
            meta = window_metadata(token_window, row["normalized_query"])
            if meta["text"]:
                windows.append(meta)
        if not windows:
            continue
        texts = [w["text"] for w in windows]
        embeddings = manager.encode(texts)
        path = config.WINDOW_EMBEDDINGS_DIR / config.WINDOW_EMBED_TEMPLATE.format(row_id=row_id)
        config.WINDOW_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings.astype(np.float32))
        meta_path = config.WINDOW_EMBEDDINGS_DIR / config.WINDOW_META_TEMPLATE.format(row_id=row_id)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(windows, f, indent=2, ensure_ascii=False)
        logger.debug("Saved %d window embeddings for row %d", len(windows), row_id)


def compute_similarity_to_centroids(embeddings: np.ndarray, centroids: Dict[str, Dict[str, object]]) -> Tuple[np.ndarray, List[str]]:
    techniques = [k for k in centroids.keys() if k != "__negative__"]
    if not techniques:
        return np.zeros(len(embeddings)), [""] * len(embeddings)
    centroid_matrix = np.stack([centroids[t]["centroid"] for t in techniques])
    sims = embeddings @ centroid_matrix.T
    best_indices = sims.argmax(axis=1)
    best_similarities = sims.max(axis=1)
    best_labels = [techniques[idx] for idx in best_indices]
    return best_similarities, best_labels


def evaluate_localisation(df: pd.DataFrame, predictions: Dict[int, Dict[str, object]]) -> Dict[str, float]:
    overlaps = []
    for _, row in df.iterrows():
        row_id = int(row["row_id"])
        user_input = row.get("user_inputs", "")
        if not user_input:
            continue
        pred = predictions.get(row_id)
        if not pred:
            continue
        pred_span = pred.get("matched_window", {})
        start = pred_span.get("start_char")
        end = pred_span.get("end_char")
        if start is None or end is None:
            continue
        full_query = row["normalized_query"]
        gt_start = full_query.find(user_input)
        if gt_start == -1:
            continue
        gt_end = gt_start + len(user_input)
        inter_start = max(start, gt_start)
        inter_end = min(end, gt_end)
        intersection = max(0, inter_end - inter_start)
        union = max(end, gt_end) - min(start, gt_start)
        iou = intersection / union if union > 0 else 0.0
        overlaps.append(iou)
    if not overlaps:
        return {}
    return {
        "mean_span_iou": float(np.mean(overlaps)),
        "median_span_iou": float(np.median(overlaps)),
        "exact_match_rate": float(np.mean([1 if o == 1.0 else 0 for o in overlaps])),
    }


def evaluate_model(df_test: pd.DataFrame, embeddings: np.ndarray, centroids: Dict[str, Dict[str, object]], thresholds: ThresholdResult, manager: EmbeddingManager) -> Dict[str, object]:
    sims, best_labels = compute_similarity_to_centroids(embeddings[df_test["row_id"].to_numpy()], centroids)
    preds = (sims >= thresholds.detection_threshold).astype(int)
    y_true = df_test["label"].to_numpy()
    detection_metrics = {
        "precision": metrics.precision_score(y_true, preds, zero_division=0),
        "recall": metrics.recall_score(y_true, preds, zero_division=0),
        "f1": metrics.f1_score(y_true, preds, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, sims) if len(np.unique(y_true)) > 1 else float("nan"),
    }

    pos_mask = y_true == 1
    typing_metrics = {}
    if pos_mask.any():
        true_types = df_test.loc[pos_mask, "attack_technique"].to_numpy()
        pred_types = np.array(best_labels)[pos_mask]
        typing_metrics = {
            "accuracy": metrics.accuracy_score(true_types, pred_types),
            "macro_f1": metrics.f1_score(true_types, pred_types, average="macro", zero_division=0),
            "classification_report": metrics.classification_report(true_types, pred_types, output_dict=True, zero_division=0),
        }

    localisation_predictions = {}
    for pos, (_, row) in enumerate(df_test.iterrows()):
        row_id = int(row["row_id"])
        if preds[pos] == 1:
            technique = best_labels[pos]
            localisation_predictions[row_id] = infer_window_for_row(
                row, technique, centroids, manager
            )

    localisation_metrics = evaluate_localisation(
        df_test[df_test["label"] == 1], localisation_predictions
    )

    report = {
        "detection": detection_metrics,
        "typing": typing_metrics,
        "localisation": localisation_metrics,
    }
    with config.REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", config.REPORT_FILE)
    return report


def infer_window_for_row(row: pd.Series, technique: str, centroids: Dict[str, Dict[str, object]], manager: EmbeddingManager) -> Dict[str, object]:
    if technique not in centroids:
        return {}
    tokens = lex_query(row["normalized_query"])
    windows = []
    for _, _, token_window in iter_windows(tokens, config.WINDOW_TOKENS, config.WINDOW_STRIDE):
        meta = window_metadata(token_window, row["normalized_query"])
        if meta["text"]:
            windows.append(meta)
    if not windows:
        return {}
    embeddings = manager.encode([w["text"] for w in windows])
    centroid = centroids[technique]["centroid"]
    sims = embeddings @ centroid
    idx = int(np.argmax(sims))
    best = windows[idx]
    best["window_score"] = float(sims[idx])
    return {"matched_window": best}


def train_pipeline(device: str | None = None) -> TrainingArtifacts:
    ensure_directories()
    manager = EmbeddingManager(device=device)
    manager.save_metadata()

    df = load_dataset()
    train_df, val_df, test_df = split_dataset(df)

    embeddings = build_embeddings(df, manager)
    build_window_embeddings(df, manager)

    centroids = compute_centroids(train_df, embeddings)
    save_centroids(centroids)
    techniques = [k for k in centroids.keys() if k != "__negative__"]
    if not techniques:
        raise RuntimeError("No positive centroids computed. Ensure the dataset contains label==1 samples in training split.")

    val_embeddings = embeddings[val_df["row_id"].to_numpy()]
    sims, _ = compute_similarity_to_centroids(val_embeddings, centroids)
    thresholds = calibrate_detection_threshold(sims, val_df["label"].to_numpy())
    save_thresholds(thresholds)

    report = evaluate_model(test_df, embeddings, centroids, thresholds, manager)

    return TrainingArtifacts(embeddings, df["row_id"].to_numpy(), centroids, thresholds, report)


