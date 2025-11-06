from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


def compute_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, Dict[str, object]]:
    """Compute centroids for each attack technique (label==1) and optional negative centroid."""
    technique_centroids: Dict[str, Dict[str, object]] = {}

    positives = df[df["label"] == 1]
    if positives.empty:
        logger.warning("No positive samples found for centroid computation.")
        return technique_centroids

    for technique, group in positives.groupby("attack_technique"):
        rows = group["row_id"].to_numpy()
        vectors = embeddings[rows]
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            logger.warning("Centroid for technique %s has zero norm, skipping", technique)
            continue
        centroid = centroid / norm
        distances = 1 - np.dot(vectors, centroid)
        technique_centroids[technique] = {
            "centroid": centroid.astype(np.float32),
            "n_examples": int(len(vectors)),
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
        }
        logger.info(
            "Computed centroid for %s with %d examples", technique, len(vectors)
        )

    negatives = df[df["label"] == 0]
    if not negatives.empty:
        vectors = embeddings[negatives["row_id"].to_numpy()]
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
            technique_centroids["__negative__"] = {
                "centroid": centroid.astype(np.float32),
                "n_examples": int(len(vectors)),
            }
            logger.info("Stored global negative centroid")
    return technique_centroids


def save_centroids(centroids: Dict[str, Dict[str, object]]) -> None:
    config.CENTROIDS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(centroids, config.CENTROIDS_FILE)
    logger.info("Saved centroids to %s", config.CENTROIDS_FILE)


def load_centroids(path: str = None) -> Dict[str, Dict[str, object]]:
    path = path or config.CENTROIDS_FILE
    centroids = joblib.load(path)
    return centroids


def update_centroid(existing: Dict[str, Dict[str, object]], technique: str, embeddings: np.ndarray) -> Dict[str, Dict[str, object]]:
    technique_entry = existing.get(technique)
    new_centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(new_centroid)
    if norm == 0:
        raise ValueError("Cannot create centroid from zero vectors")
    new_centroid = new_centroid / norm

    if technique_entry:
        n_old = technique_entry.get("n_examples", len(embeddings))
        centroid_old = technique_entry["centroid"]
        n_new = len(embeddings)
        combined = centroid_old * n_old + new_centroid * n_new
        combined_norm = np.linalg.norm(combined)
        if combined_norm == 0:
            centroid_final = new_centroid
        else:
            centroid_final = combined / combined_norm
        technique_entry["centroid"] = centroid_final.astype(np.float32)
        technique_entry["n_examples"] = int(n_old + n_new)
    else:
        existing[technique] = {
            "centroid": new_centroid.astype(np.float32),
            "n_examples": int(len(embeddings)),
        }
    return existing


